#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from common.policies.diffusion.configuration_diffusion import DiffusionConfig
from common.policies.normalize import Normalize, Unnormalize
from common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
)
import time

class DiffusionPolicy(nn.Module, PyTorchModelHubMixin):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        alignment_strategy: str = 'post-hoc',
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = DiffusionConfig()
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config, alginment_strategy=alignment_strategy)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.use_env_state = "observation.environment_state" in config.input_shapes

    @property
    def n_obs_steps(self) -> int:
        return self.config.n_obs_steps

    @property
    def input_keys(self) -> set[str]:
        return set(self.config.input_shapes)

    @torch.no_grad
    def run_inference(self, observation_batch: dict[str, Tensor], guide: Tensor | None = None, visualizer=None) -> Tensor:
        observation_batch = self.normalize_inputs(observation_batch)
        if guide is not None:
            guide = self.normalize_targets({"action": guide})["action"]
        if len(self.expected_image_keys) > 0:
            observation_batch["observation.images"] = torch.stack(
                [observation_batch[k] for k in self.expected_image_keys], dim=-4
            )
        actions = self.diffusion.generate_actions(observation_batch, guide=guide, visualizer=visualizer, normalizer=self)
        actions = self.unnormalize_outputs({"action": actions})["action"]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch)
        return {"loss": loss}


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig, alginment_strategy: str):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = config.input_shapes["observation.state"][0]
        num_images = len([k for k in config.input_shapes if k.startswith("observation.image")])
        self._use_images = False
        self._use_env_state = False
        if num_images > 0:
            self._use_images = True
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if "observation.environment_state" in config.input_shapes:
            self._use_env_state = True
            global_cond_dim += config.input_shapes["observation.environment_state"][0]

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

        assert alginment_strategy in ['post-hoc', 'guided-diffusion', 'stochastic-sampling', 'biased-initialization', 'output-perturb'], 'Invalid alignment strategy: ' + str(alginment_strategy)
        self.alignment_strategy = alginment_strategy



    # ================= SMC ===============
    def guide_energy(self, traj: torch.Tensor, guide: torch.Tensor, beta: float = 60.0):
        """Example guide energy function for a L2 norm.
        traj  : (B, H, d) predicted actions
        guide : (1, H, d) or (B, H, d) guide trajectory
        returns U(x)  shape (B,)
        """
        # broadcast guide if needed
        if guide.dim() == 2:  # (H,d)
            guide = guide.unsqueeze(0)
        assert guide.shape[1:] == traj.shape[1:]

        l2 = ((traj - guide) ** 2).sum(dim=-1)          # (B, H)
        xi = l2.mean(dim=-1)                            # (B,)
        return beta * xi 

    def SMC_sampling(
        self,                   
        batch_size,             
        K_particles = 64,      
        global_cond   = None,
        guide          = None,
        generator      = None,
        normalizer     = None,
        visualizer     = None,
    ):
        """Sequential Monte‑Carlo product‑of‑experts sampler for diffusion guidance.
        Args:
            batch_size:       = number of independent *trajectories* 
            K_particles:      = particles per trajectory
            global_cond:      = global conditioning
            guide:            = guide trajectory
            generator:        = torch.Generator
            normalizer:       = normalizer

        Returns:
            final_samples:      (batch_size, horizon, action_dim)
            log_Z_estimate:     unbiased log‑normalising‑constant estimate (optional)
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        cfg    = self.config

        # ---- initialise particle cloud at noise level N ----------------------
        horizon, a_dim = cfg.horizon, cfg.output_shapes["action"][0]
        shape = (batch_size, K_particles, horizon, a_dim)
        particles = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        logw      = torch.zeros(batch_size, K_particles, device=device)   # log‑weights

        # pre‑compute β so guide_energy gives U(x)=β ξ(x,z)
        if guide is not None:
            beta = 60.0 if self.alignment_strategy == "stochastic-sampling" else 20.0
        else:
            beta = 0.0

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        timesteps = list(self.noise_scheduler.timesteps)

        # ---- iterate over noise levels  --------------------------------------
        for t_idx, t in enumerate(timesteps):

            # one Euler reverse‑diffusion step for *all* particles in parallel
            flat_part = particles.view(-1, horizon, a_dim)               # merge batch & K
            model_out = self.unet(
                flat_part,
                torch.full(flat_part.shape[:1], t, dtype=torch.long, device=device),
                global_cond=global_cond.repeat_interleave(K_particles, 0) if global_cond is not None else None,
            ).view_as(particles)

            # add alignment guidance to the score
            if guide is not None and beta > 0:
                grad = self.guide_gradient(particles, guide)            # same shape
                model_out = model_out + beta * grad

            scheduler_out = self.noise_scheduler.step(model_out.view(-1, horizon, a_dim),
                                                    t,
                                                    flat_part,
                                                    generator=generator)
            x_next = scheduler_out.prev_sample.view_as(particles)        # proposal

            # incremental log‑weights  log w_i = -β ξ(x_next) + β ξ(x_prev)
            if guide is not None and beta > 0:
                U_prev = beta * self.guide_energy(particles, guide)     # (B,K)
                U_next = beta * self.guide_energy(x_next,  guide)
                logw  += -(U_next - U_prev)

            particles = x_next

            # ----------- normalise & resample  (systematic) -------------------
            w = (logw - logw.max(dim=1, keepdim=True).values).exp()      # avoid inf
            w = w / w.sum(dim=1, keepdim=True)                          # (B,K)

            # ESS diagnostic (optional) .......................................
            ESS = 1. / (w**2).sum(dim=1)                                 # (B,)
            # if ESS too small you could increase K adaptively

            # resample
            indices = torch.multinomial(w, num_samples=K_particles, replacement=True)
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, K_particles)
            particles = particles[batch_idx, indices]
            logw      = torch.zeros_like(logw)         # reset weights after multinomial

            # (optional) real‑time viz
            if visualizer is not None and normalizer is not None:
                best = particles[:,0]                  # show first particle from each traj
                viz = normalizer.unnormalize_outputs({"action": best})["action"].cpu().numpy()
                visualizer.update_screen(viz, keep_drawing=True)
                time.sleep(0.05)

        # return one representative sample per trajectory (first particle)
        final = particles[:,0]                         # or w‑resampled mean
        return final

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None, guide: Tensor | None = None, visualizer=None, normalizer=None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.output_shapes["action"][0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        if guide is not None and self.alignment_strategy == 'biased-initialization':
            indices = torch.linspace(0, guide.shape[0]-1, sample.shape[1], dtype=int)
            init_sample = torch.unsqueeze(guide[indices], dim=0) # (1, pred_horizon, action_dim)
            init_noise_std = 0.5
            sample = init_noise_std * sample + init_sample
            # return sample

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        MCMC_steps = 1
        if guide is not None and self.alignment_strategy == 'stochastic-sampling':
            MCMC_steps = 4

        start_influence_step = self.config.num_train_timesteps
        if guide is not None and self.alignment_strategy == 'biased-initialization':
            start_influence_step = 50

        final_influence_step = self.config.num_train_timesteps
        if self.alignment_strategy in ['guided-diffusion', 'stochastic-sampling']:
            final_influence_step = 0 

        for t in self.noise_scheduler.timesteps:
            if visualizer is not None and normalizer is not None:
                sample_viz = normalizer.unnormalize_outputs({"action": sample.clone().detach()})["action"]
                sample_viz = sample_viz.cpu().numpy()
                visualizer.update_screen(sample_viz, keep_drawing=True)
                time.sleep(0.1)

            if t > start_influence_step:
                # print('SKIPPING TIMESTEP: ', t)
                continue
            for i in range(MCMC_steps):
                # Predict model output.
                model_output = self.unet(
                    sample,
                    torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                    global_cond=global_cond,
                )

                # add interaction gradient
                if guide is not None and t > final_influence_step:
                    grad = self.guide_gradient(sample, guide)
                    if self.alignment_strategy == 'guided-diffusion':
                        guide_ratio = 20 
                    elif self.alignment_strategy == 'stochastic-sampling':
                        guide_ratio = 60 
                    else:
                        guide_ratio = 0
                    model_output = model_output + guide_ratio * grad
                else:
                    pass
                    # print('NOT ADDING INTERACTION GRADIENT AT TIMESTEP: ', t)

                # Compute previous image: x_t -> x_t-1
                scheduler_output = self.noise_scheduler.step(model_output, t, sample, generator=generator)
                prev_sample = scheduler_output.prev_sample
                clean_sample = scheduler_output.pred_original_sample

                if i < MCMC_steps - 1:
                    # print('mcmc step i: ', i, 'at t: ', t)
                    std = 1
                    noise = std * torch.randn(clean_sample.shape, device=clean_sample.device)
                    sample = self.noise_scheduler.add_noise(clean_sample, noise, t)
                else:
                    # print('final diffusion step at t:', t)
                    sample = prev_sample

        return sample
    
    def guide_gradient(self, naction, guide):
        # naction: (B, pred_horizon, action_dim);
        # guide: (guide_horizon, action_dim)
        # print('noisy action shape:', naction.shape, 'guide shape:', guide.shape)
        # print('mean and std of naction', naction.mean(), naction.std())
        # print('mean and std of guide', guide.mean(), guide.std())

        assert naction.shape[2] == 2 and guide.shape[1] == 2
        indices = torch.linspace(0, guide.shape[0]-1, naction.shape[1], dtype=int)
        guide = torch.unsqueeze(guide[indices], dim=0) # (1, pred_horizon, action_dim)
        assert guide.shape == (1, naction.shape[1], naction.shape[2])
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            dist = torch.linalg.norm(naction - guide, dim=2, ord=2) # (B, pred_horizon)
            dist = dist.mean(dim=1) # (B,)
            grad = torch.autograd.grad(dist, naction, grad_outputs=torch.ones_like(dist), create_graph=False)[0]
            # naction.detach()
        return grad    

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond_feats = [batch["observation.state"]]
        # Extract image feature (first combine batch, sequence, and camera index dims).
        if self._use_images:
            img_features = self.rgb_encoder(
                einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
            )
            # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )
            global_cond_feats.append(img_features)

        if self._use_env_state:
            global_cond_feats.append(batch["observation.environment_state"])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor], guide: Tensor | None = None, visualizer=None, normalizer=None) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps, f"{n_obs_steps=} {self.config.n_obs_steps=}"

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond, guide=guide, visualizer=visualizer, normalizer=normalizer)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.input_shapes` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.input_shapes`.
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: we have a check in the config class to make sure all images have the same shape.
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
        )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.output_shapes["action"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action"][0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")

        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out