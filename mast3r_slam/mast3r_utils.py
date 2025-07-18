import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching


def load_mast3r(path=None, device="cuda", compile=False):
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device).eval()
    
    if compile:
        print("Compiling model")
        # class _EncodeImageWrapper(torch.nn.Module):
        #     def __init__(self, outer):
        #         super().__init__()
        #         self.outer = outer
        #     def forward(self, image, true_shape):
        #         x, pos = self.outer.patch_embed(image, true_shape=true_shape)
        #         assert self.outer.enc_pos_embed is None
        #         for blk in self.outer.enc_blocks:
        #             x = blk(x, pos)
        #         x = self.outer.enc_norm(x)
        #         return x, pos, None

        # attach the compiled wrapper to the same name
        # model._encode_image = torch.compile(_EncodeImageWrapper(model))
        # model._encode_image = torch.compile(model._encode_image)
        # model._decoder = torch.compile(model._decoder, mode="max-autotune-no-cudagraphs")
        # model._downstream_head = torch.compile(model._downstream_head, mode="max-autotune-no-cudagraphs")
        print("Done compiling model")
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):

    # # pretend to be a batch
    # repeat_times = 4
    # feat1 = feat1.repeat(repeat_times, 1, 1)
    # feat2 = feat2.repeat(repeat_times, 1, 1)
    # pos1 = pos1.repeat(repeat_times, 1, 1)
    # pos2 = pos2.repeat(repeat_times, 1, 1)
    # shape1 = shape1.repeat(repeat_times, 1)
    # shape2 = shape2.repeat(repeat_times, 1)

    with torch.amp.autocast(enabled=False, device_type="cuda"):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        if frame_i.feat is None:
            frame_i.feat, frame_i.pos, _ = model._encode_image(
                frame_i.img, frame_i.img_true_shape
            )
        if frame_j.feat is None:
            frame_j.feat, frame_j.pos, _ = model._encode_image(
                frame_j.img, frame_j.img_true_shape
            )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    """Decode two frames symmetrically

    Args:
        model (AsymmetricMASt3R): MASt3R model
        feat_i (torch.Tensor): Feature of frame i
        pos_i (torch.Tensor): Position of frame i
        feat_j (torch.Tensor): Feature of frame j
        pos_j (torch.Tensor): Position of frame j
        shape_i (torch.Tensor): Shape of frames i, 1 x 2
        shape_j (torch.Tensor): Shape of frames j, 1 x 2

    Returns:
        X (torch.Tensor): 4xbxhxwxc
        C (torch.Tensor): 4xbxhxw
        D (torch.Tensor): 4xbxhxwxc
        Q (torch.Tensor): 4xbxhxw
    """
    # let's do this in batch
    feat1 = torch.cat([feat_i, feat_j], dim=0)
    pos1 = torch.cat([pos_i, pos_j], dim=0)
    shape1 = shape_i 
    
    feat2 = torch.cat([feat_j, feat_i], dim=0)
    pos2 = torch.cat([pos_j, pos_i], dim=0)
    shape2 = shape_j

    # 11, 12, 
    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    N = feat_i.shape[0]
    X = torch.stack((
        res11["pts3d"][:N], 
        res21["pts3d"][:N], 
        res11["pts3d"][N:], 
        res21["pts3d"][N:]
    ), dim=0)
    C = torch.stack((
        res11["conf"][:N], 
        res21["conf"][:N], 
        res11["conf"][N:], 
        res21["conf"][N:]
    ), dim=0)
    D = torch.stack((
        res11["desc"][:N], 
        res21["desc"][:N], 
        res11["desc"][N:], 
        res21["desc"][N:]
    ), dim=0)
    Q = torch.stack((
        res11["desc_conf"][:N], 
        res21["desc_conf"][:N], 
        res11["desc_conf"][N:], 
        res21["desc_conf"][N:]
    ), dim=0)
    
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q
    
    
    # B = feat_i.shape[0]
    # X, C, D, Q = [], [], [], []
    # for b in range(B):
    #     feat1 = feat_i[b][None]
    #     feat2 = feat_j[b][None]
    #     pos1 = pos_i[b][None]
    #     pos2 = pos_j[b][None]
    #     res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
    #     res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
    #     res = [res11, res21, res22, res12]
    #     Xb, Cb, Db, Qb = zip(
    #         *[
    #             (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
    #             for r in res
    #         ]
    #     )
    #     X.append(torch.stack(Xb, dim=0))
    #     C.append(torch.stack(Cb, dim=0))
    #     D.append(torch.stack(Db, dim=0))
    #     Q.append(torch.stack(Qb, dim=0))

    # X, C, D, Q = (
    #     torch.stack(X, dim=1),
    #     torch.stack(C, dim=1),
    #     torch.stack(D, dim=1),
    #     torch.stack(Q, dim=1),
    # )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def mast3r_inference_mono(model, frame):
    if frame.feat is None:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    """Match two frames symmetrically

    Args:
        model (AsymmetricMASt3R): MASt3R model
        feat_i (torch.Tensor): Feature of frame i
        pos_i (torch.Tensor): Position of frame i
        feat_j (torch.Tensor): Feature of frame j
        pos_j (torch.Tensor): Position of frame j
        shape_i (torch.Tensor): Shape of frames i, 1 x 2
        shape_j (torch.Tensor): Shape of frames j, 1 x 2

    Returns:
        idx_i2j (torch.Tensor): Index of frame i to frame j
        idx_j2i (torch.Tensor): Index of frame j to frame i
        valid_match_j (torch.Tensor): Valid match of frame j
        valid_match_i (torch.Tensor): Valid match of frame i
        Qii (torch.Tensor): Quality of frame i to frame j
        Qjj (torch.Tensor): Quality of frame j to frame i
        Qji (torch.Tensor): Quality of frame j to frame i
        Qij (torch.Tensor): Quality of frame i to frame j
    """
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )

@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        if frame_i.feat is None:
            frame_i.feat, frame_i.pos, _ = model._encode_image(
                frame_i.img, frame_i.img_true_shape
            )
        if frame_j.feat is None:
            frame_j.feat, frame_j.pos, _ = model._encode_image(
                frame_j.img, frame_j.img_true_shape
            )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q

# @torch.inference_mode
# def mast3r_match_batch_cache(model, frame_i, frame_j, idx_i2j_init=None):
#     X, C, D, Q = mast3r_decode_symmetric_batch(
#         model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
#     )

#     # Ordering 4xbxhxwxc
#     b = X.shape[1]

#     Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
#     Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
#     Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

#     # Always matching both
#     X11 = torch.cat((Xii, Xjj), dim=0)
#     X21 = torch.cat((Xji, Xij), dim=0)
#     D11 = torch.cat((Dii, Djj), dim=0)
#     D21 = torch.cat((Dji, Dij), dim=0)

#     # tic()
#     idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
#     # toc("Match")

#     # TODO: Avoid this
#     match_b = X11.shape[0] // 2
#     idx_i2j = idx_1_to_2[:match_b]
#     idx_j2i = idx_1_to_2[match_b:]
#     valid_match_j = valid_match_2[:match_b]
#     valid_match_i = valid_match_2[match_b:]

#     return (
#         idx_i2j,
#         idx_j2i,
#         valid_match_j,
#         valid_match_i,
#         Qii.view(b, -1, 1),
#         Qjj.view(b, -1, 1),
#         Qji.view(b, -1, 1),
#         Qij.view(b, -1, 1),
#     )
    
    

#     X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

#     b, h, w = X.shape[:-1]
#     # 2 outputs per inference
#     b = b // 2

#     Dii, Dji = D[:b], D[b:]
#     Xii, Xji = X[:b], X[b:]
#     Cii, Cji = C[:b], C[b:]
#     Qii, Qji = Q[:b], Q[b:]

#     idx_i2j, valid_match_j = matching.match(
#         Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
#     )

#     # How rest of system expects it
#     Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
#     Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
#     Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")
#     # Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
#     Dii, Dji = Dii[0], Dji[0]

#     return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji, Dii, Dji

@torch.inference_mode
def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Dii, Dji = D[:b], D[b:]
    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")
    # Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Dii, Dji = Dii[0], Dji[0]

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji, Dii, Dji


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
