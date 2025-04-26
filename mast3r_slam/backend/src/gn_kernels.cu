// Part of this source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
// Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License
//
// Any modifications made are licensed under the CC BY-NC-SA 4.0 License.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;
typedef std::vector<std::vector<long>> graph_t;
typedef std::vector<torch::Tensor> tensor_list_t;


#define THREADS 256
#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)

#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)

#define EPS 1e-6

__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid +  8];
  sdata[tid] += sdata[tid +  4];
  sdata[tid] += sdata[tid +  2];
  sdata[tid] += sdata[tid +  1];
}

__device__ void blockReduce(volatile float *sdata) {
  unsigned int tid = threadIdx.x;
  __syncthreads();

  // if (threadIdx.x < 256) {sdata[tid] += sdata[tid + 256]; } __syncthreads();
  if (threadIdx.x < 128) {sdata[tid] += sdata[tid + 128]; } __syncthreads();
  if (threadIdx.x <  64) {sdata[tid] += sdata[tid +  64]; } __syncthreads();

  if (tid < 32) warpReduce(sdata, tid);
  __syncthreads();
}

class SparseBlock {
  public:

    Eigen::SparseMatrix<double> A;
    Eigen::VectorX<double> b;
    int N, M;

    SparseBlock(int n, int m) : N(n), M(m) {
      A = Eigen::SparseMatrix<double>(N*M, N*M);
      b = Eigen::VectorXd::Zero(N*M);
    }

    SparseBlock(Eigen::SparseMatrix<double> const& A, Eigen::VectorX<double> const& b, 
        int N, int M) : A(A), b(b), N(N), M(M) {}

    void update_lhs(torch::Tensor As, torch::Tensor ii, torch::Tensor jj) {

      auto As_cpu = As.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);
      auto jj_cpu = jj.to(torch::kCPU).to(torch::kInt64);

      auto As_acc = As_cpu.accessor<double,3>();
      auto ii_acc = ii_cpu.accessor<long,1>();
      auto jj_acc = jj_cpu.accessor<long,1>();

      std::vector<T> tripletList;
      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        const int j = jj_acc[n];

        if (i >= 0 && j >= 0) {
          for (int k=0; k<M; k++) {
            for (int l=0; l<M; l++) {
              double val = As_acc[n][k][l];
              if (val != 0.0) {
                tripletList.emplace_back(M*i + k, M*j + l, val);
              }
            }
          }
        }
      }
      if (!tripletList.empty()) {
        // build a tiny block and add it into A
        Eigen::SparseMatrix<double> block(N*M, N*M);
        block.setFromTriplets(tripletList.begin(), tripletList.end());
        A += block;    // <—— accumulate
      }

    }

    void update_rhs(torch::Tensor bs, torch::Tensor ii) {
      auto bs_cpu = bs.to(torch::kCPU).to(torch::kFloat64);
      auto ii_cpu = ii.to(torch::kCPU).to(torch::kInt64);

      auto bs_acc = bs_cpu.accessor<double,2>();
      auto ii_acc = ii_cpu.accessor<long,1>();

      for (int n=0; n<ii.size(0); n++) {
        const int i = ii_acc[n];
        if (i >= 0) {
          for (int j=0; j<M; j++) {
            b(i*M + j) += bs_acc[n][j];
          }
        }
      }
    }

    SparseBlock operator-(const SparseBlock& S) {
      return SparseBlock(A - S.A, b - S.b, N, M);
    }

    std::tuple<torch::Tensor, torch::Tensor> get_dense() {
      Eigen::MatrixXd Ad = Eigen::MatrixXd(A);

      torch::Tensor H = torch::from_blob(Ad.data(), {N*M, N*M}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      torch::Tensor v = torch::from_blob(b.data(), {N*M, 1}, torch::TensorOptions()
        .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);

      return std::make_tuple(H, v);

    }

    // torch::Tensor solve(const float lm=0.0, const float ep=0.0) {

    //   torch::Tensor dx;

    //   Eigen::SparseMatrix<double> L(A);
    //   L.diagonal().array() += ep + lm * L.diagonal().array();

    //   Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    //   solver.compute(L);

    //   if (solver.info() == Eigen::Success) {
    //     Eigen::VectorXd x = solver.solve(b);
    //     dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
    //       .dtype(torch::kFloat64)).to(torch::kCUDA).to(torch::kFloat32);
    //   }
    //   else {
    //     dx = torch::zeros({N, M}, torch::TensorOptions()
    //       .device(torch::kCUDA).dtype(torch::kFloat32));
    //   }
      
    //   return dx;
    // }
    torch::Tensor solve(const float lm=0.0, const float ep=0.0) {
      torch::Tensor dx;
      Eigen::SparseMatrix<double> L(A); // Copy A.A
      L.diagonal().array() += ep + lm * L.diagonal().array();

      std::cout << "  Solving system with N=" << N << ", M=" << M << ", Size=" << L.rows() << "x" << L.cols() << std::endl;
      std::cout << "  Norm of b = " << b.norm() << std::endl;
      std::cout << "  Matrix L non-zeros = " << L.nonZeros() << std::endl;
      if (L.rows() > 0) {
          std::cout << "  Matrix L diagonal min/max = " << L.diagonal().minCoeff() << "/" << L.diagonal().maxCoeff() << std::endl;
      } else {
            std::cerr << "  Warning: Matrix L is empty!" << std::endl;
      }


      Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(L);

      Eigen::ComputationInfo solve_info = solver.info();
      std::cout << "  Solver info after compute: " << solve_info << std::endl; // Eigen::Success = 0

      if (solve_info == Eigen::Success) {
          Eigen::VectorXd x = solver.solve(b);
          std::cout << "  Solve successful. Norm of x = " << x.norm() << std::endl;
          // Check if x is numerically zero
          if (x.norm() < 1e-12) {
              std::cerr << "  Warning: Solver succeeded but solution norm is near zero." << std::endl;
          }
          dx = torch::from_blob(x.data(), {N, M}, torch::TensorOptions()
            .dtype(torch::kFloat64)).clone().to(torch::kCUDA).to(torch::kFloat32); // Added .clone()
      } else {
          std::cerr << "  Eigen Solver FAILED with code: " << solve_info << std::endl;
          dx = torch::zeros({N, M}, torch::TensorOptions()
            .device(torch::kCUDA).dtype(torch::kFloat32));
      }
      return dx;
  }

};

torch::Tensor get_unique_kf_idx(torch::Tensor ii, torch::Tensor jj) {
  std::tuple<torch::Tensor, torch::Tensor> unique_kf_idx = torch::_unique(torch::cat({ii,jj}), /*sorted=*/ true);
  return std::get<0>(unique_kf_idx);
}

std::vector<torch::Tensor> create_inds(torch::Tensor unique_kf_idx, const int pin, torch::Tensor ii, torch::Tensor jj) {
  torch::Tensor ii_ind = torch::searchsorted(unique_kf_idx, ii) - pin;
  torch::Tensor jj_ind = torch::searchsorted(unique_kf_idx, jj) - pin;
  return {ii_ind, jj_ind};
}

__forceinline__ __device__ float huber(float r) {
  const float r_abs = fabs(r);
  return r_abs < 1.345 ? 1.0 : 1.345 / r_abs;
}

// Returns qi * qj
__device__ void 
quat_comp(const float *qi, const float *qj, float *out) {
  out[0] = qi[3] * qj[0] + qi[0] * qj[3] + qi[1] * qj[2] - qi[2] * qj[1];
  out[1] = qi[3] * qj[1] - qi[0] * qj[2] + qi[1] * qj[3] + qi[2] * qj[0];
  out[2] = qi[3] * qj[2] + qi[0] * qj[1] - qi[1] * qj[0] + qi[2] * qj[3];
  out[3] = qi[3] * qj[3] - qi[0] * qj[0] - qi[1] * qj[1] - qi[2] * qj[2];
}

// Inverts quat
__device__ void 
quat_inv(const float *q, float *out) {
  out[0] = -q[0];
  out[1] = -q[1];
  out[2] = -q[2];
  out[3] =  q[3];
}

__device__ void
actSO3(const float *q, const float *X, float *Y) {
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSim3(const float *t, const float *q, const float *s, const float *X, float *Y) {
  // Rotation
  actSO3(q, X, Y);
  // Scale
  Y[0] *= s[0];
  Y[1] *= s[0];
  Y[2] *= s[0];
  // Translation
  Y[0] += t[0];
  Y[1] += t[1];
  Y[2] += t[2];
}

// Inverts quat
__device__ void 
scale_vec3_inplace(float *t, float s) {
  t[0] *= s;
  t[1] *= s;
  t[2] *= s;
}

__device__ void
crossInplace(const float* a, float *b) {
  float x[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0], 
  };

  b[0] = x[0];
  b[1] = x[1];
  b[2] = x[2];
}

__forceinline__ __device__ float
dot3(const float *t, const float *s) {
  return t[0]*s[0] + t[1]*s[1] + t[2]*s[2];
}

__forceinline__ __device__ float
squared_norm3(const float *v) {
  return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

__device__ void 
relSim3(const float *ti, const float *qi, const float* si,
        const float *tj, const float *qj, const float* sj,
        float *tij, float *qij, float *sij) {
  
  // 1. Setup scale
  float si_inv = 1.0/si[0];
  sij[0] = si_inv * sj[0];

  // 2. Relative rotation
  float qi_inv[4];
  quat_inv(qi, qi_inv);
  quat_comp(qi_inv, qj, qij);

  // 3. Translation
  tij[0] = tj[0] - ti[0];
  tij[1] = tj[1] - ti[1];
  tij[2] = tj[2] - ti[2];
  actSO3(qi_inv, tij, tij);
  scale_vec3_inplace(tij, si_inv);
}

// Order of X,Y is tau, omega, s 
// NOTE: This is applying adj inv on the right to a row vector on the left,
// The equivalent is transposing the adjoint and multiplying a column vector
__device__ void
apply_Sim3_adj_inv(const float *t, const float *q, const float *s, const float *X, float *Y) {
  // float qinv[4] = {-q[0], -q[1], -q[2], q[3]};
  const float s_inv = 1.0/s[0];

  // First component = s_inv R a
  float Ra[3];
  actSO3(q, &X[0], Ra);
  Y[0] = s_inv * Ra[0];
  Y[1] = s_inv * Ra[1];
  Y[2] = s_inv * Ra[2];
  
  // Second component = s_inv [t]x Ra + Rb
  actSO3(q, &X[3], &Y[3]); // Init to Rb
  Y[3] += s_inv*(t[1]*Ra[2] - t[2]*Ra[1]);
  Y[4] += s_inv*(t[2]*Ra[0] - t[0]*Ra[2]);
  Y[5] += s_inv*(t[0]*Ra[1] - t[1]*Ra[0]);

  // Third component = s_inv t^T R a + c
  Y[6] = X[6] + ( s_inv * dot3(t, Ra) );
}

__device__ void
expSO3(const float *phi, float* q) {
  // SO3 exponential map
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];

  float imag, real;

  if (theta_sq < EPS) {
    float theta_p4 = theta_sq * theta_sq;
    imag = 0.5 - (1.0/48.0)*theta_sq + (1.0/3840.0)*theta_p4;
    real = 1.0 - (1.0/ 8.0)*theta_sq + (1.0/ 384.0)*theta_p4;
  } else {
    float theta = sqrtf(theta_sq);
    imag = sinf(0.5 * theta) / theta;
    real = cosf(0.5 * theta);
  }

  q[0] = imag * phi[0];
  q[1] = imag * phi[1];
  q[2] = imag * phi[2];
  q[3] = real;

}

__device__ void
expSim3(const float *xi, float* t, float* q, float* s) {
  float tau[3] = {xi[0], xi[1], xi[2]};
  float phi[3] = {xi[3], xi[4], xi[5]};
  float sigma = xi[6];

  // New for sim3
  float scale = expf(sigma);

  // 1. Rotation
  expSO3(phi, q);
  // 2. Scale
  s[0] = scale;

  // 3. Translation

  // TODO: Reuse this from expSO3?
  float theta_sq = phi[0]*phi[0] + phi[1]*phi[1] + phi[2]*phi[2];
  float theta = sqrtf(theta_sq);

  // Coefficients for W
  https://github.com/princeton-vl/lietorch/blob/0fa9ce8ffca86d985eca9e189a99690d6f3d4df6/lietorch/include/rxso3.h#L190
  
  // TODO: Does this really match equations? Where is scale-1
  float A, B, C;
  const float one = 1.0;
  const float half = 0.5;
  if (fabs(sigma) < EPS) {
    C = one;
    if (fabs(theta) < EPS) {
      A = half;
      B = 1.0/6.0;
    } else {
      A = (one - cosf(theta)) / theta_sq;
      B = (theta - sinf(theta)) / (theta_sq * theta);
    }
  } else {
    C = (scale - one) / sigma;
    if (fabs(theta) < EPS) {
      float sigma_sq = sigma * sigma;
      A = ((sigma - one) * scale + one) / sigma_sq;
      B = (scale * half * sigma_sq + scale - one - sigma * scale) /
          (sigma_sq * sigma);
    } else {
      float a = scale * sinf(theta);
      float b = scale * cosf(theta);
      float c = theta_sq + sigma * sigma;
      A = (a * sigma + (one - b) * theta) / (theta * c);
      B = (C - ((b - one) * sigma + a * theta) / (c)) / (theta_sq); // Why is it C - ????? not +?
    }
  }

  // W = C * I + A * Phi + B * Phi2;
  // t = W tau
  t[0] = C * tau[0]; 
  t[1] = C * tau[1]; 
  t[2] = C * tau[2];

  crossInplace(phi, tau);
  t[0] += A * tau[0];
  t[1] += A * tau[1];
  t[2] += A * tau[2];

  crossInplace(phi, tau);
  t[0] += B * tau[0];
  t[1] += B * tau[1];
  t[2] += B * tau[2];
}

__device__ void
retrSim3(const float *xi, const float* t, const float* q, const float* s, float* t1, float* q1, float* s1) {
  
  // retraction on Sim3 manifold
  float dt[3] = {0, 0, 0};
  float dq[4] = {0, 0, 0, 1};
  float ds[1] = {0};
  
  expSim3(xi, dt, dq, ds);

  // Compose transformation from left
  // R
  quat_comp(dq, q, q1);
  // t = ds dR R + ds
  actSO3(dq, t, t1);
  scale_vec3_inplace(t1, ds[0]);
  t1[0] += dt[0];
  t1[1] += dt[1];
  t1[2] += dt[2];
  // s
  s1[0] = ds[0] * s[0];
}

__device__ void
retrSim3Right(const float *xi, const float* t, const float* q, const float* s, float* t1, float* q1, float* s1) {

  // Calculate the small transformation delta_T = Exp(xi)
  float dt_local[3] = {0, 0, 0};   // delta_t_local from expSim3
  float dq_delta[4] = {0, 0, 0, 1}; // delta_R from expSim3
  float ds_rel[1] = {0};           // delta_s_rel = exp(delta_s_tangent) from expSim3

  expSim3(xi, dt_local, dq_delta, ds_rel); // xi -> (dt_local, dq_delta, ds_rel)

  // 1. Update Scale: s' = s * delta_s_rel
  s1[0] = s[0] * ds_rel[0];

  // 2. Update Rotation: R' = R * delta_R  => q' = q * dq_delta
  quat_comp(q, dq_delta, q1); // Note the order: q * dq_delta

  // 3. Update Translation: t' = s * R * delta_t_local + t
  float temp_t[3];
  actSO3(q, dt_local, temp_t);  // temp_t = R * dt_local
  scale_vec3_inplace(temp_t, s[0]); // temp_t = s * R * dt_local
  t1[0] = temp_t[0] + t[0];
  t1[1] = temp_t[1] + t[1];
  t1[2] = temp_t[2] + t[2];
}

__global__ void pose_retr_kernel_right(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int num_fix) 
{
  const int num_poses = poses.size(0);

  for (int k=num_fix+threadIdx.x; k<num_poses; k+=blockDim.x) {
    float xi[7], q[4], q1[4], t[3], t1[3], s[1], s1[1];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];

    s[0] = poses[k][7];
    
    for (int n=0; n<7; n++) {
      xi[n] = dx[k-num_fix][n];
    }

    retrSim3Right(xi, t, q, s, t1, q1, s1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];

    poses[k][7] = s1[0];
  }
}

__global__ void pose_retr_kernel(
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> dx,
    const int num_fix) 
{
  const int num_poses = poses.size(0);

  for (int k=num_fix+threadIdx.x; k<num_poses; k+=blockDim.x) {
    float xi[7], q[4], q1[4], t[3], t1[3], s[1], s1[1];

    t[0] = poses[k][0];
    t[1] = poses[k][1];
    t[2] = poses[k][2];

    q[0] = poses[k][3];
    q[1] = poses[k][4];
    q[2] = poses[k][5];
    q[3] = poses[k][6];

    s[0] = poses[k][7];
    
    for (int n=0; n<7; n++) {
      xi[n] = dx[k-num_fix][n];
    }

    retrSim3(xi, t, q, s, t1, q1, s1);

    poses[k][0] = t1[0];
    poses[k][1] = t1[1];
    poses[k][2] = t1[2];

    poses[k][3] = q1[0];
    poses[k][4] = q1[1];
    poses[k][5] = q1[2];
    poses[k][6] = q1[3];

    poses[k][7] = s1[0];
  }
}

__device__ void quat_to_rot(const float* q, float* R) {
    float qx = q[0], qy = q[1], qz = q[2], qw = q[3];
    R[0] = 1 - 2*qy*qy - 2*qz*qz;  // R[0,0]
    R[1] = 2*qx*qy - 2*qz*qw;       // R[0,1]
    R[2] = 2*qx*qz + 2*qy*qw;       // R[0,2]
    R[3] = 2*qx*qy + 2*qz*qw;       // R[1,0]
    R[4] = 1 - 2*qx*qx - 2*qz*qz;  // R[1,1]
    R[5] = 2*qy*qz - 2*qx*qw;       // R[1,2]
    R[6] = 2*qx*qz - 2*qy*qw;       // R[2,0]
    R[7] = 2*qy*qz + 2*qx*qw;       // R[2,1]
    R[8] = 1 - 2*qx*qx - 2*qy*qy;  // R[2,2]
}


__global__ void ray_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_ray,
    const float sigma_dist,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[4];
  float w[4];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
    // Parameters
  const float sigma_ray_inv = 1.0/sigma_ray;
  const float sigma_dist_inv = 1.0/sigma_dist;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // Normalize measurement point
    const float norm2_i = squared_norm3(Xi);
    const float norm1_i = sqrtf(norm2_i);
    const float norm1_i_inv = 1.0/norm1_i;    
    
    float ri[3];
    for (int i=0; i<3; i++) ri[i] = norm1_i_inv * Xi[i];
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // Get predicted point norm
    const float norm2_j = squared_norm3(Xj_Ci);
    const float norm1_j = sqrtf(norm2_j);
    const float norm1_j_inv = 1.0/norm1_j;

    float rj_Ci[3];
    for (int i=0; i<3; i++) rj_Ci[i] = norm1_j_inv * Xj_Ci[i];
 
    // Error (difference in camera rays)
    err[0] = rj_Ci[0] - ri[0];
    err[1] = rj_Ci[1] - ri[1];
    err[2] = rj_Ci[2] - ri[2];
    err[3] = norm1_j - norm1_i; // Distance
 
    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid = 
    valid_match_ind
    & (q > Q_thresh)
    & (ci > C_thresh)
    & (cj > C_thresh);

    // Weight using confidences
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;
    
    const float sqrt_w_ray = valid ? sigma_ray_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_dist = valid ? sigma_dist_inv * sqrtf(conf_weight) : 0;
 
    // Robust weight
    w[0] = huber(sqrt_w_ray * err[0]);
    w[1] = huber(sqrt_w_ray * err[1]);
    w[2] = huber(sqrt_w_ray * err[2]);
    w[3] = huber(sqrt_w_dist * err[3]);
    
    // Add back in sigma
    const float w_const_ray = sqrt_w_ray * sqrt_w_ray;
    const float w_const_dist = sqrt_w_dist * sqrt_w_dist;
    w[0] *= w_const_ray;
    w[1] *= w_const_ray;
    w[2] *= w_const_ray;
    w[3] *= w_const_dist;

    // // print the weights for debugging TODO: remove this
    // if (block_id == 0) {
    //   printf("err: %f, %f, %f, %f\n", err[0], err[1], err[2], err[3]);
    //   printf("w: %f, %f, %f, %f\n", w[0], w[1], w[2], w[3]);
    // }
 
    // Jacobians
    
    const float norm3_j_inv = norm1_j_inv / norm2_j;
    const float drx_dPx = norm1_j_inv - Xj_Ci[0]*Xj_Ci[0]*norm3_j_inv;
    const float dry_dPy = norm1_j_inv - Xj_Ci[1]*Xj_Ci[1]*norm3_j_inv;
    const float drz_dPz = norm1_j_inv - Xj_Ci[2]*Xj_Ci[2]*norm3_j_inv;
    const float drx_dPy = - Xj_Ci[0]*Xj_Ci[1]*norm3_j_inv;
    const float drx_dPz = - Xj_Ci[0]*Xj_Ci[2]*norm3_j_inv;
    const float dry_dPz = - Xj_Ci[1]*Xj_Ci[2]*norm3_j_inv;
 
    // rx coordinate
    Ji[0] = drx_dPx;
    Ji[1] = drx_dPy;
    Ji[2] = drx_dPz;
    Ji[3] = 0.0;
    Ji[4] = rj_Ci[2]; // z
    Ji[5] = -rj_Ci[1]; // -y
    Ji[6] = 0.0; // x

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // ry coordinate
    Ji[0] = drx_dPy; // same as drx_dPy
    Ji[1] = dry_dPy;
    Ji[2] = dry_dPz;
    Ji[3] = -rj_Ci[2]; // -z
    Ji[4] = 0.0;
    Ji[5] = rj_Ci[0]; // x
    Ji[6] = 0.0; // y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // rz coordinate
    Ji[0] = drx_dPz; // same as drz_dPX
    Ji[1] = dry_dPz; // same as drz_dPy
    Ji[2] = drz_dPz;
    Ji[3] = rj_Ci[1]; // y
    Ji[4] = -rj_Ci[0]; // -x
    Ji[5] = 0.0;
    Ji[6] = 0.0; // z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }


    // dist coordinate
    Ji[0] = rj_Ci[0];
    Ji[1] = rj_Ci[1]; 
    Ji[2] = rj_Ci[2];
    Ji[3] = 0.0; 
    Ji[4] = 0.0; 
    Ji[5] = 0.0;
    Ji[6] = norm1_j;
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[3] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[3] * err[3] * Ji[n];
      vj[n] += w[3] * err[3] * Jj[n];
    }
 
 
  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}

std::vector<torch::Tensor> gauss_newton_rays_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_ray,
  const float sigma_dist,
  const float C_thresh,
  const float Q_thresh,
  const int num_fix,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    ray_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_ray, sigma_dist, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}


// helper functions
// Vec to Skew-symmetric matrix
__device__ void vec_to_skew(const float* v, float* M) {
    M[0] = 0;    M[1] = -v[2]; M[2] = v[1];
    M[3] = v[2]; M[4] = 0;    M[5] = -v[0];
    M[6] = -v[1]; M[7] = v[0]; M[8] = 0;
}

// SO(3) logarithm map: Rotation matrix (row-major) to axis-angle vector
__device__ void R_to_vec(const float* R, float* omega) {
    float trace = R[0] + R[4] + R[8];
    float cos_angle = 0.5f * (trace - 1.0f);
    // Clamp to valid range for acos
    cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle));
    float angle = acosf(cos_angle);

    if (fabsf(angle) < 1e-6f) { // Near identity
        omega[0] = 0.0f; omega[1] = 0.0f; omega[2] = 0.0f;
    } else if (fabsf(angle - M_PI) < 1e-6f) { // Near 180 degrees
       // More complex case, find axis - simplified here
       // Find largest diagonal element to find axis component sign
       if(R[0] > R[4] && R[0] > R[8]) { // xx largest
            float s = 2.0f * sqrtf(R[0] - R[4] - R[8] + 1.0f);
            omega[0] = 0.5f*s; omega[1] = (R[3]+R[1])/s; omega[2] = (R[2]+R[6])/s;
       } else if (R[4] > R[8]) { // yy largest
            float s = 2.0f * sqrtf(R[4] - R[0] - R[8] + 1.0f);
            omega[0] = (R[3]+R[1])/s; omega[1] = 0.5f*s; omega[2] = (R[7]+R[5])/s;
       } else { // zz largest
            float s = 2.0f * sqrtf(R[8] - R[0] - R[4] + 1.0f);
            omega[0] = (R[2]+R[6])/s; omega[1] = (R[7]+R[5])/s; omega[2] = 0.5f*s;
       }
       // Multiply by angle
       omega[0] *= angle; omega[1] *= angle; omega[2] *= angle;

    } else { // General case
        float sin_angle = sinf(angle);
        float factor = angle / (2.0f * sin_angle);
        omega[0] = factor * (R[7] - R[5]);
        omega[1] = factor * (R[2] - R[6]);
        omega[2] = factor * (R[3] - R[1]);
    }
}

// SE(3) Left Jacobian Inverse (J_l^{-1}) - Simplified
__device__ void Jl_inv(const float* omega, float* J_inv) {
    // J_inv is 3x3, assumes omega is axis-angle
    float angle = sqrtf(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);
    float omega_skew[9];
    vec_to_skew(omega, omega_skew); // Row-major skew matrix

    // Identity matrix
    for(int i=0; i<9; ++i) J_inv[i] = (i%4 == 0) ? 1.0f : 0.0f;

    if (fabsf(angle) < 1e-6f) {
        // J_inv = I - 0.5*[omega]_x
        for(int i=0; i<9; ++i) J_inv[i] -= 0.5f * omega_skew[i];
    } else {
        float angle_sq = angle * angle;
        float sin_angle = sinf(angle);
        float cos_angle = cosf(angle);
        float factor1 = -0.5f;
        float factor2 = (1.0f / angle_sq) - ((1.0f + cos_angle) / (2.0f * angle * sin_angle)); // Handle sin(angle)=0 case separately if needed

        float omega_skew_sq[9]; // omega_skew * omega_skew
        for(int r=0; r<3; ++r) {
            for(int c=0; c<3; ++c) {
                omega_skew_sq[r*3+c] = 0;
                for(int k=0; k<3; ++k) omega_skew_sq[r*3+c] += omega_skew[r*3+k] * omega_skew[k*3+c];
            }
        }

        for(int i=0; i<9; ++i) {
            J_inv[i] += factor1 * omega_skew[i] + factor2 * omega_skew_sq[i];
        }
    }
}

// SE(3) logarithm: T=[R,t] -> [rho, phi]
__device__ void logSE3(const float* R, const float* t, float* rho, float* phi) {
    R_to_vec(R, phi); // phi is axis-angle vector
    float J_inv[9]; // 3x3 row-major J_l^{-1}(phi)
    Jl_inv(phi, J_inv);
    // rho = J_inv * t
    rho[0] = J_inv[0]*t[0] + J_inv[1]*t[1] + J_inv[2]*t[2];
    rho[1] = J_inv[3]*t[0] + J_inv[4]*t[1] + J_inv[5]*t[2];
    rho[2] = J_inv[6]*t[0] + J_inv[7]*t[1] + J_inv[8]*t[2];
}

// --- Matrix operations ---
// Multiply 3x3 matrix A (row-major) by 3x1 vector x, output y=Ax
__device__ void mat33_vec3_mult(const float* A, const float* x, float* y) {
    y[0] = A[0]*x[0] + A[1]*x[1] + A[2]*x[2];
    y[1] = A[3]*x[0] + A[4]*x[1] + A[5]*x[2];
    y[2] = A[6]*x[0] + A[7]*x[1] + A[8]*x[2];
}

// Multiply 3x3 matrix A by 3x3 matrix B (both row-major), output C=AB
__device__ void mat33_mat33_mult(const float* A, const float* B, float* C) {
    for(int r=0; r<3; ++r) {
        for(int c=0; c<3; ++c) {
            C[r*3+c] = 0;
            for(int k=0; k<3; ++k) {
                C[r*3+c] += A[r*3+k] * B[k*3+c];
            }
        }
    }
}
// --- Main Kernel (Left Perturbation, SE(3) Log Residual) ---
__global__ void odom_constraint_kernel_left_perturb_log(
    // Inputs
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,      // World poses (N, 8) [tx, ty, tz, qx, qy, qz, qw, s]
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_meas_ij_acc, // Measured SE(3) relative poses (num_odom, 7 or 8)
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> odom_ii,   // Indices i for odometry edges
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> odom_jj,   // Indices j for odometry edges
    const float sigma_odom_t, const float sigma_odom_r,                             // Odometry std deviations (trans, rot)
    // Outputs
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs_odom,        // Output Hessian blocks (4, num_odom, 7, 7)
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs_odom         // Output Gradient blocks (2, num_odom, 7)
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_odom_edges = odom_ii.size(0);
    if (idx >= num_odom_edges) return;

    const int ix = static_cast<int>(odom_ii[idx]);
    const int jx = static_cast<int>(odom_jj[idx]);

    // --- Load Poses ---
    float ti[3], qi[4], si[1];
    float tj[3], qj[4], sj[1];
    si[0] = Twc[ix][7]; for(int k=0; k<3; ++k) ti[k] = Twc[ix][k]; for(int k=0; k<4; ++k) qi[k] = Twc[ix][k+3];
    sj[0] = Twc[jx][7]; for(int k=0; k<3; ++k) tj[k] = Twc[jx][k]; for(int k=0; k<4; ++k) qj[k] = Twc[jx][k+3];

    // --- Load Measurement (SE3) ---
    float t_m[3], q_m[4], R_m[9];
    for(int k=0; k<3; ++k) t_m[k] = T_meas_ij_acc[idx][k];
    for(int k=0; k<4; ++k) q_m[k] = T_meas_ij_acc[idx][k+3];
    quat_to_rot(q_m, R_m); // Measurement rotation R_m (row-major)

    // --- Precompute Transforms ---
    float R_i[9], R_j[9], R_i_T[9], R_j_T[9];
    quat_to_rot(qi, R_i); // R_i = R_wi
    quat_to_rot(qj, R_j); // R_j = R_wj
    // Transposes (Row Major)
    R_i_T[0]=R_i[0]; R_i_T[1]=R_i[3]; R_i_T[2]=R_i[6]; R_i_T[3]=R_i[1]; R_i_T[4]=R_i[4]; R_i_T[5]=R_i[7]; R_i_T[6]=R_i[2]; R_i_T[7]=R_i[5]; R_i_T[8]=R_i[8];
    R_j_T[0]=R_j[0]; R_j_T[1]=R_j[3]; R_j_T[2]=R_j[6]; R_j_T[3]=R_j[1]; R_j_T[4]=R_j[4]; R_j_T[5]=R_j[7]; R_j_T[6]=R_j[2]; R_j_T[7]=R_j[5]; R_j_T[8]=R_j[8];

    // --- Compute Predicted SE(3) part T_pred_SE3 = [R_ij, t'_ij] ---
    float R_ij[9];      // R_ij = R_i^T * R_j (Relative rotation)
    mat33_mat33_mult(R_i_T, R_j, R_ij);
    float t_diff[3] = {tj[0]-ti[0], tj[1]-ti[1], tj[2]-ti[2]};
    float R_iT_tdiff[3];
    mat33_vec3_mult(R_i_T, t_diff, R_iT_tdiff);
    float t_pred_prime[3]; // t'_ij = (1/s_j) * R_i^T * (tj - ti)
    float sj_inv = (sj[0] == 0.0f) ? 1e6f : 1.0f / sj[0]; // Avoid div by zero
    t_pred_prime[0] = sj_inv * R_iT_tdiff[0];
    t_pred_prime[1] = sj_inv * R_iT_tdiff[1];
    t_pred_prime[2] = sj_inv * R_iT_tdiff[2];

    // --- Compute Error Transformation T_err = T_meas^{-1} * T_pred_SE3 ---
    // T_meas = [R_m, t_m] -> T_meas_inv = [R_m^T, -R_m^T t_m]
    float R_m_T[9]; // Transpose R_m
    R_m_T[0]=R_m[0]; R_m_T[1]=R_m[3]; R_m_T[2]=R_m[6]; R_m_T[3]=R_m[1]; R_m_T[4]=R_m[4]; R_m_T[5]=R_m[7]; R_m_T[6]=R_m[2]; R_m_T[7]=R_m[5]; R_m_T[8]=R_m[8];
    float t_meas_inv[3];
    mat33_vec3_mult(R_m_T, t_m, t_meas_inv);
    t_meas_inv[0] *= -1.0f; t_meas_inv[1] *= -1.0f; t_meas_inv[2] *= -1.0f;

    float R_err[9]; // R_err = R_m^T * R_ij
    mat33_mat33_mult(R_m_T, R_ij, R_err);
    float t_err[3]; // t_err = R_m^T * t_pred_prime + t_meas_inv
    float Rm_T_tpred[3];
    mat33_vec3_mult(R_m_T, t_pred_prime, Rm_T_tpred);
    t_err[0] = Rm_T_tpred[0] + t_meas_inv[0];
    t_err[1] = Rm_T_tpred[1] + t_meas_inv[1];
    t_err[2] = Rm_T_tpred[2] + t_meas_inv[2];

    // --- Compute Residual Vector r = log(T_err) ---
    float r[6]; // [rho, phi]
    float* rho = &r[0];
    float* phi = &r[3];
    logSE3(R_err, t_err, rho, phi);

    // print rho and phi TODO: remove
    // printf("Edge %d: rho = [%.3f, %.3f, %.3f], phi = [%.3f, %.3f, %.3f]\n", idx, rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]);

    // --- Compute Weights (Anisotropic) ---
    float W[36] = {0}; // 6x6 diagonal weight matrix W = diag(info_t, info_t, info_t, info_r, info_r, info_r) * Huber
    float info_t = 1.0f / (sigma_odom_t * sigma_odom_t);
    float info_r = 1.0f / (sigma_odom_r * sigma_odom_r);
    W[0]  = info_t * huber(r[0] / sigma_odom_t); // rho_x
    W[7]  = info_t * huber(r[1] / sigma_odom_t); // rho_y
    W[14] = info_t * huber(r[2] / sigma_odom_t); // rho_z
    W[21] = info_r * huber(r[3] / sigma_odom_r); // phi_x
    W[28] = info_r * huber(r[4] / sigma_odom_r); // phi_y
    W[35] = info_r * huber(r[5] / sigma_odom_r); // phi_z

    // --- Compute Jacobians (Left Perturbation, SE(3) Log approx + Scale) ---
    // J is 6x14 = [ J_rho_i | J_phi_i | J_rho_j | J_phi_j ]
    // J_i is 6x7, J_j is 6x7
    // delta_x = [dt^w, dtheta^w, ds]
    float J_i[42] = {0}; // 6x7 Jacobian w.r.t state i (row-major)
    float J_j[42] = {0}; // 6x7 Jacobian w.r.t state j (row-major)

    // Approx J based on Adjoint(T_j^-1) structure near r=0
    float tj_inv[3]; // t part of T_j^{-1} = -R_j^T t_j
    mat33_vec3_mult(R_j_T, tj, tj_inv);
    tj_inv[0]*=-1.0; tj_inv[1]*=-1.0; tj_inv[2]*=-1.0;
    float tj_inv_skew[9]; // [tj_inv]_x
    vec_to_skew(tj_inv, tj_inv_skew);

    // J_j: Approx Ad(T_j^-1)
    // d_rho / d_tj^w = R_j^T
    J_j[0] = R_j_T[0]; J_j[1] = R_j_T[1]; J_j[2] = R_j_T[2]; // Row 0
    J_j[7] = R_j_T[3]; J_j[8] = R_j_T[4]; J_j[9] = R_j_T[5]; // Row 1
    J_j[14]= R_j_T[6]; J_j[15]= R_j_T[7]; J_j[16]= R_j_T[8]; // Row 2
    // d_rho / d_thetaj^w = [tj_inv]_x * R_j^T
    float drho_dthetaj[9]; // 3x3 matrix
    mat33_mat33_mult(tj_inv_skew, R_j_T, drho_dthetaj);
    J_j[3] = drho_dthetaj[0]; J_j[4] = drho_dthetaj[1]; J_j[5] = drho_dthetaj[2]; // Row 0
    J_j[10]= drho_dthetaj[3]; J_j[11]= drho_dthetaj[4]; J_j[12]= drho_dthetaj[5]; // Row 1
    J_j[17]= drho_dthetaj[6]; J_j[18]= drho_dthetaj[7]; J_j[19]= drho_dthetaj[8]; // Row 2
    // d_phi / d_thetaj^w = R_j^T
    J_j[24]= R_j_T[0]; J_j[25]= R_j_T[1]; J_j[26]= R_j_T[2]; // Row 3
    J_j[31]= R_j_T[3]; J_j[32]= R_j_T[4]; J_j[33]= R_j_T[5]; // Row 4
    J_j[38]= R_j_T[6]; J_j[39]= R_j_T[7]; J_j[40]= R_j_T[8]; // Row 5
    // d_rho / d_sj = -R_m^T * t'_ij (Approximation)
    float drho_dsj[3];
    mat33_vec3_mult(R_m_T, t_pred_prime, drho_dsj);
    J_j[6] = -drho_dsj[0];  // Row 0, Col 6
    J_j[13]= -drho_dsj[1];  // Row 1, Col 6
    J_j[20]= -drho_dsj[2];  // Row 2, Col 6

    // J_i: Approx -Ad(T_j^-1)
    for(int k=0; k<42; ++k) J_i[k] = -J_j[k]; // Negate most terms
    // Override d_rho / d_si = 0 (Approximation)
    J_i[6] = 0.0; J_i[13] = 0.0; J_i[20] = 0.0;
    // Override d_phi / d_si = 0
    J_i[27]= 0.0; J_i[34] = 0.0; J_i[41] = 0.0;

    // --- Accumulate H and g ---
    // H = J^T W J, g = J^T W r
    float J[84]; // Combined Jacobian 6x14 = [J_i | J_j] (Row Major)
    for(int r=0; r<6; ++r) { for(int c=0; c<7; ++c) J[r*14+c] = J_i[r*7+c]; }
    for(int r=0; r<6; ++r) { for(int c=0; c<7; ++c) J[r*14+(c+7)] = J_j[r*7+c]; }

    float J_T_W[84*6] = {0}; // J^T (14x6) * W (6x6) -> (14x6)
    for(int r_jt=0; r_jt<14; ++r_jt) {
        for(int c_w=0; c_w<6; ++c_w) {
            // (r_jt, c_w) element of J^T * W
            // = sum_k J^T(r_jt, k) * W(k, c_w)
            // = sum_k J(k, r_jt) * W(k, c_w) (W is diagonal)
             J_T_W[r_jt*6 + c_w] = J[c_w*14 + r_jt] * W[c_w*6 + c_w];
        }
    }

    float H_full[196] = {0}; // Full 14x14 Hessian block H = (J^T W) * J
    for(int r_h=0; r_h<14; ++r_h) {
        for(int c_j=0; c_j<14; ++c_j) {
            // (r_h, c_j) element of H
            // = sum_k (J^T W)(r_h, k) * J(k, c_j)
            float val = 0;
            for(int k=0; k<6; ++k) {
                val += J_T_W[r_h*6 + k] * J[k*14 + c_j];
            }
            H_full[r_h*14 + c_j] = val;
        }
    }

    float g_full[14] = {0}; // Full 14x1 gradient block g = (J^T W) * r
    for(int r_jt=0; r_jt<14; ++r_jt) {
        // (r_jt) element of g
        // = sum_k (J^T W)(r_jt, k) * r(k)
        float val = 0;
        for(int k=0; k<6; ++k) {
            val += J_T_W[r_jt*6 + k] * r[k];
        }
        g_full[r_jt] = val;
    }

    // --- Write results to Global Memory ---
    // Split g into g_i, g_j
    for (int n = 0; n < 7; ++n) gs_odom[0][idx][n] = g_full[n];     // g_i
    for (int n = 0; n < 7; ++n) gs_odom[1][idx][n] = g_full[n + 7]; // g_j
    // Unpack H into H_ii, H_ij, H_ji, H_jj
    for (int r = 0; r < 7; ++r) { for (int c = 0; c < 7; ++c) Hs_odom[0][idx][r][c] = H_full[r*14 + c]; }         // H_ii
    for (int r = 0; r < 7; ++r) { for (int c = 0; c < 7; ++c) Hs_odom[1][idx][r][c] = H_full[r*14 + (c + 7)]; }   // H_ij
    for (int r = 0; r < 7; ++r) { for (int c = 0; c < 7; ++c) Hs_odom[2][idx][r][c] = H_full[(r + 7)*14 + c]; }   // H_ji
    for (int r = 0; r < 7; ++r) { for (int c = 0; c < 7; ++c) Hs_odom[3][idx][r][c] = H_full[(r + 7)*14 + (c + 7)]; } // H_jj

} // end of kernel


// --- Main CUDA Kernel (Left Perturbation) ---
__global__ void odom_constraint_kernel_left_perturb(
    // Inputs
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,      // World poses (N, 8) [tx, ty, tz, qx, qy, qz, qw, s]
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_meas_ij, // Measured SE(3) relative poses (num_odom, 8/7 - uses first 7)
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> odom_ii,   // Indices i for odometry edges
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> odom_jj,   // Indices j for odometry edges
    const float sigma_odom,                                                       // Odometry standard deviation (scalar, for Huber)
    // Outputs
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs_odom,        // Output Hessian blocks (4, num_odom, 7, 7)
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs_odom         // Output Gradient blocks (2, num_odom, 7)
) {
    // --- Thread Indexing ---
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_odom_edges = odom_ii.size(0);
    if (idx >= num_odom_edges) return;

    // --- Get Pose Indices ---
    const int ix = static_cast<int>(odom_ii[idx]);
    const int jx = static_cast<int>(odom_jj[idx]);

    // --- Load Poses and Measurement into Local Variables (Registers) ---
    float ti[3], qi[4], si[1];
    float tj[3], qj[4], sj[1];
    float t_meas_ij[3], q_meas_ij[4]; // s_meas_ij is ignored for SE(3) measurements

    // Load pose i (World frame T_wi)
    si[0] = Twc[ix][7];
    for(int k=0; k<3; ++k) ti[k] = Twc[ix][k];
    for(int k=0; k<4; ++k) qi[k] = Twc[ix][k+3];

    // Load pose j (World frame T_wj)
    sj[0] = Twc[jx][7];
    for(int k=0; k<3; ++k) tj[k] = Twc[jx][k];
    for(int k=0; k<4; ++k) qj[k] = Twc[jx][k+3];

    // Load SE(3) measurement T_meas_ij (Relative pose j in i's frame)
    for(int k=0; k<3; ++k) t_meas_ij[k] = T_meas_ij[idx][k];
    for(int k=0; k<4; ++k) q_meas_ij[k] = T_meas_ij[idx][k+3];

    // --- Compute Estimated Relative Transformation T_ij = T_i^{-1} * T_j ---
    float tij[3], qij[4], sij[1];
    // relSim3 calculates T_ij = T_wi^{-1} * T_wj
    // tij = (1/si) * Ri^T * (tj - ti)
    // qij = qi^{-1} * qj
    // sij = sj / si
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);

    // --- Compute Residuals (Simplified) ---
    float err[6]; // [err_tx, err_ty, err_tz, err_rx, err_ry, err_rz]

    // Translation residual: e_t = t_ij - t_meas_ij
    for (int k = 0; k < 3; k++) {
        err[k] = tij[k] - t_meas_ij[k];
    }

    // Rotation residual: e_r approx 2 * vec(q_ij * q_meas_ij^{-1})
    float q_meas_ij_inv[4];
    quat_inv(q_meas_ij, q_meas_ij_inv);
    float q_err[4]; // q_err = q_ij * q_meas_ij_inv
    quat_comp(qij, q_meas_ij_inv, q_err);
    if (q_err[3] < 0.0f) { // Ensure canonical quaternion for vec approx
       q_err[0] *= -1.0f; q_err[1] *= -1.0f; q_err[2] *= -1.0f; q_err[3] *= -1.0f;
    }
    err[3] = 2.0f * q_err[0]; err[4] = 2.0f * q_err[1]; err[5] = 2.0f * q_err[2];

    // print the error, TODO: remove this
    printf("err for edge %d: %f, %f, %f, %f, %f, %f\n", idx, err[0], err[1], err[2], err[3], err[4], err[5]);
    // --- Compute Weights ---
    float w[6];
    const float sigma_odom_inv = 1.0f / sigma_odom;
    const float info = sigma_odom_inv * sigma_odom_inv;
    for (int k = 0; k < 6; k++) {
        w[k] = info * huber(err[k] * sigma_odom_inv);
    }

    // --- Precomputations for Jacobians (Left Perturbation) ---
    float R_i[9];       // R_wi (world from body_i), Row Major
    float R_j[9];       // R_wj (world from body_j), Row Major
    float R_i_T[9];     // R_iw (body_i from world), Row Major
    float R_j_T[9];     // R_jw (body_j from world), Row Major
    float si_inv = 1.0f / si[0];
    float tij_x = tij[0], tij_y = tij[1], tij_z = tij[2]; // Cache components

    quat_to_rot(qi, R_i); // R_i is R_wi (row-major)
    quat_to_rot(qj, R_j); // R_j is R_wj (row-major)

    // Compute R_i^T (Row Major)
    R_i_T[0] = R_i[0]; R_i_T[1] = R_i[3]; R_i_T[2] = R_i[6]; // Col 0 -> Row 0
    R_i_T[3] = R_i[1]; R_i_T[4] = R_i[4]; R_i_T[5] = R_i[7]; // Col 1 -> Row 1
    R_i_T[6] = R_i[2]; R_i_T[7] = R_i[5]; R_i_T[8] = R_i[8]; // Col 2 -> Row 2

    // Compute R_j^T (Row Major)
    R_j_T[0] = R_j[0]; R_j_T[1] = R_j[3]; R_j_T[2] = R_j[6]; // Col 0 -> Row 0
    R_j_T[3] = R_j[1]; R_j_T[4] = R_j[4]; R_j_T[5] = R_j[7]; // Col 1 -> Row 1
    R_j_T[6] = R_j[2]; R_j_T[7] = R_j[5]; R_j_T[8] = R_j[8]; // Col 2 -> Row 2


    // --- Compute Jacobians and Accumulate H/g ---
    // Using Left Perturbation delta_x = [dt^world, dtheta^world, ds]
    float Jx[14]; float* Ji = &Jx[0]; float* Jj = &Jx[7]; // Combined Jacobian J = [Ji | Jj]
    const int h_dim = 105; float hij[h_dim]; // Stores upper triangle of H = J^T W J
    float vi[7], vj[7]; // Stores components of g = J^T W err
    for (int l = 0; l < h_dim; ++l) hij[l] = 0.0f;
    for (int n = 0; n < 7; ++n) { vi[n] = 0.0f; vj[n] = 0.0f; }

    // --- Process Translation Residuals (k=0, 1, 2 for err_t[x,y,z]) ---
    for (int k = 0; k < 3; ++k) { // Loop over residual components err[0], err[1], err[2]
        for (int n = 0; n < 7; ++n) { Ji[n] = 0.0f; Jj[n] = 0.0f; } // Clear Jacobians

        // Jacobian Jj (w.r.t delta_x_j = [dt_j^w, dtheta_j^w, ds_j])
        // ∂err_t[k] / ∂delta_t_j^w = ( (1/s_i) * R_i^T )_row_k
        Jj[0] = si_inv * R_i_T[k*3 + 0]; // k-th row, 0-th col
        Jj[1] = si_inv * R_i_T[k*3 + 1]; // k-th row, 1-st col
        Jj[2] = si_inv * R_i_T[k*3 + 2]; // k-th row, 2-nd col
        // Other derivatives are 0
        Jj[3] = 0.0f; Jj[4] = 0.0f; Jj[5] = 0.0f; Jj[6] = 0.0f;

        // Jacobian Ji (w.r.t delta_x_i = [dt_i^w, dtheta_i^w, ds_i])
        // ∂err_t[k] / ∂delta_t_i^w = -( (1/s_i) * R_i^T )_row_k
        Ji[0] = -si_inv * R_i_T[k*3 + 0]; // k-th row, 0-th col
        Ji[1] = -si_inv * R_i_T[k*3 + 1]; // k-th row, 1-st col
        Ji[2] = -si_inv * R_i_T[k*3 + 2]; // k-th row, 2-nd col
        // ∂err_t[k] / ∂delta_theta_i^w = ( [t_ij]_x * R_i^T )_row_k
        // temp_vec = R_i^T * delta_theta_i^w
        // result = [t_ij]_x * temp_vec
        // Need k-th row of the matrix [t_ij]_x * R_i^T
        float mat_row[3]; // Row k of the 3x3 matrix [t_ij]_x * R_i^T
        mat_row[0] = (k == 1) * tij_z * R_i_T[0*3+0] - (k == 2) * tij_y * R_i_T[0*3+0]  // Row k, Col 0 of [t]_x * R^T
                   + (k == 1) * tij_z * R_i_T[1*3+0] - (k == 2) * tij_y * R_i_T[1*3+0]
                   + (k == 1) * tij_z * R_i_T[2*3+0] - (k == 2) * tij_y * R_i_T[2*3+0]; // Simplified: This is complex. Let's compute row k of [t_ij]_x first
        float tij_x_row_k[3];
        tij_x_row_k[0] = (k == 1) * tij_z - (k == 2) * tij_y; // Row k, Col 0 of [t_ij]_x
        tij_x_row_k[1] = (k == 2) * tij_x - (k == 0) * tij_z; // Row k, Col 1 of [t_ij]_x
        tij_x_row_k[2] = (k == 0) * tij_y - (k == 1) * tij_x; // Row k, Col 2 of [t_ij]_x
        // Now dot product with columns of R_i^T (which are rows of R_i)
        Ji[3] = tij_x_row_k[0] * R_i_T[0*3+0] + tij_x_row_k[1] * R_i_T[1*3+0] + tij_x_row_k[2] * R_i_T[2*3+0]; // Dot row k of [t]_x with col 0 of R_i^T
        Ji[4] = tij_x_row_k[0] * R_i_T[0*3+1] + tij_x_row_k[1] * R_i_T[1*3+1] + tij_x_row_k[2] * R_i_T[2*3+1]; // Dot row k of [t]_x with col 1 of R_i^T
        Ji[5] = tij_x_row_k[0] * R_i_T[0*3+2] + tij_x_row_k[1] * R_i_T[1*3+2] + tij_x_row_k[2] * R_i_T[2*3+2]; // Dot row k of [t]_x with col 2 of R_i^T
        // ∂err_t[k] / ∂delta_s_i = -t_ij[k]
        Ji[6] = -tij[k];

        // Accumulate H += J^T * w[k] * J and g += J^T * w[k] * err[k]
        float weight = w[k]; float error = err[k]; int l = 0;
        for (int n = 0; n < 14; ++n) { for (int m = 0; m <= n; ++m) { hij[l] += weight * Jx[n] * Jx[m]; l++; } }
        float weighted_error = weight * error;
        for (int n = 0; n < 7; ++n) { vi[n] += weighted_error * Ji[n]; }
        for (int n = 0; n < 7; ++n) { vj[n] += weighted_error * Jj[n]; }
    }

    // --- Process Rotation Residuals (k=3, 4, 5 for err_r[x,y,z]) ---
    // Using Left Perturbation Model Approximations: J_theta_i = -R_j^T, J_theta_j = R_j^T
    for (int k = 0; k < 3; ++k) { // Loop over residual components err[3], err[4], err[5]
        for (int n = 0; n < 7; ++n) { Ji[n] = 0.0f; Jj[n] = 0.0f; } // Clear Jacobians

        // Jacobian Jj (w.r.t delta_x_j = [dt_j^w, dtheta_j^w, ds_j])
        // ∂err_r[k] / ∂delta_theta_j^w[xyz] ≈ (R_j^T)_row_k
        Jj[3] = R_j_T[k*3 + 0]; // k-th row, 0-th col
        Jj[4] = R_j_T[k*3 + 1]; // k-th row, 1-st col
        Jj[5] = R_j_T[k*3 + 2]; // k-th row, 2-nd col

        // Jacobian Ji (w.r.t delta_x_i = [dt_i^w, dtheta_i^w, ds_i])
        // ∂err_r[k] / ∂delta_theta_i^w[xyz] ≈ -(R_j^T)_row_k
        Ji[3] = -R_j_T[k*3 + 0]; // k-th row, 0-th col
        Ji[4] = -R_j_T[k*3 + 1]; // k-th row, 1-st col
        Ji[5] = -R_j_T[k*3 + 2]; // k-th row, 2-nd col

        // Accumulate H/g
        int residual_idx = k + 3;
        float weight = w[residual_idx]; float error = err[residual_idx]; int l = 0;
        for (int n = 0; n < 14; ++n) { for (int m = 0; m <= n; ++m) { hij[l] += weight * Jx[n] * Jx[m]; l++; } }
        float weighted_error = weight * error;
        for (int n = 0; n < 7; ++n) { vi[n] += weighted_error * Ji[n]; }
        for (int n = 0; n < 7; ++n) { vj[n] += weighted_error * Jj[n]; }
    }

    // --- Write results directly to Global Memory ---
    int l = 0;
    for (int n = 0; n < 7; ++n) {
         gs_odom[0][idx][n] = vi[n]; // g_i
         gs_odom[1][idx][n] = vj[n]; // g_j
    }
    l = 0;
    for (int n = 0; n < 14; ++n) { for (int m = 0; m <= n; ++m) {
            float val = hij[l];
            if (n < 7) { // Row i
                if (m < 7) { // Col i -> H_ii
                    Hs_odom[0][idx][n][m] = val; if (n != m) Hs_odom[0][idx][m][n] = val;
                } else { // Col j -> H_ij / H_ji
                    Hs_odom[1][idx][n][m - 7] = val; Hs_odom[2][idx][m - 7][n] = val;
                }
            } else { // Row j
                 if (m >= 7) { // Col j -> H_jj
                    Hs_odom[3][idx][n - 7][m - 7] = val; if (n != m) Hs_odom[3][idx][m - 7][n - 7] = val;
                }
            } l++; } }
} // end of kernel


__global__ void odom_constraint_kernel(
    // Inputs
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,      // World poses (N, 8) [tx, ty, tz, qx, qy, qz, qw, s]
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> T_meas_ij, // Measured SE(3) relative poses (num_odom, 8/7 - uses first 7)
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> odom_ii,   // Indices i for odometry edges
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> odom_jj,   // Indices j for odometry edges
    const float sigma_odom,                                                       // Odometry standard deviation (scalar, for Huber)
    // Outputs
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,             // Output Hessian blocks (4, num_odom, 7, 7)
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs              // Output Gradient blocks (2, num_odom, 7)
) {
    // --- Thread Indexing ---
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_odom_edges = odom_ii.size(0);
    if (idx >= num_odom_edges) return;

    // --- Get Pose Indices ---
    const int ix = static_cast<int>(odom_ii[idx]);
    const int jx = static_cast<int>(odom_jj[idx]);

    // --- Load Poses and Measurement into Local Variables (Registers) ---
    float ti[3], qi[4], si[1];
    float tj[3], qj[4], sj[1];
    float t_meas_ij[3], q_meas_ij[4]; // s_meas_ij is ignored for SE(3) measurements

    // Load pose i (World frame)
    si[0] = Twc[ix][7];
    for(int k=0; k<3; ++k) ti[k] = Twc[ix][k];
    for(int k=0; k<4; ++k) qi[k] = Twc[ix][k+3];

    // Load pose j (World frame)
    sj[0] = Twc[jx][7];
    for(int k=0; k<3; ++k) tj[k] = Twc[jx][k];
    for(int k=0; k<4; ++k) qj[k] = Twc[jx][k+3];

    // Load SE(3) measurement T_meas_ij (Relative pose j in i's frame)
    for(int k=0; k<3; ++k) t_meas_ij[k] = T_meas_ij[idx][k];
    for(int k=0; k<4; ++k) q_meas_ij[k] = T_meas_ij[idx][k+3];
    // Ignore T_meas_ij[idx][7] as it's an SE(3) measurement

    // --- Compute Estimated Relative Transformation T_ij = T_i^{-1} * T_j ---
    float tij[3], qij[4], sij[1];
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij); // tij is (1/si)*Ri^T*(tj-ti)

    // --- Compute Residuals ---
    float err[6]; // [err_tx, err_ty, err_tz, err_rx, err_ry, err_rz]

    // Translation residual (Simplified): e_t = t_ij - t_meas_ij
    for (int k = 0; k < 3; k++) {
        err[k] = tij[k] - t_meas_ij[k];
    }

    // Rotation residual (Approximation): e_r approx 2 * vec(q_ij * q_meas_ij^{-1})
    float q_meas_ij_inv[4];
    quat_inv(q_meas_ij, q_meas_ij_inv);
    float q_err[4]; // q_err = q_ij * q_meas_ij_inv
    quat_comp(qij, q_meas_ij_inv, q_err);
    // Ensure scalar part is positive for stability of vec approx
    if (q_err[3] < 0.0f) {
       q_err[0] *= -1.0f; q_err[1] *= -1.0f; q_err[2] *= -1.0f; q_err[3] *= -1.0f;
    }
    err[3] = 2.0f * q_err[0]; // 2 * qx_err
    err[4] = 2.0f * q_err[1]; // 2 * qy_err
    err[5] = 2.0f * q_err[2]; // 2 * qz_err

    // Scale residual is ignored for SE(3) measurements.

    // --- Compute Weights ---
    float w[6];   // Robust weights: w_k = Info * huber_weight_k
    const float sigma_odom_inv = 1.0f / sigma_odom;
    const float info = sigma_odom_inv * sigma_odom_inv; // Information = 1 / sigma^2
    for (int k = 0; k < 6; k++) {
        // Use normalized error for Huber function
        w[k] = info * huber(err[k] * sigma_odom_inv);
    }

    // --- Precomputations for Jacobians (using Right Perturbation) ---
    float R_i[9];       // R_wi (world from body_i), Row Major
    float R_j[9];       // R_wj (world from body_j), Row Major
    float R_ij[9];      // R_ji (body_j from body_i) = Ri^T * Rj, Row Major
    float s_ij = sj[0] / si[0]; // Relative scale s_j / s_i
    float tij_x = tij[0], tij_y = tij[1], tij_z = tij[2]; // Cache components

    quat_to_rot(qi, R_i); // R_i = R_wi (row-major)
    quat_to_rot(qj, R_j); // R_j = R_wj (row-major)

    // Compute R_ij = R_i^T * R_j (Row Major)
    // R_ij[row, col] = sum_k (R_i_T[row, k] * R_j[k, col])
    // R_ij[row, col] = sum_k (R_i[k, row] * R_j[k, col])
    for(int row = 0; row < 3; ++row) {
        for(int col = 0; col < 3; ++col) {
            R_ij[row*3 + col] = 0.0f;
            for(int k = 0; k < 3; ++k) {
                // Access R_i[k, row] from row-major R_i[k*3 + row]
                // Access R_j[k, col] from row-major R_j[k*3 + col]
                R_ij[row*3 + col] += R_i[k*3 + row] * R_j[k*3 + col];
            }
        }
    }

    // --- Compute Jacobians and Accumulate H/g ---
    // Using Right Perturbation delta_x = [dt^body, dtheta^body, ds]
    float Jx[14]; float* Ji = &Jx[0]; float* Jj = &Jx[7]; // Combined Jacobian J = [Ji | Jj]
    const int h_dim = 105; // 14*(14+1)/2 = 105
    float hij[h_dim]; // Stores upper triangle of H = J^T W J
    float vi[7], vj[7]; // Stores components of g = J^T W err
    for (int l = 0; l < h_dim; ++l) hij[l] = 0.0f;
    for (int n = 0; n < 7; ++n) { vi[n] = 0.0f; vj[n] = 0.0f; }

    // --- Process Translation Residuals (k=0, 1, 2 for err_t[x,y,z]) ---
    for (int k = 0; k < 3; ++k) { // Loop over residual components err[0], err[1], err[2]
        for (int n = 0; n < 7; ++n) { Ji[n] = 0.0f; Jj[n] = 0.0f; } // Clear Jacobians for this residual component

        // Jacobian Jj (w.r.t delta_x_j = [dt_j^b, dtheta_j^b, ds_j])
        // ∂err_t[k] / ∂delta_t_j^b = (s_ij * R_ij)_row_k
        Jj[0] = s_ij * R_ij[k*3 + 0]; // k-th row, 0-th col of s_ij*R_ij
        Jj[1] = s_ij * R_ij[k*3 + 1]; // k-th row, 1-st col of s_ij*R_ij
        Jj[2] = s_ij * R_ij[k*3 + 2]; // k-th row, 2-nd col of s_ij*R_ij
        // ∂err_t[k] / ∂delta_theta_j^b = 0
        Jj[3] = 0.0f; Jj[4] = 0.0f; Jj[5] = 0.0f;
        // ∂err_t[k] / ∂delta_s_j = 0
        Jj[6] = 0.0f;

        // Jacobian Ji (w.r.t delta_x_i = [dt_i^b, dtheta_i^b, ds_i])
        // ∂err_t[k] / ∂delta_t_i^b = -I_row_k
        Ji[0] = (k == 0) ? -1.0f : 0.0f;
        Ji[1] = (k == 1) ? -1.0f : 0.0f;
        Ji[2] = (k == 2) ? -1.0f : 0.0f;
        // ∂err_t[k] / ∂delta_theta_i^b = [t_ij]_x_row_k
        Ji[3] = (k == 1) * tij_z - (k == 2) * tij_y; // Row k, Col 0 of [t_ij]_x
        Ji[4] = (k == 2) * tij_x - (k == 0) * tij_z; // Row k, Col 1 of [t_ij]_x
        Ji[5] = (k == 0) * tij_y - (k == 1) * tij_x; // Row k, Col 2 of [t_ij]_x
        // ∂err_t[k] / ∂delta_s_i = -t_ij[k]
        Ji[6] = -tij[k];

        // Accumulate H += J^T * w[k] * J and g += J^T * w[k] * err[k]
        float weight = w[k]; float error = err[k]; int l = 0;
        // H = J^T * W * J -> hij stores upper triangle of H
        for (int n = 0; n < 14; ++n) { // Row index of J^T (col index of J)
            for (int m = 0; m <= n; ++m) { // Col index of J^T (row index of J)
                 // Accumulate Jx[n] * weight * Jx[m]
                 hij[l] += weight * Jx[n] * Jx[m];
                 l++;
            }
        }
        // g = J^T * W * err -> vi/vj store components of g
        float weighted_error = weight * error;
        for (int n = 0; n < 7; ++n) { vi[n] += weighted_error * Ji[n]; } // g_i part
        for (int n = 0; n < 7; ++n) { vj[n] += weighted_error * Jj[n]; } // g_j part
    }

    // --- Process Rotation Residuals (k=3, 4, 5 for err_r[x,y,z]) ---
    // Using Right Perturbation Model Approximations: J_theta_i = -I, J_theta_j = I
    for (int k = 0; k < 3; ++k) { // Loop over residual components err[3], err[4], err[5]
        for (int n = 0; n < 7; ++n) { Ji[n] = 0.0f; Jj[n] = 0.0f; } // Clear Jacobians

        // Jacobian Jj (w.r.t delta_x_j = [dt_j^b, dtheta_j^b, ds_j])
        // ∂err_r[k] / ∂delta_theta_j^b[k] ≈ 1
        Jj[k + 3] = 1.0f; // Jj[3], Jj[4], Jj[5]

        // Jacobian Ji (w.r.t delta_x_i = [dt_i^b, dtheta_i^b, ds_i])
        // ∂err_r[k] / ∂delta_theta_i^b[k] ≈ -1
        Ji[k + 3] = -1.0f; // Ji[3], Ji[4], Ji[5]

        // Accumulate H/g
        int residual_idx = k + 3;
        float weight = w[residual_idx]; float error = err[residual_idx]; int l = 0;
        // H = J^T * W * J
        for (int n = 0; n < 14; ++n) {
            for (int m = 0; m <= n; ++m) {
                hij[l] += weight * Jx[n] * Jx[m];
                l++;
            }
        }
        // g = J^T * W * err
        float weighted_error = weight * error;
        for (int n = 0; n < 7; ++n) { vi[n] += weighted_error * Ji[n]; }
        for (int n = 0; n < 7; ++n) { vj[n] += weighted_error * Jj[n]; }
    }

    // --- Write results directly to Global Memory ---
    // Gradients g_i, g_j (Accumulated J^T * W * err)
    for (int n = 0; n < 7; ++n) {
         // Use atomicAdd if multiple edges might update the same pose block in parallel,
         // otherwise direct write is fine if output buffers are per-edge.
         // Assuming gs is pre-zeroed and output is per edge:
         gs[0][idx][n] = vi[n]; // g_i
         gs[1][idx][n] = vj[n]; // g_j
    }

    // Hessian blocks H_ii, H_ij, H_ji, H_jj (Accumulated J^T * W * J)
    int l = 0; // Reset index for hij (upper triangle of 14x14 H)
    for (int n = 0; n < 14; ++n) { // Row index of full H
        for (int m = 0; m <= n; ++m) { // Col index of full H (upper triangle)
            float val = hij[l];
            // Deconstruct 14x14 H into 7x7 blocks H_ii, H_ij, H_ji, H_jj
            if (n < 7) { // Row corresponds to pose i
                if (m < 7) { // Column corresponds to pose i -> H_ii block
                    // Use atomicAdd if needed, otherwise direct write:
                    Hs[0][idx][n][m] = val; // H_ii[n, m]
                    if (n != m) Hs[0][idx][m][n] = val; // H_ii[m, n] (symmetry)
                } else { // Column corresponds to pose j -> H_ij block
                    Hs[1][idx][n][m - 7] = val; // H_ij[n, m-7]
                    Hs[2][idx][m - 7][n] = val; // H_ji[m-7, n] (H_ji = H_ij^T)
                }
            } else { // Row corresponds to pose j
                 // Only need m >= 7 since we fill upper triangle (m <= n)
                 if (m >= 7) { // Column corresponds to pose j -> H_jj block
                    Hs[3][idx][n - 7][m - 7] = val; // H_jj[n-7, m-7]
                    if (n != m) Hs[3][idx][m - 7][n - 7] = val; // H_jj[m-7, n-7] (symmetry)
                 }
            }
            l++; // Move to next element in hij
        }
    }
} // end of kernel


// --- Function to Apply Scale Prior ---
/**
 * @brief Adds scale prior constraints to the Hessian and gradient vector.
 *
 * @param A The SparseBlock object holding the Hessian (A.A) and gradient (A.b). Modified in place.
 * @param Twc_cpu CPU Tensor (double) containing current world poses [num_poses, 8].
 * @param sbar_targets_cpu CPU Tensor (double) containing target scale for each pose [num_poses]. Invalid targets <= 0.
 * @param num_fix Number of fixed poses at the beginning.
 * @param sigma_scale_prior Standard deviation of the scale prior "measurement".
 */
void apply_scale_prior_cpu(
    SparseBlock& A, // Pass by reference to modify
    const torch::Tensor& Twc_cpu,
    const torch::Tensor& sbar_targets_cpu,
    int num_fix,
    double sigma_scale_prior)
{
    if (sigma_scale_prior <= 1e-9) { // Prevent division by zero / huge info
        std::cerr << "Warning: sigma_scale_prior is too small or zero. Skipping priors." << std::endl;
        return;
    }

    const int num_poses = Twc_cpu.size(0);
    const int pose_dim = A.M; // Should be 7 for Sim(3)
    const int scale_dof_idx_in_block = pose_dim - 1; // Index 6 for scale DoF

    auto Twc_acc = Twc_cpu.accessor<double, 2>();
    auto sbar_acc = sbar_targets_cpu.accessor<double, 1>();

    const double info_scale_prior = 1.0 / (sigma_scale_prior * sigma_scale_prior);
    const double jacobian_s = 1.0; // Jacobian dr/d(delta_s) for r = log(si/sbar)

    int scale_priors_added = 0;
    double g_scale_prior_norm_sq = 0;
    double h_scale_prior_sum = 0;

    for (int i = num_fix; i < num_poses; ++i) {
        // Direct mapping from global pose index to optimization index
        int i_opt = i - num_fix;

        // Ensure i_opt is within the bounds of the optimization problem size
        if (i_opt < 0 || i_opt >= A.N) {
             // This shouldn't happen with the direct mapping, but good sanity check
             std::cerr << "Warning: Optimization index i_opt=" << i_opt << " out of bounds for global pose " << i << ". Skipping prior." << std::endl;
             continue;
        }

        double sbar_i = sbar_acc[i];

        // Check if sbar is valid
        if (sbar_i <= 1e-6) { // Use small positive threshold
            continue; // Skip this pose if target scale is invalid
        }

        double si = Twc_acc[i][scale_dof_idx_in_block]; // Get current scale si

        if (si <= 1e-6) { // Avoid log(0) or issues with current scale being zero
             // std::cerr << "Warning: Current scale si for pose " << i << " is near zero (" << si << "). Skipping prior." << std::endl;
             continue;
        }

        // Residual: r_s = log(si / sbar_i)
        double residual_s = log(si / sbar_i);

        // --- Add contribution to A.A and A.b ---
        long system_dof_index = (long)i_opt * pose_dim + scale_dof_idx_in_block;

        // H_ii(scale,scale) += J^T * w * J = 1.0 * info_scale_prior * 1.0
        double h_contrib = info_scale_prior * jacobian_s * jacobian_s; // = info_scale_prior
        A.A.coeffRef(system_dof_index, system_dof_index) += h_contrib;
        h_scale_prior_sum += h_contrib; // For debug

        // g_i(scale) += J^T * w * r = 1.0 * info_scale_prior * residual_s
        double g_contrib = info_scale_prior * jacobian_s * residual_s;
        A.b(system_dof_index) += g_contrib;
        g_scale_prior_norm_sq += g_contrib * g_contrib; // For debug

        scale_priors_added++;
    }

    // Optional: Print debug info once per function call (outside the loop)
    std::cout << "Applied " << scale_priors_added << " scale priors." << std::endl;
    std::cout << "  ‖g_scale_prior_contrib‖ = " << std::sqrt(g_scale_prior_norm_sq) << std::endl;
    std::cout << "  Sum(H_scale_prior_diag_contrib) = " << h_scale_prior_sum << std::endl;
}

std::vector<torch::Tensor> gauss_newton_rays_odom_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  torch::Tensor odom_ii, torch::Tensor odom_jj, // odom edges, from i to j
  torch::Tensor Tij, // Tj in Ti, or delta T between i and j
  torch::Tensor s_bar,
  const float sigma_odom_t, const float sigma_odom_r,
  const float sigma_ray,
  const float sigma_dist,
  const float sigma_scale_prior,
  const float C_thresh,
  const float Q_thresh,
  const int num_fix,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_odom_edges = odom_ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  // Setup indexing
  // TODO: it should make sure the unique kf idx is from both the visual and odometry edges
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  // For odometry edges
  std::vector<torch::Tensor> odom_inds_opt = create_inds(unique_kf_idx, num_fix, odom_ii, odom_jj);
  torch::Tensor odom_ii_opt = odom_inds_opt[0];
  torch::Tensor odom_jj_opt = odom_inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);
  
  // Buffers for odometry constraints
  torch::Tensor Hs_odom = torch::zeros({4, num_odom_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs_odom = torch::zeros({2, num_odom_edges, pose_dim}, opts);

  // Ensure sbar_targets is on CPU and double
  auto sbar_targets_cpu = s_bar.to(torch::kCPU).to(torch::kFloat64);

  // For debugging outputs
  torch::Tensor dx;
  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {
    // Visual constraints
    ray_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_ray, sigma_dist, C_thresh, Q_thresh
    );

    // Odometry constraints
    if (num_odom_edges > 0) {
      odom_constraint_kernel_left_perturb_log<<<num_odom_edges, THREADS>>>(
        Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        Tij.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        odom_ii.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        odom_jj.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
        sigma_odom_t, sigma_odom_r,
        Hs_odom.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        gs_odom.packed_accessor32<float,3,torch::RestrictPtrTraits>()
      );
    }

    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    // Add visual constraints
    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));
    
    // Add odometry constraints if available
    if (num_odom_edges > 0) {
      // print the Hs_odom and gs_odom for debugging TODO: remove this
      auto g_vis  = gs     .reshape({-1,7}).cpu();
      auto g_odom = gs_odom.reshape({-1,7}).cpu();
      std::cout << "‖g_vis‖  = " << g_vis.norm().item<float>()  << std::endl;
      std::cout << "‖g_odom‖ = " << g_odom.norm().item<float>() << std::endl;

      auto H = Hs_odom.reshape({-1,7,7}).cpu();
      std::cout << "Column‑6 infinity norm = "
                << H.index({torch::indexing::Slice(), torch::indexing::Slice(), 6})
                  .abs().sum().item<float>() << std::endl;

      A.update_lhs(Hs_odom.reshape({-1, pose_dim, pose_dim}), 
          torch::cat({odom_ii_opt, odom_ii_opt, odom_jj_opt, odom_jj_opt}), 
          torch::cat({odom_ii_opt, odom_jj_opt, odom_ii_opt, odom_jj_opt}));

      A.update_rhs(gs_odom.reshape({-1, pose_dim}), 
          torch::cat({odom_ii_opt, odom_jj_opt}));
    }

    // --- Add the Scale Prior ---
    // Get current poses on CPU for prior calculation
    auto Twc_cpu = Twc.to(torch::kCPU).to(torch::kFloat64);
    apply_scale_prior_cpu(A, Twc_cpu, sbar_targets_cpu, num_fix, sigma_scale_prior);
    // --- End of Scale Prior Addition ---

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
  }

  return {dx}; // For debugging
}

__global__ void point_align_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const float sigma_point,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[3];
  float w[3];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
    // Parameters
  const float sigma_point_inv = 1.0/sigma_point;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);
 
    // Error (difference in camera rays)
    err[0] = Xj_Ci[0] - Xi[0];
    err[1] = Xj_Ci[1] - Xi[1];
    err[2] = Xj_Ci[2] - Xi[2];
 
    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid = 
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh);

    // Weight using confidences
    const float conf_weight = q;
    // const float conf_weight = q * ci * cj;
    
    const float sqrt_w_point = valid ? sigma_point_inv * sqrtf(conf_weight) : 0;
 
    // Robust weight
    w[0] = huber(sqrt_w_point * err[0]);
    w[1] = huber(sqrt_w_point * err[1]);
    w[2] = huber(sqrt_w_point * err[2]);
    
    // Add back in sigma
    const float w_const_point = sqrt_w_point * sqrt_w_point;
    w[0] *= w_const_point;
    w[1] *= w_const_point;
    w[2] *= w_const_point;
 
    // Jacobians
    
    // x coordinate
    Ji[0] = 1.0;
    Ji[1] = 0.0;
    Ji[2] = 0.0;
    Ji[3] = 0.0;
    Ji[4] = Xj_Ci[2]; // z
    Ji[5] = -Xj_Ci[1]; // -y
    Ji[6] = Xj_Ci[0]; // x

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }
 
    // y coordinate
    Ji[0] = 0.0;
    Ji[1] = 1.0;
    Ji[2] = 0.0;
    Ji[3] = -Xj_Ci[2]; // -z
    Ji[4] = 0; 
    Ji[5] = Xj_Ci[0]; // x
    Ji[6] = Xj_Ci[1]; // y
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }
 
    // z coordinate
    Ji[0] = 0.0;
    Ji[1] = 0.0;
    Ji[2] = 1.0;
    Ji[3] = Xj_Ci[1]; // y
    Ji[4] = -Xj_Ci[0]; // -x 
    Ji[5] = 0;
    Ji[6] = Xj_Ci[2]; // z
 
    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }
 
    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }
 
 
  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}

std::vector<torch::Tensor> gauss_newton_points_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const float sigma_point,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  // TODO: make this a parameter
  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    point_align_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      sigma_point, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();
    
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}

__global__ void calib_proj_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> Twc,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Xs,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Cs,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> K,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> ii,
    const torch::PackedTensorAccessor32<long,1,torch::RestrictPtrTraits> jj,
    const torch::PackedTensorAccessor32<long,2,torch::RestrictPtrTraits> idx_ii2_jj,
    const torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> valid_match,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> Q,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> Hs,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> gs,
    const int height,
    const int width,
    const int pixel_border,
    const float z_eps,
    const float sigma_pixel,
    const float sigma_depth,
    const float C_thresh,
    const float Q_thresh)
{
 
  // Twc and Xs first dim is number of poses
  // ii, jj, Cii, Cjj, Q first dim is number of edges
 
  const int block_id = blockIdx.x;
  const int thread_id = threadIdx.x;
 
  const int num_points = Xs.size(1);
 
  int ix = static_cast<int>(ii[block_id]);
  int jx = static_cast<int>(jj[block_id]);

  __shared__ float fx;
  __shared__ float fy;
  __shared__ float cx;
  __shared__ float cy;
 
  __shared__ float ti[3], tj[3], tij[3];
  __shared__ float qi[4], qj[4], qij[4];
  __shared__ float si[1], sj[1], sij[1];

  // load intrinsics from global memory
  if (thread_id == 0) {
    fx = K[0][0];
    fy = K[1][1];
    cx = K[0][2];
    cy = K[1][2];
  }
 
  __syncthreads();
 
  // load poses from global memory
  if (thread_id < 3) {
    ti[thread_id] = Twc[ix][thread_id];
    tj[thread_id] = Twc[jx][thread_id];
  }
 
  if (thread_id < 4) {
    qi[thread_id] = Twc[ix][thread_id+3];
    qj[thread_id] = Twc[jx][thread_id+3];
  }
 
  if (thread_id < 1) {
    si[thread_id] = Twc[ix][thread_id+7];
    sj[thread_id] = Twc[jx][thread_id+7];
  }
 
  __syncthreads();
 
  // Calculate relative poses
  if (thread_id == 0) {
    relSim3(ti, qi, si, tj, qj, sj, tij, qij, sij);
  }
 
  __syncthreads();
 
  // //points
  float Xi[3];
  float Xj[3];
  float Xj_Ci[3];
 
  // residuals
  float err[3];
  float w[3];
 
  // // jacobians
  float Jx[14];
  // float Jz;
 
  float* Ji = &Jx[0];
  float* Jj = &Jx[7];
 
  // hessians
  const int h_dim = 14*(14+1)/2;
  float hij[h_dim];
 
  float vi[7], vj[7];
 
  int l; // We reuse this variable later for Hessian fill-in
  for (l=0; l<h_dim; l++) {
    hij[l] = 0;
  }
 
  for (int n=0; n<7; n++) {
    vi[n] = 0;
    vj[n] = 0;
  }
 
  // Parameters
  const float sigma_pixel_inv = 1.0/sigma_pixel;
  const float sigma_depth_inv = 1.0/sigma_depth;
 
  __syncthreads();
 
  GPU_1D_KERNEL_LOOP(k, num_points) {
 
    // Get points
    const bool valid_match_ind = valid_match[block_id][k][0]; 
    const int64_t ind_Xi = valid_match_ind ? idx_ii2_jj[block_id][k] : 0;

    Xi[0] = Xs[ix][ind_Xi][0];
    Xi[1] = Xs[ix][ind_Xi][1];
    Xi[2] = Xs[ix][ind_Xi][2];
 
    Xj[0] = Xs[jx][k][0];
    Xj[1] = Xs[jx][k][1];
    Xj[2] = Xs[jx][k][2];

    // Get measurement pixel
    const int u_target = ind_Xi % width; 
    const int v_target = ind_Xi / width;
 
    // Transform point
    actSim3(tij, qij, sij, Xj, Xj_Ci);

    // // Check if in front of camera
    const bool valid_z = ((Xj_Ci[2] > z_eps) && (Xi[2] > z_eps));

    // Handle depth related vars
    const float zj_inv = valid_z ? 1.0/Xj_Ci[2] : 0.0;
    const float zj_log = valid_z ? logf(Xj_Ci[2]) : 0.0;
    const float zi_log = valid_z ? logf(Xi[2]) : 0.0; 

    // Project point
    const float x_div_z = Xj_Ci[0] * zj_inv;
    const float y_div_z = Xj_Ci[1] * zj_inv;
    const float u = fx * x_div_z + cx;
    const float v = fy * y_div_z + cy;

    // Handle proj
    const bool valid_u = ((u > pixel_border) && (u < width - 1 - pixel_border));
    const bool valid_v = ((v > pixel_border) && (v < height - 1 - pixel_border));

    // Error (difference in camera rays)
    err[0] = u - u_target;
    err[1] = v - v_target;
    err[2] = zj_log - zi_log; // Log-depth

    // Weights (Huber)
    const float q = Q[block_id][k][0];
    const float ci = Cs[ix][ind_Xi][0];
    const float cj = Cs[jx][k][0];
    const bool valid =
      valid_match_ind
      & (q > Q_thresh)
      & (ci > C_thresh)
      & (cj > C_thresh)
      & valid_u & valid_v & valid_z; // Check for valid image and depth
    
    // Weight using confidences
    const float conf_weight = q;
    
    const float sqrt_w_pixel = valid ? sigma_pixel_inv * sqrtf(conf_weight) : 0;
    const float sqrt_w_depth = valid ? sigma_depth_inv * sqrtf(conf_weight) : 0;

    // Robust weight
    w[0] = huber(sqrt_w_pixel * err[0]);
    w[1] = huber(sqrt_w_pixel * err[1]);
    w[2] = huber(sqrt_w_depth * err[2]);
    
    // Add back in sigma
    const float w_const_pixel = sqrt_w_pixel * sqrt_w_pixel;
    const float w_const_depth = sqrt_w_depth * sqrt_w_depth;
    w[0] *= w_const_pixel;
    w[1] *= w_const_pixel;
    w[2] *= w_const_depth;

    // Jacobians    

    // x coordinate
    Ji[0] = fx * zj_inv;
    Ji[1] = 0.0;
    Ji[2] = -fx * x_div_z * zj_inv;
    Ji[3] = -fx * x_div_z * y_div_z;
    Ji[4] = fx * (1 + x_div_z*x_div_z);
    Ji[5] = -fx * y_div_z; 
    Ji[6] = 0.0;

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];


    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[0] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[0] * err[0] * Ji[n];
      vj[n] += w[0] * err[0] * Jj[n];
    }

    // y coordinate
    Ji[0] = 0.0; 
    Ji[1] = fy * zj_inv;
    Ji[2] = -fy * y_div_z * zj_inv;
    Ji[3] = -fy * (1 + y_div_z*y_div_z);
    Ji[4] = fy * x_div_z * y_div_z;
    Ji[5] = fy * x_div_z; 
    Ji[6] = 0.0;

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[1] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[1] * err[1] * Ji[n];
      vj[n] += w[1] * err[1] * Jj[n];
    }

    // z coordinate
    Ji[0] = 0.0; 
    Ji[1] = 0.0; 
    Ji[2] = zj_inv;
    Ji[3] = y_div_z; // y
    Ji[4] = -x_div_z; // -x
    Ji[5] = 0.0;
    Ji[6] = 1.0; // z

    apply_Sim3_adj_inv(ti, qi, si, Ji, Jj);
    for (int n=0; n<7; n++) Ji[n] = -Jj[n];

    l=0;
    for (int n=0; n<14; n++) {
      for (int m=0; m<=n; m++) {
        hij[l] += w[2] * Jx[n] * Jx[m];
        l++;
      }
    }

    for (int n=0; n<7; n++) {
      vi[n] += w[2] * err[2] * Ji[n];
      vj[n] += w[2] * err[2] * Jj[n];
    }

  }
 
  __syncthreads();
 
  __shared__ float sdata[THREADS];
  for (int n=0; n<7; n++) {
    sdata[threadIdx.x] = vi[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[0][block_id][n] = sdata[0];
    }
 
    __syncthreads();
 
    sdata[threadIdx.x] = vj[n];
    blockReduce(sdata);
    if (threadIdx.x == 0) {
      gs[1][block_id][n] = sdata[0];
    }
 
  }
 
  l=0;
  for (int n=0; n<14; n++) {
    for (int m=0; m<=n; m++) {
      sdata[threadIdx.x] = hij[l];
      blockReduce(sdata);
 
      if (threadIdx.x == 0) {
        if (n<7 && m<7) {
          Hs[0][block_id][n][m] = sdata[0];
          Hs[0][block_id][m][n] = sdata[0];
        }
        else if (n >=7 && m<7) {
          Hs[1][block_id][m][n-7] = sdata[0];
          Hs[2][block_id][n-7][m] = sdata[0];
        }
        else {
          Hs[3][block_id][n-7][m-7] = sdata[0];
          Hs[3][block_id][m-7][n-7] = sdata[0];
        }
      }
 
      l++;
    }
  }
}


std::vector<torch::Tensor> gauss_newton_calib_cuda(
  torch::Tensor Twc, torch::Tensor Xs, torch::Tensor Cs,
  torch::Tensor K,
  torch::Tensor ii, torch::Tensor jj, 
  torch::Tensor idx_ii2jj, torch::Tensor valid_match,
  torch::Tensor Q,
  const int height, const int width,
  const int pixel_border,
  const float z_eps,
  const float sigma_pixel, const float sigma_depth,
  const float C_thresh,
  const float Q_thresh,
  const int max_iter,
  const float delta_thresh)
{
  auto opts = Twc.options();
  const int num_edges = ii.size(0);
  const int num_poses = Xs.size(0);
  const int n = Xs.size(1);

  const int num_fix = 1;

  // Setup indexing
  torch::Tensor unique_kf_idx = get_unique_kf_idx(ii, jj);
  // For edge construction
  std::vector<torch::Tensor> inds = create_inds(unique_kf_idx, 0, ii, jj);
  torch::Tensor ii_edge = inds[0];
  torch::Tensor jj_edge = inds[1];
  // For linear system indexing (pin=2 because fixing first two poses)
  std::vector<torch::Tensor> inds_opt = create_inds(unique_kf_idx, num_fix, ii, jj);
  torch::Tensor ii_opt = inds_opt[0];
  torch::Tensor jj_opt = inds_opt[1];

  const int pose_dim = 7; // sim3

  // initialize buffers
  torch::Tensor Hs = torch::zeros({4, num_edges, pose_dim, pose_dim}, opts);
  torch::Tensor gs = torch::zeros({2, num_edges, pose_dim}, opts);

  // For debugging outputs
  torch::Tensor dx;

  torch::Tensor delta_norm;

  for (int itr=0; itr<max_iter; itr++) {

    calib_proj_kernel<<<num_edges, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      Xs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Cs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      K.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      ii_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      jj_edge.packed_accessor32<long,1,torch::RestrictPtrTraits>(),
      idx_ii2jj.packed_accessor32<long,2,torch::RestrictPtrTraits>(),
      valid_match.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
      Q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      Hs.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      gs.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
      height, width, pixel_border, z_eps, sigma_pixel, sigma_depth, C_thresh, Q_thresh
    );


    // pose x pose block
    SparseBlock A(num_poses - num_fix, pose_dim);

    A.update_lhs(Hs.reshape({-1, pose_dim, pose_dim}), 
        torch::cat({ii_opt, ii_opt, jj_opt, jj_opt}), 
        torch::cat({ii_opt, jj_opt, ii_opt, jj_opt}));

    A.update_rhs(gs.reshape({-1, pose_dim}), 
        torch::cat({ii_opt, jj_opt}));

    // NOTE: Accounting for negative here!
    dx = -A.solve();

    
    pose_retr_kernel<<<1, THREADS>>>(
      Twc.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      dx.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
      num_fix);

    // Termination criteria
    // Need to specify this second argument otherwise ambiguous function call...
    delta_norm = torch::linalg::linalg_norm(dx, std::optional<c10::Scalar>(), {}, false, {});
    if (delta_norm.item<float>() < delta_thresh) {
      break;
    }
        

  }

  return {dx}; // For debugging
}