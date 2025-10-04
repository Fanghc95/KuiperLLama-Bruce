#include <device_launch_parameters.h>
#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"
namespace kernel {
/**
 * 计算多维输入 in = (dim1, dim2), 计算在dim2维度上的rmsnorm
 */
/**
 * @brief 对多维输入张量执行按行（维度）的 RMSNorm 计算
 * @details RMSNorm（Root Mean Square Normalization）是一种归一化技术，计算输入的均方根并进行归一化
 *          此函数处理形状为 (dim_size, size) 的输入张量，在第二个维度（size维度）上执行RMSNorm
 * 
 * @param in 输入数据指针，类型为float，形状为[dim_size, size]
 * @param wei 权重参数指针，类型为float，形状为[size]
 * @param out 输出数据指针，类型为float，形状为[dim_size, size]
 * @param dim_size 第一个维度的大小（行数）
 * @param size 第二个维度的大小（每行元素数）
 * @param eps 防止除零错误的小值，通常为1e-6f
 */
static __global__ void row_rmsnorm_f32_dim(float* in, float* wei, float* out, int dim_size, 
                                           int size, float eps) {
  // 获取当前线程块ID和线程ID
  const int bid = blockIdx.x;    // 线程块ID，对应输入张量的第一维索引
  const int tid = threadIdx.x;   // 线程ID，对应线程块内的线程索引
  
  // 边界检查：确保当前处理的行索引在有效范围内
  if (bid >= dim_size) {
    return;
  }

  // 计算当前块处理的输入和输出数据起始地址
  float* block_in = in + bid * size;    // 当前行的输入数据起始位置
  float* block_out = out + bid * size;  // 当前行的输出数据起始位置
  
  // 向量打包参数：使用float4（4个float一组）提高内存访问效率
  constexpr int pack_size = 4;          // 每个向量包的元素数量
  const int pack_num = size / pack_size;    // 可以完整打包的组数
  const int pack_off = pack_size * pack_num; // 剩余元素的起始索引

  // 第一步：计算输入数据的平方和
  float sum = 0.0f;  // 当前线程计算的平方和
  
  // 将输入数据指针转换为float4指针，用于向量访问
  float4* in_pack = reinterpret_cast<float4*>(block_in);
  
  // 处理可以完整打包的元素（向量化访问）
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i); // 一次加载4个float元素
    // 累加4个元素的平方值
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  // 处理剩余的无法完整打包的元素（标量访问）
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += block_in[i] * block_in[i];
  }

  // 第二步：使用CUB库的BlockReduce进行线程块内的归约，计算总和
  using BlockReduce = cub::BlockReduce<float, 128>;  // 定义BlockReduce类型，块大小为128
  __shared__ typename BlockReduce::TempStorage temp; // 声明共享内存用于归约操作
  __shared__ float shared_val;                       // 共享内存变量，用于存储归约结果
  
  // 执行块内归约，计算所有线程的平方和
  sum = BlockReduce(temp).Sum(sum);
  
  // 主线程（线程ID为0）将归约结果存储到共享内存
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads(); // 同步所有线程，确保所有线程都能读取到正确的总和
  
  // 所有线程读取共享内存中的总和
  sum = shared_val;
  
  // 计算归一化缩放因子：1/sqrt(mean(square) + eps)
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  // 第三步：应用归一化和权重乘法
  float4* wei_pack = reinterpret_cast<float4*>(wei);  // 将权重指针转换为float4指针
  float4* out_pack = reinterpret_cast<float4*>(block_out); // 将输出指针转换为float4指针
  
  // 处理可以完整打包的元素（向量化操作）
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);   // 加载输入数据
    float4 wei_float4 = *(wei_pack + i); // 加载权重数据
    // 计算输出：scale * 输入 * 权重，并存储结果
    *(out_pack + i) = 
        make_float4(scale * in_float4.x * wei_float4.x, 
                    scale * in_float4.y * wei_float4.y, 
                    scale * in_float4.z * wei_float4.z, 
                    scale * in_float4.w * wei_float4.w);
  }

  // 处理剩余的无法完整打包的元素（标量操作）
  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    block_out[i] = wei[i] * block_in[i] * scale;
  }
}

template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float* in, float* wei, float* out, int size, float eps) {
  const int tid = threadIdx.x;

  constexpr int pack_size = 4;
  const int pack_num = size / pack_size;
  const int pack_off = pack_size * pack_num;

  float sum = 0.0f;
  float4* in_pack = reinterpret_cast<float4*>(in);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    sum += in_float4.x * in_float4.x;
    sum += in_float4.y * in_float4.y;
    sum += in_float4.z * in_float4.z;
    sum += in_float4.w * in_float4.w;
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    sum += in[i] * in[i];
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;
  const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

  float4* wei_pack = reinterpret_cast<float4*>(wei);
  float4* out_pack = reinterpret_cast<float4*>(out);
  for (int i = tid; i < pack_num; i += blockDim.x) {
    float4 in_float4 = *(in_pack + i);
    float4 wei_float4 = *(wei_pack + i);
    *(out_pack + i) =
        make_float4(scale * in_float4.x * wei_float4.x, scale * in_float4.y * wei_float4.y,
                    scale * in_float4.z * wei_float4.z, scale * in_float4.w * wei_float4.w);
  }

  for (int i = pack_off + tid; i < size; i += blockDim.x) {
    out[i] = wei[i] * in[i] * scale;
  }
}

void rmsnorm_kernel_cu(const tensor::Tensor& input, const tensor::Tensor& weight,
                       const tensor::Tensor& output, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
  const float eps = 1e-6f;
#else
  const float eps = 1e-5f;
#endif
  const int32_t size = static_cast<int32_t>(input.size());
  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  } else {
    row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
  }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  CHECK(!input.is_empty());
  CHECK(!weight.is_empty());
  CHECK(!output.is_empty());

  CHECK(input.device_type() == base::DeviceType::kDeviceCUDA &&
        weight.device_type() == base::DeviceType::kDeviceCUDA &&
        output.device_type() == base::DeviceType::kDeviceCUDA);

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1);
  const int32_t dim_size = total_size / size;

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                               size, eps);
  } else {
    row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
}
}  // namespace kernel