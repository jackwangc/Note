# Tilelang 算子精度问题定位

## example 地址

`examples/cross_entropy_loss/example_cross_entro.py`

## example 转换生成的 AcendC 代码

```c++
#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR x_handle,  GM_ADDR y_handle,  GM_ADDR loss_handle,  GM_ADDR log_prob_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<half> x;
  x.SetGlobalBuffer((__gm__ half*)x_handle);
  AscendC::GlobalTensor<int> y;
  y.SetGlobalBuffer((__gm__ int*)y_handle);
  AscendC::GlobalTensor<half> loss;
  loss.SetGlobalBuffer((__gm__ half*)loss_handle);
  AscendC::GlobalTensor<half> log_prob;
  log_prob.SetGlobalBuffer((__gm__ half*)log_prob_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 131072);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto prev_max = ascend_ub.GetWithOffset<float>(64,0);
  auto prev_sum = ascend_ub.GetWithOffset<float>(64,256);
  auto x_ub = ascend_ub.GetWithOffset<half>(8192,512);
  auto x_32 = ascend_ub.GetWithOffset<float>(8192,16896);
  auto tile_max = ascend_ub.GetWithOffset<float>(64,49664);
  auto temp_reduce = ascend_ub.GetWithOffset<uint8_t>(98304,49920);
  auto temp_exp = ascend_ub.GetWithOffset<float>(64,148224);
  auto tile_sum = ascend_ub.GetWithOffset<float>(64,148480);
  auto y_ub = ascend_ub.GetWithOffset<int>(64,148736);
  auto l_n_32 = ascend_ub.GetWithOffset<float>(64,148992);
  auto l_n = ascend_ub.GetWithOffset<half>(64,149248);
  auto vid = AscendC::GetSubBlockIdx();

  if ASCEND_IS_AIV {
    tl::ascend::Fill<float>(prev_max[0], -CUDART_INF_F, 64);
    tl::ascend::Fill<float>(prev_sum[0], 0.000000e+00f, 64);

    for (int32_t bc = 0; bc < 8; ++bc) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2);
      tl::ascend::copy_gm_to_ub<half, 128, 64>(x_ub[0], x[(((cid * 131072) + (vid * 65536)) + (bc * 128))], 1024);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::Cast(x_32[0], x_ub[0], AscendC::RoundMode::CAST_NONE, 8192);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::PipeBarrier<PIPE_ALL>();
      tl::ascend::reduce_max<float, 64, 128, -1>(tile_max[0], x_32[0], temp_reduce[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max(tile_max[0], prev_max[0], tile_max[0], 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(temp_exp[0], prev_max[0], tile_max[0], 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(temp_exp[0], temp_exp[0], 64);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(temp_exp[0], prev_sum[0], temp_exp[0], 64);
      for (int32_t n_idx = 0; n_idx < 64; ++n_idx) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PipeBarrier<PIPE_ALL>();
        auto tile_max_scalar = tile_max.GetValue(n_idx);
        AscendC::Adds(x_32[(n_idx * 128)], x_32[(n_idx * 128)], -tile_max_scalar, 128);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(x_32[0], x_32[0], 8192);
      AscendC::PipeBarrier<PIPE_V>();

      tl::ascend::reduce_sum<float, 64, 128, -1>(tile_sum[0], x_32[0], temp_reduce[0]);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(prev_sum[0], tile_sum[0], temp_exp[0], 64);
      tl::ascend::copy_ub_to_ub<float, float, 64>(prev_max[0], tile_max[0]);

    }
    tl::ascend::copy_gm_to_ub<int, 64>(y_ub[0], y[((cid * 128) + (vid * 64))], 1024);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Ln(prev_sum[0], prev_sum[0], 64);

    for (int32_t bc_1 = 0; bc_1 < 8; ++bc_1) {
      AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(4);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(4);
      tl::ascend::copy_gm_to_ub<half, 128, 64>(x_ub[0], x[(((cid * 131072) + (vid * 65536)) + (bc_1 * 128))], 1024);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(5);
      AscendC::Cast(x_32[0], x_ub[0], AscendC::RoundMode::CAST_NONE, 8192);
      for (int32_t n_idx_1 = 0; n_idx_1 < 64; ++n_idx_1) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(x_32[(n_idx_1 * 128)], x_32[(n_idx_1 * 128)], -(prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1)), 128);
      }
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Cast(x_ub[0], x_32[0], AscendC::RoundMode::CAST_RINT, 8192);
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(6);
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(6);
      tl::ascend::copy_ub_to_gm<half, 128, 64>(log_prob[(((cid * 131072) + (vid * 65536)) + (bc_1 * 128))], x_ub[0], 1024);
      for (int32_t n_idx_2 = 0; n_idx_2 < 64; ++n_idx_2) {
        AscendC::PipeBarrier<PIPE_ALL>();
        if ((0 <= y_ub.GetValue(n_idx_2)) && (y_ub.GetValue(n_idx_2) < 128)) {
          l_n_32.SetValue(n_idx_2, (x_32.GetValue(((n_idx_2 * 128) + y_ub.GetValue(n_idx_2))) * -1.000000e+00f));
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::PipeBarrier<PIPE_ALL>();

      }
      AscendC::Adds(y_ub[0], y_ub[0], -128, 64);
    }
    AscendC::Cast(l_n[0], l_n_32[0], AscendC::RoundMode::CAST_RINT, 64);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);

    tl::ascend::copy_ub_to_gm<half, 64>(loss[((cid * 128) + (vid * 64))], l_n[0], 1024);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* x_handle, uint8_t* y_handle, uint8_t* loss_handle, uint8_t* log_prob_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<8, nullptr, stream>>>(x_handle, y_handle, loss_handle, log_prob_handle, fftsAddr);
}
```

## example 生成的 PTO 代码

```c++
#include "tl_templates/pto/common.h"
#include <pto/pto-inst.hpp>
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace pto;

AICORE void main_kernel(__gm__ half *x_handle, __gm__ int *y_handle, __gm__ half *loss_handle, __gm__ half *log_prob_handle, uint64_t ffts_Addr) {
  auto cid = get_block_idx();
  set_ffts_base_addr(ffts_Addr);

  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> prev_max;
  TASSIGN(prev_max, 0);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> prev_sum;
  TASSIGN(prev_sum, 256);
  tl::ascend_pto::TileUbDataND<half, 64, 128, 64, 128> x_ub;
  TASSIGN(x_ub, 512);
  tl::ascend_pto::TileUbDataND<float, 64, 128, 64, 128> x_32;
  TASSIGN(x_32, 16896);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> tile_max;
  TASSIGN(tile_max, 49664);
  tl::ascend_pto::TileUbDataND<uint8_t, 1, 98304, 1, 98304> temp_reduce;
  TASSIGN(temp_reduce, 49920);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> temp_exp;
  TASSIGN(temp_exp, 148224);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> tile_sum;
  TASSIGN(tile_sum, 148480);
  tl::ascend_pto::TileUbDataND<int, 1, 64, 1, 64> y_ub;
  TASSIGN(y_ub, 148736);
  tl::ascend_pto::TileUbDataND<float, 1, 64, 1, 64> l_n_32;
  TASSIGN(l_n_32, 148992);
  tl::ascend_pto::TileUbDataND<half, 1, 64, 1, 64> l_n;
  TASSIGN(l_n, 149248);
  auto vid = get_subblockid();
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    TEXPANDS(prev_max, -CUDART_INF_F);
    TEXPANDS(prev_sum, 0.000000e+00f);


  for (int32_t bc = 0; bc < 8; ++bc) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
      tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 1, 1, 1024 * 1024, 1024, 1, 64, 128>(x_handle + (((cid * 131072) + (vid * 65536)) + (bc * 128)), x_ub);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TCVT(x_32, x_ub, RoundMode::CAST_NONE);

      pipe_barrier(PIPE_V);
      tl::ascend_pto::TileUbDataDN <float, 64, 1, 64, 1> tile_max_DN;
      TASSIGN(tile_max_DN, 49664);
      TROWMAX(tile_max_DN, x_32, temp_reduce);
      pipe_barrier(PIPE_ALL);
      TRESHAPE(tile_max, tile_max_DN);
      pipe_barrier(PIPE_V);
      TMAX(tile_max, prev_max, tile_max);
      pipe_barrier(PIPE_V);
      TSUB(temp_exp, prev_max, tile_max);
      pipe_barrier(PIPE_V);
      TEXP(temp_exp, temp_exp);
      pipe_barrier(PIPE_V);
      TMUL(temp_exp, prev_sum, temp_exp);

      for (int32_t n_idx = 0; n_idx < 64; ++n_idx) {
        pipe_barrier(PIPE_V);
        pipe_barrier(PIPE_ALL);
        auto tile_max_scalar= tile_max.GetValue(n_idx);
        pipe_barrier(PIPE_ALL);
        tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> x_32_temp;
        TASSIGN(x_32_temp, 16896 + (n_idx * 128) * 4);
        pipe_barrier(PIPE_ALL);
        TADDS(x_32_temp, x_32_temp, -tile_max_scalar);
        
      }
      pipe_barrier(PIPE_V);
      TEXP(x_32, x_32);
      pipe_barrier(PIPE_V);

      tl::ascend_pto::TileUbDataDN <float, 64, 1, 64, 1> tile_sum_DN;
      TASSIGN(tile_sum_DN, 148480);
      TROWSUM(tile_sum_DN, x_32, temp_reduce);
      pipe_barrier(PIPE_ALL);
      TRESHAPE(tile_sum, tile_sum_DN);
      pipe_barrier(PIPE_V);
      TADD(prev_sum, tile_sum, temp_exp);
      TMOV(prev_max, tile_max);
  }

  tl::ascend_pto::copy_gm_to_ub<int, int, 1, 1, 1, 1, 64, 1, 1, 1, 1024, 1, 1, 64>(y_handle + ((cid * 128) + (vid * 64)), y_ub);
  pipe_barrier(PIPE_V);
  TLOG(prev_sum, prev_sum);

  for (int32_t bc_1 = 0; bc_1 < 8; ++bc_1) {
      set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
      tl::ascend_pto::copy_gm_to_ub<half, half, 1, 1, 1, 64, 128, 1, 1, 1024 * 1024, 1024, 1, 64, 128>(x_handle + (((cid * 131072) + (vid * 65536)) + (bc_1 * 128)), x_ub);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID5);
      TCVT(x_32, x_ub, RoundMode::CAST_NONE);

      for (int32_t n_idx_1 = 0; n_idx_1 < 64; ++n_idx_1) {
        pipe_barrier(PIPE_V);
        TADDS(x_32, x_32, -(prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1)));
      }
      pipe_barrier(PIPE_V);
      TCVT(x_ub, x_32, RoundMode::CAST_RINT);
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID6);
      tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 64, 128, 1, 1, 1024 * 1024, 1024, 1, 64, 128, 64, 128>(log_prob_handle + (((cid * 131072) + (vid * 65536)) + (bc_1 * 128)), 512, 0,2);

      for (int32_t n_idx_2 = 0; n_idx_2 < 64; ++n_idx_2) {
        pipe_barrier(PIPE_ALL);
        if ((0 <= y_ub.GetValue(n_idx_2)) && (y_ub.GetValue(n_idx_2) < 128)) {
          l_n_32.SetValue(n_idx_2, (x_32.GetValue(((n_idx_2 * 128) + y_ub.GetValue(n_idx_2))) * -1.000000e+00f));
        }
        pipe_barrier(PIPE_ALL);
        pipe_barrier(PIPE_ALL);

      }
      TADDS(y_ub, y_ub, -128);

  }
  TCVT(l_n, l_n_32, RoundMode::CAST_RINT);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  tl::ascend_pto::copy_ub_to_gm<half, half, 1, 1, 1, 1, 64, 1, 1, 1, 1024, 1, 1, 64, 1, 64>(loss_handle + ((cid * 128) + (vid * 64)), 149248, 0,2);
#endif
}

extern "C" __global__ AICORE void launch_kernel(__gm__ uint8_t *x_handle, __gm__ uint8_t *y_handle, __gm__ uint8_t *loss_handle, __gm__ uint8_t *log_prob_handle, uint64_t fftsAddr)
{
    main_kernel(reinterpret_cast<__gm__ half *>(x_handle),
     reinterpret_cast<__gm__ int *>(y_handle),
     reinterpret_cast<__gm__ half *>(loss_handle),
     reinterpret_cast<__gm__ half *>(log_prob_handle),
     reinterpret_cast<uint64_t>(fftsAddr));
}

extern "C" void call(uint8_t *x_handle, uint8_t *y_handle, uint8_t *loss_handle, uint8_t *log_prob_handle, void *stream)
{
    uint32_t fftsLen{0};
    uint64_t fftsAddr{0};
    rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    launch_kernel<<<8, nullptr, stream>>>(x_handle, y_handle, loss_handle, log_prob_handle, fftsAddr);
}
```

## 定位结论

问题发生在

```python
                for bc in T.serial(c_num):
                    T.copy(x[bn * block_N_2, bc * block_C], x_ub)
                    cast_or_copy(x_32, x_ub, CAST_MODE_LOW2HIGH, block_N_2 * block_C)

                    for n_idx in T.serial(block_N_2):
                        T.tile.sub(x_32[n_idx, :], x_32[n_idx, :], prev_max[n_idx] + prev_sum[n_idx])  # x_c - (x_max + log(sum e^{x_c - x_max}))  # 问题发生在这里

                    cast_or_copy(x_ub, x_32, CAST_MODE_HIGH2LOW, block_N_2 * block_C)
                    T.copy(x_ub, log_prob[bn * block_N_2, bc * block_C])

                    for n_idx in T.serial(block_N_2):
                        if 0 <= y_ub[n_idx] and y_ub[n_idx] < block_C:
                            l_n_32[n_idx] = -x_32[n_idx, y_ub[n_idx]]  # -(x_c - x_max - log(sum e^{x_c - x_max}))
                    T.tile.sub(y_ub, y_ub, block_C)
```

其中  `T.tile.sub(x_32[n_idx, :], x_32[n_idx, :], prev_max[n_idx] + prev_sum[n_idx])  # x_c - (x_max + log(sum e^{x_c - x_max})) `, 由 codegen  转换生成的代码发生了问题，导致该 example 发生了精度问题

对应 codegen 部分的转换 (文件位置：`/src/target/codegen_ascend_pto.cc`) 方法如下：

```python
void CodeGenTileLangAscendPto::BinaryVecOpsCodegen(const CallNode *op,
                                               const std::string &op_name)
```

经过 codegen 转换之后，生成了如下代码：
```c++
        pipe_barrier(PIPE_V);
        TADDS(x_32, x_32, -(prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1)));
```

经过我的验证，生成上述的代码会产生精度问题，正确的代码应该是下面这样的：

```
        pipe_barrier(PIPE_V);
        // 1. 获取当前行对应的标量
        float scalar = -(prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1));
        // 2. 关键：定义一个只代表“一行”的临时视图
        tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> x_32_row_view;
        // 3. 关键：将视图绑定到 x_32 的具体行起始地址 (16896 是 x_32 的基址)
        TASSIGN(x_32_row_view, 16896 + (n_idx_1 * 128) * 4); 
        pipe_barrier(PIPE_ALL);
        // 4. 只对这一行进行 Adds 操作
        TADDS(x_32_row_view, x_32_row_view, scalar);

```

-----
```sh
-exec p op->args[2]->GetTypeKey()
$3 = "tir.Add"

  std::cerr << "[DEBUG] BinaryVecOpsCodegen: Checking args[2]" << std::endl;
  std::cerr << "[DEBUG] args[2] type: " << op->args[2]->GetTypeKey() << std::endl;

      // Debug: Print the type key
    std::cerr << "[DEBUG] IsComplexExpression called, type: " << expr->GetTypeKey() << std::endl;
```

```c++
bool IsComplexExpression(const PrimExpr& expr) {
    // Debug: Print the type key
    std::cerr << "[DEBUG] IsComplexExpression called, type: " << expr->GetTypeKey() << std::endl;

    // Check if it's a CallNode (e.g., buffer.GetValue(index))
    if (expr.as<CallNode>()) {
        std::cerr << "[DEBUG] Detected as CallNode" << std::endl;
        return true;
    }
    // Check if it's an arithmetic operation (Add/Sub/Mul/Div/Mod/etc.)
    // Note: Use tir:: prefix for arithmetic nodes as they're in the tir namespace
    if (expr.as<tir::AddNode>()) {
        std::cerr << "[DEBUG] Detected as AddNode" << std::endl;
        return true;
    }
    if (expr.as<tir::SubNode>()) {
        std::cerr << "[DEBUG] Detected as SubNode" << std::endl;
        return true;
    }
    if (expr.as<tir::MulNode>()) {
        std::cerr << "[DEBUG] Detected as MulNode" << std::endl;
        return true;
    }
    if (expr.as<tir::DivNode>()) {
        std::cerr << "[DEBUG] Detected as DivNode" << std::endl;
        return true;
    }
    if (expr.as<tir::ModNode>() || expr.as<tir::FloorDivNode>() ||
        expr.as<tir::FloorModNode>() || expr.as<tir::MaxNode>() ||
        expr.as<tir::MinNode>()) {
        return true;
    }
    std::cerr << "[DEBUG] Not detected as complex expression" << std::endl;
    return false;
}
```

```
        T.tile.sub(x_32[n_idx, :], x_32[n_idx, :], tile_max[n_idx])

        float scalar = -(prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1));
        auto tile_max_scalar= tile_max.GetValue(n_idx);

        T.tile.sub(x_32[n_idx, :], x_32[n_idx, :], prev_max[n_idx] + prev_sum[n_idx])
        auto x_32_scalar= x_32.GetValue((prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1)));

        pipe_barrier(PIPE_ALL);
        tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> x_32_temp;
        TASSIGN(x_32_temp, 16896 + (n_idx * 128) * 4);
        pipe_barrier(PIPE_ALL);
        TADDS(x_32_temp, x_32_temp, -tile_max_scalar);


        pipe_barrier(PIPE_V);
        // 1. 获取当前行对应的标量
        float scalar = -(prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1));
        // 2. 关键：定义一个只代表“一行”的临时视图
        tl::ascend_pto::TileUbDataND<float, 1, 128, 1, 128> x_32_row_view;
        // 3. 关键：将视图绑定到 x_32 的具体行起始地址 (16896 是 x_32 的基址)
        TASSIGN(x_32_row_view, 16896 + (n_idx_1 * 128) * 4); 
        pipe_barrier(PIPE_ALL);
        // 4. 只对这一行进行 Adds 操作
        TADDS(x_32_row_view, x_32_row_view, scalar);

        pipe_barrier(PIPE_ALL);
        auto scalar= (prev_max.GetValue(n_idx_1) + prev_sum.GetValue(n_idx_1));
        pipe_barrier(PIPE_ALL);
        tl::ascend_pto::TileUbDataND<float, 1, 16, 1, 16> x_32_temp;
        TASSIGN(x_32_temp, 320 + (n_idx_1 * 16) * 4);
        pipe_barrier(PIPE_ALL);
        TADDS(x_32_temp, x_32_temp, -scalar);


```

```修改代码
enum class BinaryOps {
    TADDS,
    TSUBS,
    TMULS,
    TDIVS,
    TMAXS,
    TMINS
};

template <BinaryOps Op, typename T, int32_t shape>
AICORE PTO_INLINE void binarys_tile(int32_t addr,
                int32_t offset, int32_t len, T scalar_value) {
    TileUbDataND<T, 1, shape, 1, shape> temp_ub;
    pto::TASSIGN(temp_ub, addr + offset * len);
    pipe_barrier(PIPE_ALL);
    if constexpr (Op == BinaryOps::TADDS) {
        pto::TADDS(temp_ub, temp_ub, scalar_value);
    } else if constexpr (Op == BinaryOps::TSUBS) {
        pto::TSUBS(temp_ub, temp_ub, scalar_value);
    } else if constexpr (Op == BinaryOps::TMULS) {
        pto::TMULS(temp_ub, temp_ub, scalar_value);
    } else if constexpr (Op == BinaryOps::TDIVS) {
        pto::TDIVS(temp_ub, temp_ub, scalar_value);
    } else if constexpr (Op == BinaryOps::TMAXS) {
        pto::TMAXS(temp_ub, temp_ub, scalar_value);
    } else if constexpr (Op == BinaryOps::TMINS) {
        pto::TMINS(temp_ub, temp_ub, scalar_value);
    }
    pipe_barrier(PIPE_ALL);
}


bool IsComplexExpression(const PrimExpr& expr) {
    // Check if it's an arithmetic operation (Add/Sub/Mul/Div/Mod/etc.)
    // Note: Use tir:: prefix for arithmetic nodes as they're in the tir namespace
    if (expr.as<tir::AddNode>()) {
        return true;
    }
    if (expr.as<tir::SubNode>()) {
        return true;
    }
    if (expr.as<tir::MulNode>()) {
        return true;
    }
    if (expr.as<tir::DivNode>()) {
        return true;
    }
    if (expr.as<tir::ModNode>() || expr.as<tir::FloorDivNode>() ||
        expr.as<tir::FloorModNode>() || expr.as<tir::MaxNode>() ||
        expr.as<tir::MinNode>()) {
        return true;
    }
    return false;
}

void CodeGenTileLangAscendPto::BinaryVecOpsCodegen(const CallNode *op,
                                               const std::string &op_name) {
  std::vector<std::string> var_names;
  std::string operation = (op_name == "TSUBS") ? "TADDS" : op_name;
  for (int i = 0; i < op->args.size() - 2; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  std::string raw_index = PrintExpr(op->args[op->args.size() - 2]);
  std::string final_scalar = (op_name == "TSUBS") ? ("-" + raw_index) : raw_index;
  bool is_call = op->args[2].as<CallNode>() != nullptr;
  if (op->args[2].as<CallNode>() || IsComplexExpression(op->args[2])) {
    std::string index = PrintExpr(op->args[op->args.size() - 2]);
    std::string offset = PrintExpr(op->args[0].as<CallNode>()->args[2]);
    std::string ub_name = var_names[1];
    auto& ub_metadata = ub_data_map_[ub_name];

    // 1. 统一处理标量获取逻辑
    // 如果是 CallNode，从 UB 中 GetValue；如果是复杂表达式，直接使用 index 字符串
    bool is_call = (op->args[2].as<CallNode>() != nullptr);
    std::string scalar_expr = is_call ? (PrintBufferOffset(op->args[2].as<CallNode>()) + ".GetValue(" + index + ")") : index;
    std::string scalar_name = is_call ? (PrintBufferOffset(op->args[2].as<CallNode>()) + "_scalar") : "scalar";
    
    // 这里的 final_scalar 逻辑已经在外部预处理，我们直接定义生成的变量名
    this->stream << "pipe_barrier(PIPE_ALL);\n";
    this->PrintIndent();
    this->stream << "auto " << scalar_name << " = " << scalar_expr << ";\n";
    this->stream << "pipe_barrier(PIPE_ALL);\n";

    // 2. 统一计算处理的列数（防止 TASSIGN 越界或不满足 32 字节对齐）
    std::string loop_num = getValueOrProcess(for_num_map_, index);
    int32_t total_elements = std::stoi(ub_metadata[1]) * std::stoi(ub_metadata[2]);
    int32_t ub_data_temp_col = total_elements / std::stoi(loop_num);

    // 3. 处理符号转换 (TSUBS -> TADDS)
    std::string final_op_name = operation;
    std::string applied_scalar = (op_name == "TSUBS") ? ("-" + scalar_name) : scalar_name;

    this->PrintIndent();
    if (is_call) {
      // 分支 A: 使用 binarys_tile 模板接口
      this->stream << kAscendPtoScope << "binarys_tile<" << kAscendPtoScope << "BinaryOps::" << final_op_name 
                   << ", " << ub_metadata[0] << ", " << ub_data_temp_col << ">("
                   << ub_metadata[3] << ", " << offset << ", " << GetTypeLenString(ub_metadata[0]) 
                   << ", " << applied_scalar << ");\n";
    } else {
      // 分支 B: 手动 TASSIGN 临时变量
      std::string var_name_temp = ub_name + "_temp";
      this->stream << kAscendPtoScope << "TileUbDataND<" << ub_metadata[0] << ", 1, "
                   << ub_data_temp_col << ", 1, " << ub_data_temp_col << "> " << var_name_temp << ";\n";
      this->PrintIndent();
      this->stream << "TASSIGN(" << var_name_temp << ", " << ub_metadata[3] << " + " 
                   << offset << " * " << GetTypeLenString(ub_metadata[0]) << ");\n";
      this->PrintIndent();
      this->stream << "pipe_barrier(PIPE_ALL);\n";
      this->PrintIndent();
      this->stream << final_op_name << "(" << var_name_temp << ", " << var_name_temp << ", " << applied_scalar << ");\n";
    }
  }
  else {
    this->PrintIndent();
    this->stream << operation << "(";
    for (size_t i = 0; i < var_names.size(); ++i) {
      this->stream << var_names[i] << ", ";
    }
    this->stream << final_scalar << ");\n";
  }
  }


  void CodeGenTileLangAscendPto::UnaryVecOpCodegen(const CallNode *op, const std::string& op_name) {
  std::vector<std::string> var_names;

  std::string src_name = PrintExpr(op->args[1].as<CallNode>()->args[1]);
  std::string dst_name = PrintExpr(op->args[0].as<CallNode>()->args[1]);

  std::string src_offset = PrintExpr(op->args[1].as<CallNode>()->args[2]);
  std::string dst_offset = PrintExpr(op->args[0].as<CallNode>()->args[2]);

  std::string src_addr = ub_data_map_[src_name][3];
  std::string dst_addr = ub_data_map_[dst_name][3];

  std::string shape = PrintExpr(op->args[2]);
  std::string ub_type = ub_data_map_[dst_name][0];
  int32_t type_len = GetTypeLen(ub_type);
  for (int i = 0; i < op->args.size() - 1; i++) {
    auto var_name = PrintBufferOffset(op->args[i].as<CallNode>());
    var_names.push_back(var_name);
  }

  if (src_offset != "0" || dst_offset != "0") {
    this->PrintIndent();
    this->stream << kAscendPtoScope << "unary_tile" << "<" << kAscendPtoScope << "UnaryOp::" << op_name << ", " << ub_type << ", " << shape << ">" << "("
    << dst_addr << ", " << src_addr << ", " << dst_offset
    << ", " << src_offset << ", "  << type_len << ");\n";
  } else {
    this->PrintIndent();
    this->stream << op_name << "(";
    for (int i = 0; i < var_names.size(); i++) {
      this->stream << var_names[i];
      if (i != var_names.size() - 1) {
        this->stream << ", ";
      }
    }
    this->stream << ");\n";
  } 
}
```
----

## dump_tensor support

```
# 424
        bool only_in_ascend_copy = false;
        if (!use_info->used_in_cube) {
          bool only_valid_funcs = true;
          for (const auto& func_name : use_info->func_names) {
            if (func_name != "tl.ascend_copy" && func_name != "tl.ascend_dump_tensor") {
              only_valid_funcs = false;
              break;
            }
          }
          if (only_valid_funcs) {
            only_in_ascend_copy = true;
          }
        }

# 472
        bool has_dump_tensor = false;
        if (use_info) {
          for (const auto& func : use_info->func_names) {
            if (func == "tl.ascend_dump_tensor" || func == "tl.dump_tensor") {
              has_dump_tensor = true;
              break;
            }
          }
        }

        bool should_apply = false;
        if (has_dump_tensor && alloc_info->corrected_scope == "shared.dyn") {
          should_apply = true;
        } else if (alloc_info->corrected_scope != alloc_info->original_scope) {
          should_apply = true;
        }

        if (should_apply) {

```

```
# 484
      for (auto it : map) {
          Var target_var = it.first;      
          Layout target_layout = it.second; 

          Buffer found_buffer;
          if (buffer_data_to_buffer_.count(target_var)) {
              found_buffer = buffer_data_to_buffer_[target_var];
          } else {
              for (auto const& kv : buffer_data_to_buffer_) {
                  Var current_var = kv.first;
                  Buffer current_buf = kv.second;
                  if (current_var->name_hint == target_var->name_hint &&
                      current_var->dtype == target_var->dtype) {
                      if (StructuralEqual()(target_layout->InputShape(), current_buf->shape)) {
                          found_buffer = current_buf;
                          break;
                      }
                  }
              }
          }
          if (!found_buffer.defined()) {
              ICHECK(false) << "buffer " << target_var->name_hint << " is not found in the block";
          }
          ICHECK(StructuralEqual()(target_layout->InputShape(), found_buffer->shape));
          annotated_layout_map_.Set(found_buffer, target_layout);
      }
```

