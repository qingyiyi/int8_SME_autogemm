# INT8 GEMM + inverse-scaling epilogue

## 适用范围

当前生产路径只承诺以下调用合同：

- `K` 固定为 `2048`，一次 GEMM 调用对应一个完整的 K panel。
- `M`、`N` 为正数，且都是 `2048` 的倍数；当前重点尺寸是
  `8192 × 8192 × 2048`。
- 列主序 NN，`alpha=1`、`beta=0`，A/B offset 为 0。
- `num_moduli=0`：将 C32 结果写入 C8 的低 8 位；
  `num_moduli=1..19`：使用原实现的 19 个奇数模数，结果与原来的
  FP64 `rint + fma` inverse-scaling 逐元素一致。
- C8 是下游消费的最终结果。当前实现仍保留原有 C32 tile store，C32 同时作为
  可观测的中间结果和 inverse-scaling 的输入；本版本还没有做 C32-free 优化。

不在本合同内的尺寸（例如非 2048 倍数）不应作为生产验收条件，也不要为了这些
尺寸改动已经确认正确的 GEMM 累加、packing 或分块逻辑。

## 生产实现

现在只有一条生产 API/内核路径：

```text
cblas_gemm_s8s8s32()
    -> 原有 INT8 SME packing 与 GEMM 累加
    -> 原有 C32 tile store
    -> GEMM 汇编 epilogue：inverse scaling
    -> C8 tile store
```

`int8_gemm.cpp` 不再包含 C++ inverse-scaling 双重循环，也不再包含浮点
`rint/fma` 计算。它只负责：

1. 按 `num_moduli` 选择 modulus；
2. 为当前 GEMM tile 填充 `Int8FusedStoreParams`；
3. 通过原 NN kernel 的既有 `buf` 参数把该结构传给汇编。

inverse-scaling 算术位于 `assemble/gemm_sme_inverse_scaling.S`，并由生产
`assemble/gemm_sme_nn.S` 包含到 `int8_sme_gemm_kernel_nn`。汇编使用 signed
integer division (`sdiv`)、余数重构 (`msub`) 和 centered-remainder 修正；这与
19 个正奇数模数下的原 FP64 语义一致。`modulus=0` 路径直接写 C32 的低字节。

没有独立的 fused C++ API、fused kernel 符号、shadow driver 或阶段性 fallback。
原有的 GEMM 累加和 packing 代码保持不变。

### 汇编 ABI

`Int8FusedStoreParams` 通过 kernel 的第 11 个参数传递，布局必须保持为：

| offset | 字段 | 含义 |
|---:|---|---|
| 0 | `c32` | 当前 C32 tile 首地址 |
| 8 | `c8` | 当前 C8 tile 首地址 |
| 16 | `ldc32` | C32 leading dimension（元素数） |
| 24 | `ldc8` | C8 leading dimension（元素数） |
| 32 | `rows` | 当前 tile 有效行数 |
| 40 | `cols` | 当前 tile 有效列数 |
| 48 | `modulus` | 0 或选中的正奇数模数 |
| 52 | `reserved` | 保留 |

当前 epilogue 在正常 C32 store 完成后读取 `sp+184` 的 `buf` 参数，因此不能
改变原 kernel 的栈参数布局，不能把 C32 store 简单删除后继续复用本 ABI。

## 本机验证（不执行 SME 汇编）

x86 主机可运行独立 oracle 和 C++ driver wiring 检查：

```bash
cd /rjs/cxz/huawei/int8gemm-kblas-fuse
make clean
make -B test-host
```

该命令验证：

- FP64 reference、整数 centered-remainder oracle、所有 19 个模数和低字节路径；
- INT32 边界、guard/padding 和固定 K=2048 的参考 GEMM；
- 生产 C++ driver 的 tile 地址、`ldc32 != ldc8`、modulus 传递、1/32 线程分块；
- C++ 返回后不会再次执行 inverse-scaling（mock kernel 写入 marker）。

host 测试不具备 AArch64 SME 指令执行能力，所以不能替代目标机验收。

## ARM SME 目标机验证

目标机上必须强制重建，确保 `gemm_sme_inverse_scaling.S` 被重新包含：

```bash
cd /rjs/cxz/huawei/int8gemm-kblas-fuse
make clean
make -B test
make -B test-driver-host
make check
```

先跑 2048 基准尺寸，再跑用户重点尺寸：

```bash
./test_kblas_gemm --m 2048 --n 2048 --threads 1  --repeat 1 --seed 42
./test_kblas_gemm --m 2048 --n 2048 --threads 32 --repeat 1 --seed 42
./test_kblas_gemm --m 8192 --n 8192 --threads 32 --modes 0,1,19 --repeat 1 --seed 42
```

也可以使用 Makefile 封装的命令：

```bash
make verify
make verify-large
```

2048 验证默认覆盖 `num_moduli=0..19`；`make verify-large` 默认在 8192 重点尺寸
覆盖低字节路径和模数表两端（`--modes 0,1,19`），需要全模数回归时可显式传
`--modes all`。测试对每个实际 C32 元素穷举检查
`C8 == inverse_scaling(C32)`，并检查 C32/C8 的 padding/guard 以及 A/B 输入不被
修改。由于业务最小尺寸已经是 `2048 × 2048`，测试不会尝试构造不可用的
`M × N × K` 标量 reference；它会在每个 2048 块边界、128 行/32 列 kernel 边界
和确定性的内部坐标上运行独立 C32 点 reference。C32 的完整 GEMM 累加路径仍以
已经验证过的原始实现为准。自定义 `--m/--n` 必须是正的 2048 倍数，最大为 8192；
32 线程路径还要求 `N` 能被 32 整除。

目标机成功时应看到类似：

```text
PRODUCTION PASS: ... business-shape cases; K=2048, modes=....
```

若失败，请保留完整的第一条 `RUN ...` 和 `PRODUCTION FAIL: ...` 输出，以及
编译器报告的第一处汇编/链接错误。优先检查：

- 是否执行了 `make clean && make -B test`；
- `nm test_kblas_gemm | grep int8_sme_gemm_kernel_nn` 是否存在生产 NN 符号；
- SVE VL/SME SVL 是否均为 64 bytes；
- OpenMP 是否确实创建了 1 或 32 个线程；
- 进程是否保持 `FE_TONEAREST`。

## 文件说明

- `int8_gemm.cpp`：原生产 driver；只新增汇编 epilogue 所需的 tile 参数传递。
- `int8_gemm.hpp`：kernel 声明和 C++/汇编共享 ABI。
- `assemble/gemm_sme_nn.S`：生产 NN 符号入口。
- `assemble/gemm_sme_inverse_scaling.S`：inverse-scaling epilogue。
- `test_int8gemm_kblas.cpp`：目标机生产路径验收程序。
- `tests/test_reference.cpp`、`tests/reference.hpp`：可移植 reference/oracle。
- `tests/test_production_driver_host.cpp`：不执行 SME 指令的 driver wiring mock。
