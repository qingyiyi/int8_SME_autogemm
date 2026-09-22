# INT8 GEMM + inverse-scaling epilogue

## 适用范围

当前生产路径只承诺以下调用合同：

- `K` 固定为 `2048`，一次 GEMM 调用对应一个完整的 K panel。
- `M`、`N` 为正数，且都是 `2048` 的倍数；当前重点尺寸是
  `8192 × 8192 × 2048`。
- 列主序 NN，`alpha=1`、`beta=0`，A/B offset 为 0。
- `num_moduli=0`：将 ZA 中 INT32 累加结果的低 8 位写入 C8；
  `num_moduli=1..19`：使用原实现的 19 个奇数模数，结果与原来的
  FP64 `rint + fma` inverse-scaling 逐元素一致。
- C8 是下游消费的最终结果。GEMM 在 ZA 中完成 INT32 累加后，汇编会从 ZA
  取出结果向量、立即 inverse scaling 并直接写 C8；生产 NN 路径不分配、不落地
  C32。生产 API 只接收最终的 `c8/ldc8` 输出参数；不存在 C32 输出指针、
  C32 leading dimension 或 C32 offset-vector 参数。fused NN 汇编中的既有输出
  寄存器槽现在直接保存 C8 byte cursor，`LDC` 保持为调用方传入的 `ldc8` byte stride。

不在本合同内的尺寸（例如非 2048 倍数）不应作为生产验收条件，也不要为了这些
尺寸改动已经确认正确的 GEMM 累加、packing 或分块逻辑。

## 生产实现

现在只有一条 C8-only 生产 API/内核路径：

```text
cblas_gemm_s8s8s8(..., sa, sb, c8, ldc8, num_moduli)
    -> 原有 INT8 SME packing 与 GEMM ZA INT32 累加
    -> 从 ZA 取出 INT32 结果向量
    -> 汇编内 inverse scaling
    -> 直接 C8 tile store（不落地 C32）
```

`int8_gemm.cpp` 不再包含 C++ inverse-scaling 双重循环，也不再包含浮点
`rint/fma` 计算。它只负责：

1. 在一次 GEMM 调用开始时按 `num_moduli` 选择 `modulus/inv_p/neg_p`；
2. 为当前 GEMM tile 填充 `Int8FusedStoreParams`；
3. 通过原 NN kernel 的既有 `buf` 参数把该结构传给汇编。

B1 已经完成上述常量的外提和 ABI 传递，但当前汇编仍使用原来已验证的
`SDIV + MLS` 路径；因此 B1 的目标是改变参数准备，不是改变结果算法或性能。

inverse-scaling 算术位于 `assemble/gemm_sme_inverse_scaling.S`，并由生产
`assemble/gemm_sme_nn.S` 包含到 `int8_sme_gemm_kernel_nn`。汇编在每个 ZA
结果向量刚被 `MOVA` 取出后完成 inverse scaling，再用原有的 `UZP1 × 2 + ST1B`
直接写 C8；`modulus=0` 路径直接写该 INT32 向量的低字节。

非零 modulus 的 quotient 使用已验证的 signed vector `SDIV` 实现。对每个刚从
ZA 取出的 INT32 输出向量，汇编计算 `q = trunc(value / modulus)`，再用 `MLS`
重构余数并修正到 centered-remainder 区间。这里的 `SDIV` 分子是当前输出元素，
不是计算 `1 / modulus`；因此它必须对每个输出向量执行。`modulus=0` 路径跳过
`SDIV`，直接保留 INT32 的低字节。

该实现保留 MOPA、ZA accumulation、packing 和已验证的 `SDIV + MLS` inverse-
scaling 数学；P3 仅将 fused 输出 traversal 从虚拟 C32 坐标改成直接 C8 byte cursor，
并保留最终 `UZP1 × 2 + ST1B` low-byte/wrap store。

### 汇编 ABI

`Int8FusedStoreParams` 通过 kernel 的现有 `buf` 参数传递，布局必须保持为：

| offset | 字段 | 含义 |
|---:|---|---|
| 0 | `c8` | 当前 C8 tile 首地址 |
| 8 | `modulus` | 0 或选中的正奇数模数 |
| 12 | `reserved` | 保留，必须为 0 |
| 16 | `inv_p` | `1.0 / modulus`；mode 0 为 0.0 |
| 24 | `neg_p` | `-modulus`；mode 0 为 0.0 |

结构体总大小为 32 bytes，按 16 bytes 对齐。生产 driver 在一次 public GEMM
调用开始时根据 `num_moduli` 选定这三个常量，然后为每个 NN tile 填入参数块。
表中的 `1.0 / p` 是 `constexpr` 初始化，只在编译期求值，不会在 GEMM 调用或
tile 循环中执行除法。
当前 SDIV epilogue 仍只读取 offset 0/8 的 `c8`/`modulus`；offset 16/24 的
FP64 常量保留 ABI，但当前整数路径不消费它们。P3 中原 kernel 的输出 traversal
直接按 C8 byte cursor 和未缩放的 `ldc8` 推进，不再执行每向量的 `>> 2` 地址映射。
固定生产 SVL=64 bytes 时，一个 `.s` ZA 向量有 16 个 INT32 lane，压缩后对应 16 个
C8 bytes，因此 `SAVE_ZACOL` offset 0..3 显式写到 `pc + {0,16,32,48}`。`buf` 的
实际栈位置由现有 prologue/save-area 约定确定，修改 kernel 参数顺序或 `SAVE_REGS`
布局时必须同步更新汇编。

## 本机验证（不执行 SME 汇编）

x86 主机可运行独立 oracle 和 C++ driver wiring 检查：

```bash
cd /rjs/cxz/huawei/int8gemm-kblas-fuse
make clean
make -B test-host
```

该命令验证：

- FP64 reference、整数 centered-remainder oracle、所有 19 个模数和低字节路径；
- INT32 累加器边界、guard/padding 和固定 K=2048 的标量参考；
- 生产 C++ driver 的 C8 tile 地址、C8 stride、modulus/`inv_p`/`neg_p` 传递和
  1/32 线程分块；
- 直接 C8 cursor 地址：覆盖 2048/8192、非 4 字节对齐的 C8 stride、tile 边界，
  并验证在固定 64-byte SVL 下 `SAVE_ZACOL` 的 0..3 偏移对应 `0/16/32/48` bytes；
- mock kernel 写入 marker，确认 C++ 返回后没有隐藏的 inverse-scaling pass。

host 测试不具备 AArch64 SME 指令执行能力，所以不能替代目标机验收。

## ARM SME 目标机验证

本机不能编译或执行 AArch64 SME 汇编；只以目标业务尺寸
`8192 × 8192 × 2048`、32 threads、all modes 验收。

### 1. 正确性：8192、全 mode

```bash
cd /rjs/cxz/huawei/int8gemm-kblas-fuse
make clean
make -B test

export OMP_PROC_BIND=close
export OMP_PLACES=cores

./test_kblas_gemm \
  --m 8192 --n 8192 --threads 32 \
  --modes all --repeat 1 --seed 42
```

也可以直接执行 `make verify`；它的默认参数就是上面的固定生产合同。预期是 20 个
mode 全部通过，类似：

```text
PASS 20 calls; C8 direct-store probes, output guards and inputs exact.
```

### 2. 性能采样：8192、全 mode

```bash
make clean
make -B perf
./test_fused_gemm 8192 8192 2048 32 all 10 3
```

`mode 0` 是 low-byte 路径，跳过 `SDIV`；`mode 1..19` 使用 centered-remainder
SDIV epilogue。最后一个参数 `3` 是**每个 mode 的 untimed warmup 次数**；它会在
该 mode 的统计前完整执行 GEMM，但不计入 `fused_time`。默认值也是 3，只有需要
故意观察冷启动时才应显式传入 `0`。

正确性测试只分配最终 C8，并检查 C8 padding/guard 和 A/B 输入不被修改；生产路径
不会分配或检查 C32。测试在确定性的 2048 block 边界、128 行/32 列 kernel 边界和
内部坐标处运行独立 INT32 标量 accumulator reference，再以独立的 integer 和 FP64
oracle 验证 C8。

若失败，请保留完整的第一条 `RUN ...`、`PRODUCTION FAIL: ...`，以及编译器报告的
第一处汇编/链接错误。优先确认：

- 执行的是 `make clean && make -B test`；
- `nm test_kblas_gemm | grep int8_sme_gemm_kernel_nn` 存在生产 NN 符号；
- SVE VL/SME SVL 均为 64 bytes；
- OpenMP 确实创建 32 个线程；
- 进程保持 `FE_TONEAREST`。

## 文件说明

- `int8_gemm.cpp`：原生产 driver；只新增汇编 epilogue 所需的 tile 参数传递。
- `int8_gemm.hpp`：kernel 声明和 C++/汇编共享 ABI。
- `assemble/gemm_sme_nn.S`：生产 NN 符号入口。
- `assemble/gemm_sme_inverse_scaling.S`：inverse-scaling epilogue。
- `test_int8gemm_kblas.cpp`：目标机生产路径验收程序。
- `tests/test_reference.cpp`、`tests/reference.hpp`：可移植 reference/oracle。
- `tests/test_production_driver_host.cpp`：不执行 SME 指令的 driver wiring mock。
