# INT8 GEMM + inverse scaling 分阶段融合

## 范围、接口与当前状态

本次只覆盖以下实际业务合同：

- `K` **固定为 2048**；不支持多 K-panel 的部分和再做 inverse scaling。
- 阶段 1 验收的 `M/N` 仅为 `2048` 的倍数，当前测试程序允许
  `2048 / 4096 / 6144 / 8192`；重点用例是 **`8192 × 8192 × 2048`**。
- 输入继续使用现有列主序 NN、`alpha=1`、`beta=0`、A/B/C 偏移为 0 的合同。
- 下游最终只消费 `INT8`。不过**阶段 1 仍保留 C32 shadow 输出**，仅用于与已验证原路径逐元素对照；此阶段不能传 `C == nullptr`。
- `num_moduli=0` 表示保留 C32 的低 8 位；`1…19` 对应既有的 19 个奇数模数，结果必须与原 FP64 `rint + fma` 路径逐元素一致。

**阶段 1 代码已完成，但尚未在 ARM SME 目标机实际汇编和运行验收。**

- 用户已确认正确的
  `/rjs/cxz/huawei/int8gemm-kblas-fuse/int8_gemm.cpp` **没有被修改**。
  `make verify-original-driver` 会校验它的冻结 SHA-256。
- 阶段 1 新增了独立入口 `cblas_gemm_s8s8s32_fused()` 和独立 SME 符号
  `int8_sme_gemm_kernel_nn_fused()`；原始入口与原始 kernel 保持为对照基线。
- fused API 只在 `K=2048` 且 `M/N` 均为正的 `2048` 倍数时进入新路径；其余
  尺寸直接回退到原 API，因此不会把阶段 1 用于历史的 `512×64` 问题。
- fused kernel 目前的流程是：

  ```text
  原 SME GEMM 计算 ZA → 原有 C32 store → kernel 内读取本 tile 的 C32
  → inverse scaling → C8 store → 返回
  ```

  因此它是安全的 **shadow fused** 版本，不是最终的 C32-free 优化版本。
  这样一旦出现不一致，可以先比较 C32，再比较 C8，定位边界清晰。

不为历史上的 `M=512, N=64, K=2048, threads=32` 基线失败修改
`int8_gemm.cpp`：该尺寸不属于本次业务合同，也不应作为阶段 1 的准入条件。

## 分阶段交付与停点

| 阶段 | 交付 | 正确性门槛 |
| --- | --- | --- |
| 0 | 冻结已确认正确的原始 driver/kernel | `int8_gemm.cpp` 的哈希一致；原路径只作为对照，不为非业务尺寸改动 |
| 1（当前） | 独立 fused API；保留 C32 store，在同一次 kernel 调用结束前生成 C8 | 原始路径与 fused 路径的 C32、C8、padding/guard、A/B 输入全部精确一致 |
| 2 | 将 inverse scaling 移近 ZA/寄存器结果，仍暂时保留 C32 诊断 store | 阶段 1 的全部逐元素测试继续通过，并单独测量性能变化 |
| 3 | 去掉 C32 store/read，形成只写 C8 的最终路径 | 以独立 C32 oracle / 阶段 1 金样为依据，C8 精确一致；接口不再依赖 C32 |

每一阶段失败都停在当前阶段修复；不把性能优化和语义改动叠加在同一次修改中。

## 已有的本机检查（不等于 SME 验收）

在当前 x86 主机可执行：

```bash
cd /rjs/cxz/huawei/int8gemm-kblas-fuse
make test-host
```

它会运行两类检查：

```text
PASS host reference: 1711831 inverse cases; guard fault injection; fixed-K GEMM oracle.
This is NOT an SME kernel acceptance test.
PASS host fused-driver mock: fixed-K tiling, business-shape fallback, C32/C8 tile addresses, strides, modes, padding and 1/32-thread paths verified.
```

- `tests/test_reference.cpp`：冻结原 FP64 inverse scaling、独立整数 centered-remainder oracle、与阶段 1 `sdiv + msub` 分支等价的标量模型、INT32 极值和保护区检查。
- `tests/test_fused_driver_host.cpp`：以 mock packer/kernel 编译独立 fused C++ driver，验证 K=2048 分块、非业务尺寸回退原 API、C32/C8 tile 地址、`ldc32 != ldc8`、模式到模数映射、padding，以及 1/32 线程路径。
- 两者都**不执行** AArch64 SME 汇编，不能替代目标机验收。

本机还已确认 `make test` 的链接输入只包含原始 base 对象；它不会链接
`int8_gemm_fused.o` 或 `gemm_sme_nn_fused.o`。阶段 1 对照程序仅由
`make test-stage1` 构建。

## ARM SME 目标机：阶段 1 验收顺序

目标机需要现有 BiSheng/OpenMP/NUMA/HBM 环境和支持 INT8 SME 的 CPU。
下面的 `/rjs/cxz/huawei/int8gemm-kblas-fuse` 是当前工作区路径；复制到其他
目录时请替换路径。

### 1. 确认原 driver 没有被触碰，并强制重建 fused 测试

```bash
cd /rjs/cxz/huawei/int8gemm-kblas-fuse
make verify-original-driver
make -B test-stage1
```

`make test-stage1` 只构建；它不会执行测试。`-B` 用于确保汇编 include 文件
发生变化后不会复用旧对象。

若目标机的编译器路径不同，可覆盖 `GCC_DIR`，或者直接覆盖 `CC` / `CXX`；
本次没有新增系统库依赖，也不会自动安装任何依赖。

### 2. 先验证最低风险的低字节路径

```bash
./test_fused_stage1 \
  --m 2048 --n 2048 --threads 32 \
  --modes 0 --repeat 1 --seed 42
```

### 3. 再验证两个有代表性的模数 inverse scaling 路径

```bash
./test_fused_stage1 \
  --m 2048 --n 2048 --threads 32 \
  --modes 1,19 --repeat 1 --seed 42
```

### 4. 在小业务尺寸覆盖全部 20 个模式

```bash
./test_fused_stage1 \
  --m 2048 --n 2048 --threads 32 \
  --modes all --repeat 1 --seed 42
```

### 5. 最后跑实际重点尺寸

```bash
./test_fused_stage1 \
  --m 8192 --n 8192 --threads 32 \
  --modes 0,1,19 --repeat 1 --seed 42
```

成功时程序输出：

```text
STAGE1 PASS: ... original-vs-fused calls exact; C32 shadow, C8, guards and inputs verified.
```

阶段 1 程序刻意使用不同的 `ldc32` 和 `ldc8`，并检查 C32/C8 前后 guard、列间
padding 和 A/B 输入未被改写。对 `8192 × 8192 × 2048`，它不重新做极慢的朴素
完整 GEMM；它以用户已确认正确的原 GEMM 为 C32 对照，并对每一个 C32/C8 元素执行
独立 inverse-scaling oracle 校验。

若失败，请保留完整的第一条 `RUN STAGE1 ...` 和 `STAGE1 FAIL: ...` 输出，尤其是
`mode`、`row`、`col`、`C32`、`C8`、`integer8`。若是汇编/链接失败，也请保留完整
命令和第一处报错；可额外检查：

```bash
nm test_fused_stage1 | grep int8_sme_gemm_kernel_nn_fused
```

不要用旧的 `verify TEST_ARGS='--suite smoke/full'` 作为本次阶段 1 的通过条件：
其中含有本次不支持、且用户不需要的非 `2048` 倍数尺寸。

## 阶段 1 实现约束

- 不修改 `/rjs/cxz/huawei/int8gemm-kblas-fuse/int8_gemm.cpp`。
- 原 API `cblas_gemm_s8s8s32()`、原 SME kernel 和 `make test` 保持为基线对照；
  只有显式调用 `cblas_gemm_s8s8s32_fused()` / `test_fused_stage1`，且尺寸满足
  `K=2048`、`M/N` 为正的 `2048` 倍数时，才进入 fused 路径。
- `Int8FusedStoreParams` 是 C++ 到汇编的稳定 ABI：包含 C32/C8 tile 首地址、两套
  leading dimension、有效 rows/cols 和模数。C8 的列步长绝不能误用 C32 的步长。
- 阶段 1 的模数实现使用整数 signed division + centered remainder。因为 19 个模数
  全为正奇数，整数输入不存在 0.5 tie；它与既有 FP64 `rint + fma` 结果等价。
- 最终删除 C32 store 时，必须同步检查原 kernel 的 C 指针计算、预取和参数合同；
  不能仅仅把 C32 store 指令删掉后就允许 `C == nullptr`。
