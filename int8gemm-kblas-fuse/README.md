# INT8 GEMM + inverse scaling 分阶段融合

## 已确认的目标合同

- **K 固定为 2048**，本轮不实现跨 K 块累加，也不为其他 K 宣称正确性。
- 输入沿用当前实现已使用的列主序 NN、`alpha=1`、`beta=0`、A/B/输出偏移为 0。
- 下游只使用 **INT8 输出**。最终优化目标是 `ZA → INT32 寄存器 → inverse scaling → INT8 store`。
- `num_moduli=0` 保留低 8 位；`1…19` 每次选取表中的一个模数，并保持当前 FP64 `rint` + `fma` 语义。
- 验收固定 `FE_TONEAREST`。INT8 结果逐元素完全一致，不使用浮点误差阈值。
- 暂不修改 packing、SMOPA 主循环、P/Q/R 分块常量或生产线程划分。

**当前状态：阶段 0 测试已准备，本机参考测试已通过，目标机基线验收待执行。尚未启用 fused kernel。**

目前生产接口仍要求有效的 INT32 C 缓冲区；不要传 `nullptr`。INT32 C 只用于本阶段诊断和后续过渡，不是最终接口的输出需求。

## 阶段与验收门槛

| 阶段 | 交付 | 进入下一阶段的条件 |
| --- | --- | --- |
| 0 | 冻结当前实现，增加独立参考、目标机回归、保护区检查 | 目标机 smoke/full 和实际 M/N 用例全部通过 |
| 1 | 独立 fused 入口和参数块，先不改变结果处理 | 对阶段 0 的 C32/C8 全部精确一致 |
| 2 | 仅融合 `num_moduli=0`，暂时保留 C32 store | 低 8 位、完整块/尾块、输出步长、保护区全部通过；其他模式保持原路径 |
| 3 | 融合 `num_moduli=1…19`，暂时保留 C32 store | 与原 FP64 语义和独立整数参考全部精确一致 |
| 4 | 删除不再需要的 C32 store，以及对应地址/预取依赖 | 只依赖 C8 的最终路径通过全部回归，再单独做性能测量 |

每阶段目标机验收失败就停在该阶段，不继续叠加优化。本机测试、语法检查和编译成功均不能替代 SME 目标机执行。

## 测试构成

- `tests/reference.hpp`
  - 冻结当前 FP64 inverse scaling 算法和模数表，不从生产文件导入表。
  - 独立整数中心余数 oracle，仅用于测试，不代表生产算法已经改为整数实现。
  - 使用 INT64 累加的朴素 GEMM，固定 K=2048，独立验证 A/B 布局。
- `tests/test_reference.cpp`
  - 所有 20 种模式下穷举 INT16 输入范围，增加 INT32 极值、随机值、模数边界。
  - 包含 FP32 精度反例、低字节回绕、参考函数非法模式检查。
  - 对输出保护区检测器注入破坏，验证它确实能报错。
- `test_int8gemm_kblas.cpp`
  - 调用当前生产 GEMM，检查 C32、C8、输入未被修改和输出 padding/前后 canary。
  - 所有用例运行全部 20 种模式，默认各重复两次；第二次使用非零旧 C，验证 beta=0 覆盖语义。
  - 含大于 `2^24` 的奇数 C（例如 33032065），防止以后误用 FP32。
  - 在每组 OpenMP worker 中检查实际线程数、舍入模式，以及 SVE VL / SME SVL 都为 64 字节；检查不会修改向量长度。
  - 使用普通页对齐内存，不要求测试程序分配 HBM；因此这不是性能基准。

保护区只能检查写越界，不能声称检测了汇编中的全部读越界或线程数据竞争。

### 用例覆盖边界

- smoke：5 组输入，默认 200 次 GEMM 调用。
- full：32 组输入，默认 1280 次 GEMM 调用。
- M/N 尾块覆盖 1、15、16、17、31、32、33、127、128、129 附近，采用单线程隔离验证。
- 32×1 线程布局测试 N 为 32 的倍数，同时包含 M 尾块。
- padded 用例故意令 `lda > M`、`ldb > K`，且 `ldc32 != ldc8`；另外保留紧密布局用例。
- 不扩展现有 B packing 的线程划分边界：本阶段 32 线程自定义用例要求 N 为 32 的倍数；其他 N 尾块使用单线程。**这不代表多线程 N 尾块已经验收。** 如果生产需要它，必须另行补充该配置的基线验证。
- 未覆盖 row-major、转置、非零偏移、任意 alpha/beta、其他 K、0 维度、输入输出重叠或 `C=nullptr`。

## 本机可运行的参考检查

在工作区执行：

```bash
make -C /rjs/cxz/huawei/int8gemm-kblas-fuse test-host
```

只使用 `HOST_CXX`（默认 `g++`），不使用 BiSheng、SME 编译参数、OpenMP、NUMA/HBM 头文件。

成功标记：

```text
PASS host reference: 1711831 inverse cases; guard fault injection; fixed-K GEMM oracle.
This is NOT an SME kernel acceptance test.
```

## 目标机阶段 0 验收

以下路径是当前工作区路径；复制到目标机的其他目录时，相应替换绝对路径。
目标机需要现有工程的 BiSheng/OpenMP/NUMA/HBM 头文件环境，以及支持 INT8 SME 的处理器。

先确认测试对应的是未修改的基线源文件：

```bash
cd /rjs/cxz/huawei/int8gemm-kblas-fuse
sha256sum -c tests/baseline.sha256
```

然后构建并执行：

```bash
make -C /rjs/cxz/huawei/int8gemm-kblas-fuse -B test
make -C /rjs/cxz/huawei/int8gemm-kblas-fuse verify TEST_ARGS='--suite smoke'
make -C /rjs/cxz/huawei/int8gemm-kblas-fuse verify TEST_ARGS='--suite full'
```

`test` 只构建；`verify` 真正执行，失败返回非零状态。`-B` 强制重建，避免现有汇编 include 依赖未完整列入 Makefile 时误用旧对象。

若目标编译器不在 Makefile 原来的路径，可以覆盖 `GCC_DIR`，或直接覆盖 `CC` / `CXX`。测试没有新增系统库依赖，也不会自动安装任何依赖。

实际业务 M/N 也要单独验收，例如：

```bash
/rjs/cxz/huawei/int8gemm-kblas-fuse/test_kblas_gemm \
  --m 1024 --n 1024 --threads 32 --repeat 3 --seed 123
```

这里的 M/N 只是命令示例，不代表已经确认的业务尺寸。自定义 M/N 范围为 1…8192；标量 GEMM oracle 的耗时随 M×N×2048 增长。

成功标记分别为：

```text
STAGE0 PASS: 200 calls across 5 cases; K=2048, modes=0..19. No fused kernel is enabled yet.
STAGE0 PASS: 1280 calls across 32 cases; K=2048, modes=0..19. No fused kernel is enabled yet.
```

若失败，请保留首个失败前的 `RUN` 行和 `STAGE0 FAIL` 行；其中包含尺寸、线程数、步长、数据模式、种子，以及发现不一致时的模数、重复次数、坐标和新旧值。编译失败或环境预检失败也属于未验收，不能跳过后宣称通过。

## 后续实现约束

- 原始 GEMM 路径与 fused 路径分开，方便逐阶段对照；不要在阶段 0 修改生产内核。
- 输出参数块需要明确 C8 tile 地址、ldc8、模式和模数常量；C8 与 C32 的地址推进不能混用。
- FP64 常量和临时寄存器的生存期必须按实际乘加/写回路径核对，不能占用仍在使用的寄存器。
- 取消 C32 store 不等于自然支持 `C=nullptr`。最终阶段必须同步移除或重定向原有 C 指针计算与预取。
- 阶段 4 验收应改为以 C8 为最终输出合同；过渡阶段的 C32 检查不能反过来阻碍 INT8-only 接口。
