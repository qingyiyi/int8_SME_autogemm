# INT8 SME AutoGEMM 下一阶段执行计划

## 当前基线

- 已完成：AutoGEMM 侧能生成与参考库同 ABI 的 C driver，并链接现有 SME
  kernel/packer object；远端已确认生成库导出 `cblas_gemm_s8s8s32`、
  `nthreadsM` 和 `nthreadsN`。
- 不改：现有 SME 汇编、A/B packer、导出函数签名、调用方分配的 `sa/sb`。
- 固定合同：`K=2048`，`M/N` 为 2048 的倍数，`alpha=1`，`beta=0`，
  `threadsM=32`，`threadsN=1`。
- 同步基线（2026-09-07）：以最新 `int8gemm-kblas/int8_gemm.c/.h` 为准。每个
  `(N,K)` panel 用两道 barrier 完成「上一 panel 结束 → 并行完整 pack B → B 全部
  就绪 → pack A + kernel」；旧版 ready-flag snoop、轮询、`omp flush` 和 B-pack /
  compute overlap 已取消。
- 最终搜索默认是正确性门控：任何 candidate 必须先通过全部九组 `--verify`，才会
  测性能或进入全局排名；未验证记录只能由明确的调试选项产生，不能发布为最终库。

## P1：候选配置生成器

目标：把当前固定 baseline JSON 改为受约束的候选配置，仍然生成同一个 CBLAS
driver 和同一组汇编链接方式。

首轮可搜索参数：

| 参数 | 初始候选 | 固定原因/边界 |
| --- | --- | --- |
| `P` | `64, 128, 256` | 调用方现有 `sa` 容量按 `P=256` 分配；只允许不超过 256。 |
| `R` | `2048, 4096, 8192` | 调用方现有 `sb` 容量按 `R=8192` 分配；只允许不超过 8192。 |
| `Q` | `2048` | 当前 K 和 kernel/packer 合同固定。 |
| `Jblock` | `32` | 暂不开放，避免改变 32 线程 B packing 分工和 kernel 列块大小。 |
| 线程与分区 | `32 x 1`、`region_align=4` | 首轮保持参考实现。 |

交付：候选 JSON schema、参数验证、每个候选的生成 manifest，manifest 明确记录
所需 `sa/sb` 上界、编译命令、汇编 object 和 ABI。

实现约束：`R<8192` 会使一个调用进入多个 N block。每一个 N/K block 都沿用参考
代码的两道 barrier：第一道确保全部线程已离开上一 block 后才覆盖共享 `sb`，第二
道确保该 block 的 B 已完整 pack 后才开始 A packing 和 SME kernel。不得重新引入
B-ready snoop、轮询或 `omp flush`；当前参考协议本身不包含 B-pack / compute overlap。

验收：本机 Python/render/Makefile 静态检查；远端对每个候选执行 `make all check`
和 `nm -D` 导出检查。

## P2：远端完整搜索器

目标：一次命令生成、构建、验证、测量和发布 P1 的候选，不让“性能好但未验证”的
库进入最终结果。

流程：

1. 生成候选目录并链接已验证的四个参考汇编 object。
2. 对每个候选做编译和 ABI 检查。
3. 对每个候选的全部九个 shape 先运行正确性测试；任一 shape 失败，该 candidate
   标记 `correctness_failed`，后续不测性能。
4. 只对完整正确性通过的 candidate 测全部九个 shape 的完整
   `cblas_gemm_s8s8s32` 调用时间。
5. 使用九个 GFLOPS 的几何平均值作为默认全局排序；可选择总延迟作为替代目标。
6. 只有完整测量的第一名会复制到
   `best/libint8gemm_autogemm.so`；没有合格者则非零退出且不发布新库。

交付入口：`scripts/sme_int8/run_search.sh`。它在一个输出根目录中保留候选源、每个
build/ABI/evaluation 日志、JSONL 原始记录、`summary.json`、`best_candidate.json` 和
最终 `.so`。旧的 `generate_*`、`sweep_*`、`evaluate_*` 脚本仅保留作开发定位工具。

验收：远端一次 `run_search.sh` 可从九个 P/R 候选中产生可复现的 summary，且只有
九组正确性均通过、九组性能均可解析的候选可成为 winner。

## P3：AutoGEMM/AutoTVM 适配

目标：把 P1 的受约束候选空间接入 AutoGEMM，而不是让 TVM 0.10 直接生成
SME `smopa` 汇编。

- 新增独立的 SME driver workload/config，不能修改现有 FP32 NEON/SVE template。
- AutoTVM 负责候选枚举、搜索顺序和记录；候选构建仍由 SME 专用 Makefile 和
  现有汇编 objects 完成。
- 先实现可重复的网格搜索路径，再根据远端 TVM 0.10 实际 builder/runner 行为
  增加 AutoTVM adapter，避免在本机猜测旧版 TVM 的测量接口。

验收：同一 `(M,N,K,P,R)` 产生稳定 workload key 和可追溯结果；不修改 kernel
汇编，不在库内分配 `sa/sb`。

## P4：逐步扩大搜索空间

P1-P3 稳定后，按一次只增加一个变量的顺序扩展：

1. `Jblock`，同时添加 `R >= Jblock * threadsM` 等分工约束。
2. M/N 分区和线程数；线程数变化需要调用方重新确认 `sa/sb` 容量。
3. 现有汇编中的 prefetch 编译变体，包括 C prefetch 的位置、偏移和次数；
   `no-prefetch` 保留为基线。
4. 只有准备修改汇编时，才进入 SME micro-kernel 形状、K loop 和流水调优。

## P5：后续扩展后的最终正确性与发布候选

- P2 已对首轮 P/R 全部候选实行正确性门控；P5 适用于未来扩大搜索变量后的重复验收。
- 对每一轮的新候选仍执行完整九组 `--verify`，只有完全通过者才能测性能和发布。
- 若某个候选失败，记录失败并让全局排序自动排除它；不修改现有汇编来掩盖错误。

## 当前进度

- P1 代码已完成：`generate_candidates.py` 可生成 9 个 P/R bundle，且保留
  baseline 单 bundle 入口；2026-09-07 已将模板、缓冲区合同和 manifest 从旧 snoop
  协议同步为最新参考 C driver 的完整 panel barrier 协议；本机静态复验已通过。
- P2 总控入口已完成：`run_search.py` / `run_search.sh` 生成九个 candidate、逐个
  构建和 ABI 检查、逐个九 shape 正确性验证、仅对通过者做九 shape 测量，并按全局目标
  选择和复制最佳 `.so`。旧 `sweep_candidates.py` 保留为快速调试工具，不再是最终入口。
- 本机已完成静态复验：24 个 Python 单元测试通过，Python 与 shell 语法检查通过。
  其中包含对真实 `int8/test_unigemm.cpp` 性能输出格式、`LD_PRELOAD` 注入顺序和
  `verify` / `0` 命令行参数合同的 evaluator 回归测试。
- P2 尚待远端真实验证：需要在 SME 机器上用真实汇编 object、编译器和
  `int8/test_unigemm` 运行 `run_search.sh`，才能取得真实性能和最终 winner。

## 本次启动记录（2026-09-03 16:31 +08:00）

- 本机已复核 P2：24 个 Python 单元测试通过，Python 与 shell 语法检查通过，
  `git diff --check` 通过；9 个候选 bundle 生成烟测通过。
- 本机没有 SME ISA、BiSheng 工具链或远端参考 object，因此不宣称完成真实
  `make all check`、性能测量或正确性验收。
- 当前唯一待办是远端 P2 门槛：执行一次完整正确性门控搜索；结果需回传
  `summary.json`、`best_candidate.json`、至少一个 build log 和一个完整评测 log。
  未带 `--verify` 的记录只能用于开发调试，不能选出最终 best 库。

## 本次复验记录（2026-09-07）

- 已确认 `int8/test_unigemm.cpp` 的实际性能行是
  `average int8gemm time = <seconds>`；`evaluate_candidate.py` 的解析正则与其一致。
- 已确认每次调用参数严格为
  `test_unigemm M N K kblas verify`（正确性阶段）和
  `test_unigemm M N K kblas 0`（性能阶段）；候选库通过 `LD_PRELOAD` 排在已有
  preload 库之前注入。
- 已运行完整本机复验：24/24 Python 单元测试通过，`py_compile`、`bash -n`、根仓库
  与 `autoGEMM` 嵌套仓库的 `git diff --check` 均通过。
- 本机仍不具备 BiSheng、SME ISA、远端四个 object 或 HBM/NUMA 测试环境；因此上述
  结果不替代远端 `make all check`、九 shape 正确性或真实性能验收。

## 当前下一步

将最新 `int8_gemm.c/.h` 与包含 `run_search.py/run_search.sh` 的 SME AutoGEMM 目录
同步到远端，直接运行 README 中的一条总控命令。回传 `summary.json`、
`best_candidate.json`、一个 build log 和一个 evaluation log 后，再进入 P3 的
AutoGEMM/AutoTVM workload 接入；在此之前不修改 SME 汇编或扩大搜索变量。
