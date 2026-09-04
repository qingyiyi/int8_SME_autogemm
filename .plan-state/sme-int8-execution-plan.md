# INT8 SME AutoGEMM 下一阶段执行计划

## 当前基线

- 已完成：AutoGEMM 侧能生成与参考库同 ABI 的 C driver，并链接现有 SME
  kernel/packer object；远端已确认生成库导出 `cblas_gemm_s8s8s32`、
  `nthreadsM` 和 `nthreadsN`。
- 不改：现有 SME 汇编、A/B packer、导出函数签名、调用方分配的 `sa/sb`。
- 固定合同：`K=2048`，`M/N` 为 2048 的倍数，`alpha=1`，`beta=0`，
  `threadsM=32`，`threadsN=1`。
- 搜索记录默认是测量结果，不等同于正确性已验收；最终候选才运行 `--verify`。

## P1：候选配置生成器

目标：把当前固定 baseline JSON 改为受约束的候选配置，仍然生成同一个 CBLAS
driver 和同一组汇编链接方式。

首轮可搜索参数：

| 参数 | 初始候选 | 固定原因/边界 |
| --- | --- | --- |
| `P` | `64, 128, 256` | 调用方现有 `sa` 容量按 `P=256` 分配；只允许不超过 256。 |
| `R` | `2048, 4096, 8192` | 调用方现有 `sb` 容量按 `R=8192` 分配；只允许不超过 8192。 |
| `Q` | `2048` | 当前 K 和 kernel/packer 合同固定。 |
| `Jblock` | `32` | 暂不开放，避免改变 32 线程 B packing 分工与等待逻辑。 |
| 线程与分区 | `32 x 1`、`region_align=4` | 首轮保持参考实现。 |

交付：候选 JSON schema、参数验证、每个候选的生成 manifest，manifest 明确记录
所需 `sa/sb` 上界、编译命令、汇编 object 和 ABI。

实现约束：`R<8192` 会使一个调用进入多个 N block。生成的 driver 必须先用
barrier 保证全部线程已离开上一 N/K block，再清空每个线程的 B-ready 标记，并
用第二道 barrier 保证全部标记已清空；发布/轮询处执行 OpenMP flush。不得复用
上一 N block 的非空标记，也不得把异步 B packing 改为全局完成后才计算的同步路径。

验收：本机 Python/render/Makefile 静态检查；远端对每个候选执行 `make all check`
和 `nm -D` 导出检查。

## P2：快速远端搜索器

目标：在不重复执行九组正确性测试的前提下，构建、测量和记录 P1 的候选。

流程：

1. 生成候选目录并链接已验证的四个参考汇编 object。
2. 对每个候选做编译和 ABI 检查。
3. 先对一个指定 shape 测量全部候选，用于快速筛选。
4. 将每个 shape 的前 K 个候选再测完整九组 shape。
5. 原始记录保留 `candidate_id`、P/R、shape、库哈希、完整命令和时间；
   未执行验证的记录标记 `correctness=not_checked`，不得当作最终正确结果。

默认排序按 shape 分开保存。若九个 shape 的最优 P/R 不同，后续再决定生成单一
全局 profile，或在保持同一 ABI 的前提下增加按 `(M,N)` 分派的 profile 表。

验收：远端能从任一候选生成 `.so`、解析完整 `cblas_gemm_s8s8s32` 调用时间，
并写出可复现实验记录；失败候选写拒绝记录但不参与排名。

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

## P5：最终正确性与发布候选

- 对 baseline、每个 shape 的最优候选，以及最终合并 profile 执行九组 `--verify`。
- 只把通过验证的记录标记为 `correctness=passed` 并作为最终性能结论。
- 若某个候选失败，回退到该 shape 的下一个已测候选；不修改现有汇编来掩盖错误。

## 当前进度

- P1 代码已完成：`generate_candidates.py` 可生成 9 个 P/R bundle，且保留
  baseline 单 bundle 入口；主机侧生成/约束测试已通过。
- P2 代码已完成：`sweep_candidates.py` 支持 quick shape 初筛、top-k 全 shape
  测量、独立日志和汇总；主机侧伪目标烟测已通过。
- P2 尚待远端验证：需要在 SME 机器上用真实汇编 object、编译器和
  `int8/test_unigemm` 执行批量构建与测量。

## 本次启动记录（2026-09-03 16:31 +08:00）

- 本机已复核 P2：13 个 Python 单元测试通过，Python 与 shell 语法检查通过，
  `git diff --check` 通过；9 个候选 bundle 生成烟测通过。
- 本机没有 SME ISA、BiSheng 工具链或远端参考 object，因此不宣称完成真实
  `make all check`、性能测量或正确性验收。
- 当前唯一待办是远端 P2 门槛：生成 9 个 bundle，批量构建并运行 quick/top-k
  搜索；结果需回传 `sweep_summary.json`、至少一个 build log 和一个完整评测
  log。未带 `--verify` 的记录只能用于候选排序。

## 当前下一步

将本目录同步到远端，先生成 9 个 bundle，再运行 README 中的 P2 命令。回传
`sweep_summary.json`、一个 build log 和一个 full evaluation log 后，再进入 P3
的 AutoGEMM/AutoTVM workload 接入；在此之前不修改 SME 汇编或扩大搜索变量。
