# AutoGEMM 交互讲解流程

## 讲解契约

- 读者：熟悉 C/C++，正在建立 TVM/AutoTVM 心智模型。
- 主线形状：`M=26, N=36, K=64`。
- 讲解语言：中文；源码符号、命令和论文公式保留原文。
- 每个里程碑：概念引入 -> 论文证据 -> 源码行号 -> 静态验证命令 -> 理解检查 -> 等待继续。
- 当前主机只做静态检查；任何 Arm/SME 性能结论都标为“目标机待验证”。

## 里程碑脚手架

### M1：GEMM 与最小 TVM 心智模型

论文入口：Background / Matrix Multiplication。

源码入口：`autoGEMM/tvm_asm_kernel/python/template/asm_micro_kernel_template.py:8-106`。

讲解动作：从 C++ 三重循环写出 `C[i,j] += A[i,k] * B[k,j]`，再用最小伪代码建立 `placeholder`、`compute`、`reduce_axis`、`schedule`、`split`、`reorder`、`tensorize` 的对应关系。

静态命令：

```bash
rg -n "placeholder|reduce_axis|te.compute|create_schedule|define_split|reorder|tensorize" \
  autoGEMM/tvm_asm_kernel/python/template/asm_micro_kernel_template.py
nl -ba autoGEMM/tvm_asm_kernel/python/template/asm_micro_kernel_template.py | sed -n '8,106p'
```

停顿检查：能否指出哪一个轴是 reduction、哪三个内轴会交给 tensorize？

### M2：论文架构与微内核生成

论文入口：Fig.1、Eq.(1)-(3)、Section III-A。

源码入口：`tvm_extern_asm_micro_kernel.py` 与 NEON/SVE generator。

讲解动作：解释 `mr x nr`、向量 lane、寄存器预算、AI、prologue/mainloop/epilogue、尾块写回；只讲实现原理，不逐条手算寄存器。

静态命令：

```bash
rg -n "SIMD_LANE|CONST_UNROLL_LANE|micro_kernel_loop_asm|fmla|fmul|st1|ld1w|MRSA|NRSA|RBSA" \
  autoGEMM/tvm_asm_kernel/python/template/gen_asm_code
nl -ba autoGEMM/tvm_asm_kernel/python/template/gen_asm_code/tvm_extern_asm_micro_kernel.py | sed -n '1,125p'
```

停顿检查：为什么 `mr` 增大可能提高复用，却也会耗尽向量寄存器？

### M3：固定形状的 AutoTVM 端到端追踪

论文入口：Fig.1 第 ③/④ 步、Section IV-C。

源码入口：`scripts/utils/tune.sh`、`tune_scheduler.py`、`utils_func/tune.py`、`evaluate.py`、`src/benchmark.cpp`。

讲解动作：把 `26x64x36` 从输入参数追踪到 task/config、PackedB、schedule、TensorIntrin、`call_extern`、LLVM、DSO 和 benchmark；明确哪些值是搜索 knob，哪些是生成器内部派生值。

静态命令：

```bash
rg -n "tune_scheduler|autotvm.task.create|XGBTuner|apply_history_best|tensorize|import_llvm|time_evaluator" autoGEMM
nl -ba autoGEMM/tvm_asm_kernel/python/utils_func/tune.py | sed -n '8,48p'
nl -ba autoGEMM/tvm_asm_kernel/python/utils_func/evaluate.py | sed -n '1,90p'
```

停顿检查：`PackedB` 的四个维度分别承担什么布局责任？

### M4：流水优化、RBSA 与动态微分块

论文入口：Section III-C、Section IV-A/IV-B、Algorithm 1、Fig.5。

源码入口：`experiment/pipeline_optimization/make_c_file_asm_pipeline_expreiment.py` 与 `experiment/RBSA_optimization/make_c_file_asm_RBSA_experiment.py`。

讲解动作：解释 rotating register allocation、load/FMA overlap、epilogue-next-prologue fusion、MRSA/NRSA/RBSA；随后建立“论文 DMT/AI 模型 vs 当前仓库可见实现”的证据等级表，明确不把局部余数重排说成完整 DP。

静态命令：

```bash
rg -n "Pipeline_strategy_level|MRSA|RBSA|NRSA|Dynamic|DMT|sigma|AI" \
  autoGEMM/experiment autoGEMM/tvm_asm_kernel
pdftotext -layout autogemm.pdf - | rg -n "Dynamic Micro-Tiling|Algorithm 1|Rotating Register|Fusing the Epilogue"
```

停顿检查：为什么 `26x36` 的边缘 tile 不能简单全部 padding 成同一个 `5x16`？

### M5：评测、正确性与复现边界

论文入口：Section V、Fig.8-11。

源码入口：`evaluate.py`、`evaluate_scheduler.py`、`benchmark.cpp`、`Makefile`。

讲解动作：区分 compute-only、offline B packing、online packing；检查 Python/C++ 容差差异、工具链依赖、日志覆盖行为和 x86 主机限制；给出目标 Arm 机闭环命令，但不伪造结果。

静态命令：

```bash
nl -ba autoGEMM/tvm_asm_kernel/src/benchmark.cpp | sed -n '28,150p'
nl -ba autoGEMM/tvm_asm_kernel/scripts/utils/evaluate.sh | sed -n '1,80p'
rg -n "rtol|atol|time_evaluator|pack_dso|TVM_CC|clang" autoGEMM
```

停顿检查：为什么只报告 compute-only 会高估带 packing 的真实端到端收益？

### M6：INT8 SME 对照与自动生成可行性评估

论文/源码入口：当前 SME 汇编、`KunpengRuntimeWrappers.cpp`、`.pipeline/architecture/module-sme-kernel.md`、`.pipeline/architecture/decisions.md`。

讲解动作：逐项对照 dtype、ZA/SMOPA、SVL、K padding、A/B packing、C+= 语义、corner、streaming 状态、导出 ABI 和候选 registry；输出三档结论：固定 kernel external-call MVP、参数化 SME 生成器、完整 AutoTVM 搜索。

静态命令：

```bash
rg -n "gemm_kernel_s8s8s32|pack_a_for_sme|pack_b_for_sme|SMOPA|svread|svwrite|__arm_streaming|KERNEL_.*VL" \
  KunpengRuntimeWrappers.cpp kernel kblas
nl -ba .pipeline/architecture/module-sme-kernel.md | sed -n '1,220p'
```

停顿检查：为什么不能只把原模板里的 `instruction` 增加一个 `"sme"` 分支？

## 最终评估输出

评估分为三层：

1. 固定 16x16、`s8 x s8 -> s32`、目标机独立编译的 external-call MVP：复用 TVM host 调度思想，重写 ABI/packing 桥。
2. 多 tile/corner/K-tail 的 SME 生成器：重写 dtype、ZA、predicate、streaming 生命周期和边界合同。
3. 完整 AutoTVM/性能模型：新增 SME 特征、候选 registry、端到端 packing 口径和目标机测量闭环。

每一层都会给出：复用项、必须重写项、阻塞证据、目标机验证命令、预估难度和不应提前承诺的结论。
