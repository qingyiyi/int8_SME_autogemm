# INT8 SME：候选 `.so` 构建与直接动态链接测试

这套流程有且只有两个日常入口：

1. `build_candidate_sos.sh`：生成、编译并检查一批候选 `.so`；**不启动、也不链接
   `test_unigemm`**。
2. `test_candidate_sos.sh`：由你在目标机手动启动。它逐个把候选库原子切换到测试程序
   固定加载的 `/data1/cxz/int8/lib/libint8gemm.so`，然后运行已有的
   `test_unigemm`。

因此，大页、HBM/NUMA 分配、输入初始化、参考结果、正确性比较都仍完全属于
`int8/test_unigemm.cpp`。候选库只负责导出原 ABI：

```text
cblas_gemm_s8s8s32
```

当前默认候选范围是：`K=2048`、`M/N ∈ {2048,4096,8192}`、`alpha=1`、`beta=0`；
搜索参数完全由
[`python/sme_int8/baseline_config.json`](../../python/sme_int8/baseline_config.json)
控制：`P={64,128,256}`、`R={2048,4096,8192}`，以及两个不可拆分的线程组
`(threads_m, threads_n) ∈ {(32,1),(16,2)}`。每个线程组的乘积均为 32，因此默认会构建
`3 × 3 × 2 = 18` 个候选，而不是将 `threads_m` 与 `threads_n` 做笛卡尔积。
A/B packing 与 SME kernel 的完整汇编源码已内置在 AutoGEMM 中。每个候选包会复制这些源码，
再与该候选的 C driver 一起编译、链接，不再要求目标机预先提供四个 `.o` 文件。

## 调参：只修改 `baseline_config.json`

日常搜索不需要再改 Python 常量或在命令行填写 P/R。配置分为两层：

```json
"driver": {
  "threads_m": 32,
  "threads_n": 1,
  "p": 256,
  "q": 2048,
  "r": 8192,
  "jblock": 32,
  "region_align": 4,
  "b_packing": "reference_full_panel_barrier"
},
"search_space": {
  "p": [64, 128, 256],
  "r": [2048, 4096, 8192],
  "thread_groups": [
    {"threads_m": 32, "threads_n": 1},
    {"threads_m": 16, "threads_n": 2}
  ]
}
```

- `shapes`：`test_candidate_sos.sh --all-shapes` 要测试的 `(M,N,K)` 列表。
- `search_space.p`、`search_space.r`、`search_space.thread_groups`：批量构建时真正遍历的
  候选参数；候选数为三者长度的乘积。
- `threads_m` 和 `threads_n` 始终作为一个对象设置。当前只接受 `(32,1)` 或 `(16,2)`，且
  `threads_m * threads_n` 必须为 `32`；不允许例如 `(8,4)` 或 `(32,2)`。
- `driver`：直接调用 `generate_driver.py` 时的默认单个候选；其中的 P/R/线程组也必须出现在
  `search_space` 中。`q`、`jblock`、`region_align` 也都由这里提供给生成的 C driver。
- 当前后端仍限制 `q=K=2048`、P/R 为正的 16 字节对齐值，`jblock` 与 `region_align` 为正的
  4 对齐值。修改搜索空间后，请重新做完整正确性验证；生成的 `manifest.json` 会记录每个候选
  所需的 `sa`/`sb` 容量。

正常构建命令不带 `--p-values` / `--r-values`，就会严格使用 JSON 中的完整搜索空间。这两个
旧选项只保留为兼容用途，并且只能从 JSON 已声明的 P/R 中缩小范围，不能额外加入值。

## 先同步什么

建议把下面两个目录整体同步到目标机的 `autoGEMM/tvm_asm_kernel/`：

```text
python/sme_int8/
scripts/sme_int8/
```

其中新流程实际使用的文件是：

```text
scripts/sme_int8/build_candidate_sos.sh
scripts/sme_int8/test_candidate_sos.sh
python/sme_int8/build_candidate_sos.py
python/sme_int8/test_candidate_sos.py
python/sme_int8/generate_candidates.py
python/sme_int8/generate_driver.py
python/sme_int8/baseline_config.json
python/sme_int8/driver.c.tmpl
python/sme_int8/Makefile.tmpl
python/sme_int8/assembly/
```

目标机还必须已有且已验证：

- `/data1/cxz/int8gemm-kblas/int8_gemm.h`、`common.h` 和 `include/`；
- 已编好的 `/data1/cxz/int8/test_unigemm`；
- `/data1/cxz/int8/lib/libint8gemm.so`（这是测试前会备份、结束后会恢复的原始库）；
- BiSheng-clang、OpenMP、`numactl` 与测试程序所需运行环境。

其中 `--reference-root` 仍用于读取 C driver 所需的头文件；SME 汇编则来自同步过去的
`python/sme_int8/assembly/`。构建过程会在每个候选自己的 `build/asm/` 下生成临时 `.o`，
再链接为 `.so`，但这些 `.o` 是正常中间产物，不是外部输入。

## 步骤 1：只生成所有候选 `.so`

在目标机的 `autoGEMM/tvm_asm_kernel` 下运行：

```bash
bash scripts/sme_int8/build_candidate_sos.sh \
  --reference-root /data1/cxz/int8gemm-kblas \
  --sme-cc /data1/env/HPCKit_26.0.RC1/HPCKit/latest/compiler/bisheng/bin/BiSheng-clang \
  --output /tmp/autogemm-sme-candidates
```

这个命令只会对每个 `(P,R,threads_m×threads_n)` 候选执行等价于下面形式的构建：

```bash
make -C <candidate-dir> REF_ROOT=... SME_CC=... all check
```

它不会读取、编译、链接或运行 `/data1/cxz/int8/test_unigemm`。成功后最方便的产物是：

```text
/tmp/autogemm-sme-candidates/libraries/*.so
/tmp/autogemm-sme-candidates/build_summary.json
/tmp/autogemm-sme-candidates/logs/*.build.log
```

`libraries/` 下的每个文件都有候选标识，例如：

```text
sme-int8-k2048-p256-r8192-j32-t32x1.so
sme-int8-k2048-p256-r8192-j32-t16x2.so
```

如果输出目录已存在，明确确认要刷新时加 `--force`。构建会继续尝试其余候选；只要有一个
失败，最后返回非零，详情位于 `build_summary.json` 与对应的构建日志。

## 步骤 2：按你的运行命令逐个测试候选库

首先建议对你给出的单个大 shape 做正确性测试：

```bash
bash scripts/sme_int8/test_candidate_sos.sh \
  --libraries-dir /tmp/autogemm-sme-candidates/libraries \
  --test-bin /data1/cxz/int8/test_unigemm \
  --install-library /data1/cxz/int8/lib/libint8gemm.so \
  --shape 8192,8192,2048 \
  --verify verify \
  --results-dir /tmp/autogemm-sme-verify-8192
```

每个候选真正被启动时，内部命令严格是：

```bash
OMP_PROC_BIND=false OMP_PLACES=cores OMP_NUM_THREADS=32 \
numactl --all --physcpubind=570-601 -m 15,31 \
/data1/cxz/int8/test_unigemm 8192 8192 2048 kblas verify
```

唯一变化是测试前的原子库切换：

```text
/data1/cxz/int8/lib/libint8gemm.so
    -> 当前候选 libraries/<candidate>.so
```

脚本不会使用 `LD_PRELOAD`，也不会重新链接测试程序。每个候选都在独立的
`test_unigemm` 进程中测试；结束、普通报错、`Ctrl-C`、`SIGTERM`、`SIGHUP` 时都会恢复原
`libint8gemm.so`。它还会对这一路径加排他锁，所以同一时刻不能有两个本脚本的实例并行
切换同一库。运行期间不要手动启动其它依赖该库的测试进程。

若机器异常断电或使用无法捕获的 `SIGKILL`，可利用结果目录中的恢复日志手工恢复：

```bash
bash scripts/sme_int8/test_candidate_sos.sh \
  --recover /tmp/autogemm-sme-verify-8192/install_state.json
```

## 正确性与性能必须分两次运行

你当前的 `test_unigemm.cpp` 在传入 `verify` 时会设置 `num_moduli=1`。因此它只做一次
正确性计算，最后的 `average int8gemm time` 没有可用统计意义。脚本会把这种运行记录成
`correctness_passed`，并刻意不记录性能。

确认候选正确后，再用 `false` 做性能测试：

```bash
bash scripts/sme_int8/test_candidate_sos.sh \
  --libraries-dir /tmp/autogemm-sme-candidates/libraries \
  --test-bin /data1/cxz/int8/test_unigemm \
  --install-library /data1/cxz/int8/lib/libint8gemm.so \
  --shape 8192,8192,2048 \
  --verify false \
  --results-dir /tmp/autogemm-sme-perf-8192
```

这一次内部命令末尾正是 `kblas false`，会保留测试程序的 10 次输入/计时逻辑，并将解析到的
`average int8gemm time` 和计算出的 GFLOPS 写入：

```text
/tmp/autogemm-sme-perf-8192/summary.json
/tmp/autogemm-sme-perf-8192/logs/<candidate>/<M>x<N>x<K>.log
```

要测试 `baseline_config.json` 中配置的全部输入规模，把 `--shape ...` 换成 `--all-shapes`：

```bash
bash scripts/sme_int8/test_candidate_sos.sh \
  --libraries-dir /tmp/autogemm-sme-candidates/libraries \
  --all-shapes \
  --verify verify \
  --results-dir /tmp/autogemm-sme-verify-all
```

CPU 绑定、NUMA 节点和 OpenMP 参数可通过 `--physcpubind`、`--membind`、
`--omp-proc-bind`、`--omp-places`、`--omp-num-threads` 覆盖；默认值就是上面的命令。

## 旧脚本的状态

`run_search.sh`、`evaluate_candidate.sh`、`sweep_candidates.sh` 是之前的实验性入口，内部
采用 `LD_PRELOAD` 方式，不符合当前“测试程序固定加载
`/data1/cxz/int8/lib/libint8gemm.so`、保留大页测试环境”的要求。当前 P2 流程不要调用它们。
后续在新路径经过目标机验证后，再统一删除或迁移这些旧入口，避免一次同时改动过多测试面。
