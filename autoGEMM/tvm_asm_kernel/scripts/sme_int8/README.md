# INT8 SME 基线生成

该 backend 生成对外导出的 `cblas_gemm_s8s8s32` C driver，同时保持现有的
SME packed 汇编 kernel 与 A/B packing 实现不变。当前仅支持 `K=2048`，且
`M`、`N` 必须为 `2048` 的倍数。

在远端 SME 机器上，先生成构建 bundle：

```bash
cd autoGEMM/tvm_asm_kernel
bash scripts/sme_int8/generate_baseline.sh \
  /path/to/int8gemm-kblas \
  /tmp/autogemm-sme-baseline
```

第一次做等价性验证时，先在参考工程自身目录中构建原始库；然后让生成的
driver 链接该成功构建产生的同一组汇编对象：

```bash
cd /path/to/int8gemm-kblas
make

make -C /tmp/autogemm-sme-baseline \
  REF_ROOT=/path/to/int8gemm-kblas \
  KERNEL_OBJECTS="/path/to/int8gemm-kblas/assemble/gemm_sme_nt.o \
  /path/to/int8gemm-kblas/assemble/gemm_sme_nn.o \
  /path/to/int8gemm-kblas/assemble/gemm_tcopy_zip.o \
  /path/to/int8gemm-kblas/assemble/gemm_ncopy_unzip.o" \
  all check
```

生成的 Makefile 会显式使用参考 BiSheng 编译器。如需改用另一个已经验证过的
编译器二进制，在 `make` 命令中传入
`SME_CC=/path/to/compiler`。

推荐优先使用这种 object-link 模式，因为它只替换生成的 C driver，不会重新
编译或改动 kernel/packer 汇编，也不要求本地汇编源码目录结构与远端完全一致。

确认等价后，后续 kernel 变体才可以使用源码重新编译。如果已经验证的汇编源码
位于 `arm64_sme`，可改用如下方式：

```bash
make -C /tmp/autogemm-sme-baseline \
  REF_ROOT=/path/to/int8gemm-kblas \
  ASM_DIR=/path/to/int8gemm-kblas/arm64_sme \
  all check
```

生成的库名为 `libint8gemm_autogemm.so`。它导出的函数名和参数列表与参考库
一致。第一份配置刻意复现参考实现的参数：

```text
threadsM=32, threadsN=1, P=256, Q=2048, R=8192, Jblock=32
```

## P/R 候选 bundle

在不改动 SME 汇编、packer 或 CBLAS ABI 的前提下，可以批量生成首轮 driver
候选。首轮只开放 `P={64,128,256}` 和 `R={2048,4096,8192}`；`Q=2048`、
`Jblock=32`、`threadsM=32`、`threadsN=1` 和两阶段 B packing 保持固定。若候选
会让一个目标调用跨多个 N/K block（首轮即 `R=2048/4096`），生成的 driver 才会
在每个新 block 前重置 B packing 的 ready 标记；这是避免复用旧标记、同时保留
异步 B packing/计算重叠所必需的同步修正。`R=8192` 基线保持参考实现的一道
barrier，不引入额外同步开销。该差异记录在每个 `manifest.json` 的
`driver_synchronization` 字段中，不会改变已有
`b_packing=reference_two_phase` 配置的兼容性。

```bash
cd /path/to/autoGEMM/tvm_asm_kernel

bash scripts/sme_int8/generate_candidates.sh \
  --config python/sme_int8/baseline_config.json \
  --reference-root /path/to/int8gemm-kblas \
  --output /tmp/autogemm-sme-candidates
```

该命令生成 9 个子目录及 `candidates.json` 索引。每个子目录都是一个独立的
build bundle，包含它自己的 `config.json` 与 `manifest.json`。manifest 会记录
候选的 P/R、driver ABI、汇编未修改声明，以及调用方 `sa/sb` 的容量合同。

当前调用方的 `sa/sb` 没有长度参数，因此生成器只允许不超过参考测试已分配
容量的 P/R 值；库本身不会申请或释放这些 buffer。P2 搜索器会负责批量编译和
测量这些 bundle，本阶段只生成候选，不执行远端编译或性能测试。

## P2 远端快速搜索

把候选目录同步到 SME 远端、准备好已验证的四个汇编/packer object 和测试程序
后，可用下面的命令批量构建。脚本先用 `--quick-shape` 测量全部候选，默认再
对最快的 3 个候选测完整配置中的 9 个 shape：

```bash
cd /path/to/autoGEMM/tvm_asm_kernel

bash scripts/sme_int8/sweep_candidates.sh \
  --candidates-root /tmp/autogemm-sme-candidates \
  --reference-root /data1/cxz/int8gemm-kblas \
  --kernel-objects \
    /data1/cxz/int8gemm-kblas/assemble/gemm_sme_nt.o \
    /data1/cxz/int8gemm-kblas/assemble/gemm_sme_nn.o \
    /data1/cxz/int8gemm-kblas/assemble/gemm_tcopy_zip.o \
    /data1/cxz/int8gemm-kblas/assemble/gemm_ncopy_unzip.o \
  --test-bin /data1/cxz/int8/test_unigemm \
  --results-dir /tmp/sme-results/p_r_sweep \
  --quick-shape 2048,2048,2048 \
  --top-k 3
```

`--kernel-objects` 通过环境变量传给每个 bundle 的 Makefile，避免在命令行中
重复处理 make 变量转义；路径本身请不要包含空格。每个候选会产生独立的
build/quick/full 日志；
最终汇总在 `sweep_summary.json`。搜索阶段不加 `--verify`，quick/full 记录中的
`correctness` 为 `not_checked`。搜索结束后，单独对参考库和入选候选运行评测器
并追加 `--verify` 做正确性验收。

性能比较时，将输入内存申请和 C 清零放在计时区间外，只计量
`cblas_gemm_s8s8s32` 的完整调用时间。正确性可在搜索完成后单独验收。

## 自动评测

评测器默认直接测量配置中的全部 shape，并把能解析到的性能写入性能 JSONL，
适合批量搜索参数。它不会为每个候选自动重复运行正确性测试。

需要验收某个候选时，加上 `--verify`：此时会先逐个运行正确性测试，任意
shape 失败都会拒绝候选并跳过性能测试。拒绝记录写入
`valid_results.jsonl.rejects.jsonl`，且不包含 `latency_seconds` 或 `gflops`。

先构建现有测试程序。测试程序可以继续链接参考 `libint8gemm.so`，评测器会
通过 `LD_PRELOAD` 将待测库插入进程：

```bash
cd /path/to/int8
make
```

然后分别评测参考库和生成库。`--config` 使用生成 bundle 中的 `config.json`：

```bash
cd /path/to/autoGEMM/tvm_asm_kernel

bash scripts/sme_int8/evaluate_candidate.sh \
  --library /path/to/int8gemm-kblas/libint8gemm.so \
  --test-bin /path/to/int8/test_unigemm \
  --config /tmp/autogemm-sme-baseline/config.json \
  --results /tmp/sme-results/reference.valid.jsonl

bash scripts/sme_int8/evaluate_candidate.sh \
  --library /tmp/autogemm-sme-baseline/libint8gemm_autogemm.so \
  --test-bin /path/to/int8/test_unigemm \
  --config /tmp/autogemm-sme-baseline/config.json \
  --results /tmp/sme-results/autogemm.valid.jsonl
```

验收时，在上述命令末尾追加 `--verify`。搜索阶段不要追加该选项；搜索
完成后再对最优候选和参考库各运行一次带 `--verify` 的命令。

如果测试程序运行时还需要参考工程的其他动态库，请照常设置其
`LD_LIBRARY_PATH`；评测器只额外设置 `LD_PRELOAD`。默认每个 shape 最多运行
`3600` 秒，可用 `--timeout` 调整。

性能计时仍来自测试程序报告的完整 `cblas_gemm_s8s8s32` 调用时间；建议在
搜索结束后用 `--verify` 对最终结果做完整验收。
