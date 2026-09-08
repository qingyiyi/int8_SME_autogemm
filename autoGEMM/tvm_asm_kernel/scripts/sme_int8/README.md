# INT8 SME：一条命令生成、验证并选出最佳 `.so`

正常使用只需要运行 `run_search.sh`。它会复用现有且已验证的 SME kernel / A、B
packer object，不修改汇编，也不会把 `int8_gemm.c` 链接进候选库；每个候选只生成
自己的 CBLAS driver，并导出同一个函数：

```text
cblas_gemm_s8s8s32
```

当前固定合同如下：`K=2048`、`M/N` 是 `2048` 的倍数、`alpha=1`、`beta=0`、
`threadsM=32`、`threadsN=1`。`sa/sb` 仍由调用方分配，库内不会申请或释放它们。

默认搜索 `P={64,128,256}`、`R={2048,4096,8192}` 的全部 9 个组合。其中
`P=256,R=8192` 就是此前的参考 baseline，不需要再单独生成一次。

## 远端一次性执行

先确保远端已有：

- 最新的 `int8gemm-kblas/int8_gemm.h`（建议 `int8_gemm.c` 也同步，保证参考库一致）；
- 四个已验证的 `.o`：SME NT/NN kernel 与 A/B packer；
- 已编译、可执行的 `int8/test_unigemm`；
- BiSheng-clang / OpenMP 运行环境；若运行测试程序还需要其它动态库，也先设置好
  `LD_LIBRARY_PATH`。

将本目录对应的整个 `python/sme_int8/` 和 `scripts/sme_int8/` 同步到远端后，在
`autoGEMM/tvm_asm_kernel` 下运行：

```bash
bash scripts/sme_int8/run_search.sh \
  --reference-root /data1/cxz/int8gemm-kblas \
  --kernel-objects \
    /data1/cxz/int8gemm-kblas/assemble/gemm_sme_nt.o \
    /data1/cxz/int8gemm-kblas/assemble/gemm_sme_nn.o \
    /data1/cxz/int8gemm-kblas/assemble/gemm_tcopy_zip.o \
    /data1/cxz/int8gemm-kblas/assemble/gemm_ncopy_unzip.o \
  --test-bin /data1/cxz/int8/test_unigemm \
  --sme-cc /data1/env/HPCKit_26.0.RC1/HPCKit/latest/compiler/bisheng/bin/BiSheng-clang \
  --output /tmp/autogemm-sme-final
```

这就是唯一的正常入口。它按下面顺序完成所有工作：

```text
9 个 P/R driver
    ↓
逐个 make all check + 动态导出 ABI 检查
    ↓
每个候选依次验证全部 9 个 (M,N,K) 规模
    ↓
只有全部正确的候选才测全部 9 个规模的性能
    ↓
按全局目标排序，复制第一名到 best/libint8gemm_autogemm.so
```

默认全局目标是九个 shape 的 GFLOPS 几何平均值。若你的验收更关心九个调用的总
耗时，可追加：

```bash
--objective total_latency
```

每个 candidate 的验证失败会记录为 `correctness_failed`，不会执行该 candidate 的
性能测试；缺少任一个 shape 的性能记录也不会参与排名。若没有任一候选同时通过完整
正确性和完整性能测量，脚本会非零退出，并且不会为这一次运行发布或覆盖最佳库。

## 输出在哪里

上例成功结束后，最需要的文件是：

```text
/tmp/autogemm-sme-final/best/libint8gemm_autogemm.so
/tmp/autogemm-sme-final/best_candidate.json
/tmp/autogemm-sme-final/summary.json
```

`best/libint8gemm_autogemm.so` 是实际复制出的 `.so`，不是符号链接，因此可以直接
给原来的测试程序使用或单独同步到其它位置：

```bash
LD_PRELOAD=/tmp/autogemm-sme-final/best/libint8gemm_autogemm.so \
  /data1/cxz/int8/test_unigemm 2048 2048 2048 kblas verify
```

完整的候选源、每个构建/ABI/评测日志及 JSONL 原始测量记录在：

```text
/tmp/autogemm-sme-final/candidates/
/tmp/autogemm-sme-final/runs/<run-id>/logs/
/tmp/autogemm-sme-final/runs/<run-id>/records/
```

`summary.json` 会列出每个候选的 `P/R`、构建状态、正确性状态、九个 shape 的记录、
总延迟、GFLOPS 几何平均值、排名和 winner。`best_candidate.json` 额外记录最终库与
源候选库的 SHA-256，便于回溯。

## 重跑与调试

同一个输出目录重跑时，加 `--force`。它会重新生成/构建候选并创建新的
`runs/<run-id>/`，旧运行日志不会与新记录混在一起：

```bash
bash scripts/sme_int8/run_search.sh ... --output /tmp/autogemm-sme-final --force
```

`--skip-generate`、`--skip-build` 和 `--skip-verify` 只用于调试或恢复中断的实验。
特别是 `--skip-verify` 会把记录标成 `not_checked`，并且**绝不会**产生新的
`best/libint8gemm_autogemm.so`。

`generate_baseline.sh`、`generate_candidates.sh`、`sweep_candidates.sh` 和
`evaluate_candidate.sh` 仍保留给开发/定位问题使用；日常搜索和最终选库不需要手动
调用它们。
