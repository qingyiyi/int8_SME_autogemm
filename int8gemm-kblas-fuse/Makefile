# ============================================
# INT8 GEMM SME - Makefile
# ============================================

# ---- 编译器配置 ----
GCC_DIR    := /data1/env/HPCKit_26.0.RC1/HPCKit/latest/compiler/bisheng
CC         := $(GCC_DIR)/bin/BiSheng-clang
AR         := $(GCC_DIR)/bin/ar
CXX        := $(GCC_DIR)/bin/BiSheng-clang++

# ---- 编译选项 ----
# -O3: 优化级别
# -fPIC: 生成位置无关代码（.so 必需）
# -Wall: 显示警告
CFLAGS     := 	-O2 -fPIC -Wall \
				-mllvm=-disable-lsr \
				-mllvm=-aarch64-insert-extract-base-cost=3 \
				-mllvm=-disable-aggressive-fabs-opt=true \
				-mllvm=-disable-folding-shifts-loads=true \
				-mllvm=-disable-small-spreads-precision-increase=true \
				-mllvm=-ignore-tti-inline-compatible=true

# ARMv9 + SME (⚠️ 需要 GCC 13+，GCC 12 不支持 SME，见下方说明)
ARCH_FLAGS := -march=armv8.6-a+sve+sve2+sme-f64f64+fp16+bf16

# 项目宏定义
DEFINES    := -DUSE_SME_OPT \
              -DINT8_SME \
              -DUSE_SIMPLE_THREADED_LEVEL3 \
              -DCOMPSIZE=1 \
			  -DPARAM_CONF_KP920F_SME

# OpenMP
OPENMP     := -fopenmp=libomp

# 头文件路径
# ⚠️ 如果 common.h 在 OpenBLAS 源码里，需要加上 -I/path/to/openblas/common
INCLUDES   := -I. -I./assemble -I./include

# 汇总
ALL_CFLAGS := $(CFLAGS) $(ARCH_FLAGS) $(DEFINES) $(OPENMP) $(INCLUDES)

# ---- 源文件 ----
# BASE_* is the proven implementation.  The phase-1 shadow implementation is
# deliberately separate so `make test` keeps compiling/linking only the base
# path.  `make test-stage1` is the first target that needs fused sources.
BASE_CPP_SRCS  := int8_gemm.cpp
FUSED_CPP_SRCS := int8_gemm_fused.cpp

# gemm_sme_base.S / int8_gemm_common.S / gemm_sme_reg_defs.h are included,
# rather than assembled directly.
BASE_ASM_SRCS  := assemble/gemm_sme_nt.S \
                  assemble/gemm_sme_nn.S \
                  assemble/gemm_tcopy_zip.S \
                  assemble/gemm_ncopy_unzip.S
FUSED_ASM_SRCS := assemble/gemm_sme_nn_fused.S

# ---- 目标文件 ----
BASE_CPP_OBJS  := $(BASE_CPP_SRCS:%.cpp=%.o)
FUSED_CPP_OBJS := $(FUSED_CPP_SRCS:%.cpp=%.o)
BASE_ASM_OBJS  := $(BASE_ASM_SRCS:assemble/%.S=assemble/%.o)
FUSED_ASM_OBJS := $(FUSED_ASM_SRCS:assemble/%.S=assemble/%.o)
BASE_OBJS      := $(BASE_CPP_OBJS) $(BASE_ASM_OBJS)
FUSED_OBJS     := $(FUSED_CPP_OBJS) $(FUSED_ASM_OBJS)
ALL_OBJS       := $(BASE_OBJS) $(FUSED_OBJS)

# 输出
TARGET     := libint8gemm.so
TEST_SRC   := test_int8gemm_kblas.cpp
TEST_OBJ   := $(TEST_SRC:%.cpp=%.o)
TEST_TARGET:= test_kblas_gemm
STAGE1_TEST_SRC := test_fused_stage1.cpp
STAGE1_TEST_OBJ := $(STAGE1_TEST_SRC:%.cpp=%.o)
STAGE1_TEST_TARGET := test_fused_stage1

# Portable reference checks never use target-only ARCH_FLAGS or target dependencies.
HOST_CXX   ?= g++
HOST_CXXFLAGS ?= -O2 -Wall -Wextra -Werror
HOST_TEST  := tests/test_reference
HOST_DRIVER_TEST := tests/test_fused_driver_host
HOST_DRIVER_HEADERS := tests/reference.hpp int8_gemm.hpp \
                       tests/host_stubs/hbwmalloc.h tests/host_stubs/numa.h \
                       tests/host_stubs/numaif.h
TEST_HEADERS := tests/reference.hpp tests/guarded_matrix.hpp
REFERENCE_FP_FLAGS := -fno-fast-math -frounding-math -ffp-contract=off
TEST_ARGS  ?= --suite smoke
# Start small, then run the 8192x8192 business-size command documented below.
STAGE1_TEST_ARGS ?= --m 2048 --n 2048 --threads 32 --modes 0,1,19 --repeat 1 --seed 42

# ============================================
# 编译规则
# ============================================

.PHONY: all clean check test test-stage1 test-host test-driver-host verify verify-stage1 \
        verify-original-driver

all: $(TARGET)

# 链接动态库：导出原始 API 和阶段 1 shadow API。
$(TARGET): $(ALL_OBJS)
	$(CC) -shared $(OPENMP) -o $@ $^ -lm -lpthread -lnuma
	@echo ">>> 生成成功: $@"
	@echo ">>> 导出符号:"
	@nm -D $@ | grep -E "cblas_gemm_s8s8s32|int8_sme_gemm" || echo "警告: 未找到目标符号!"

# 阶段 0 测试可执行文件：仅链接原始、已确认正确的路径。
test: $(TEST_TARGET)

# 阶段 1 测试可执行文件：链接 base + fused shadow 路径进行逐元素对照。
test-stage1: $(STAGE1_TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJ) $(BASE_OBJS)
	$(CXX) $(OPENMP) -o $@ $^ -lm -lpthread
	@echo ">>> 生成阶段0测试程序（仅原始路径）: $@"

$(STAGE1_TEST_TARGET): $(STAGE1_TEST_OBJ) $(BASE_OBJS) $(FUSED_OBJS)
	$(CXX) $(OPENMP) -o $@ $^ -lm -lpthread
	@echo ">>> 生成阶段1测试程序（原始路径 vs fused shadow）: $@"

# `tests/baseline.sha256` freezes the original driver source.  Phase 1 does
# intentionally add a conditional fused epilogue elsewhere, so checking the
# full phase-0 manifest here would be misleading; this target asserts the file
# the user identified as already correct is still byte-for-byte untouched.
verify-original-driver:
	@grep -E '  int8_gemm\.cpp$$' tests/baseline.sha256 | sha256sum -c -

# Run target regressions explicitly; `test` / `test-stage1` are build-only.
verify: verify-original-driver $(TEST_TARGET)
	./$(TEST_TARGET) $(TEST_ARGS)

verify-stage1: verify-original-driver $(STAGE1_TEST_TARGET)
	./$(STAGE1_TEST_TARGET) $(STAGE1_TEST_ARGS)

test-host: $(HOST_TEST) $(HOST_DRIVER_TEST)
	./$(HOST_TEST)
	./$(HOST_DRIVER_TEST)

# The driver mock compiles the phase-1 C++ driver with ordinary g++ and fake
# empty NUMA/HBM headers.  It deliberately does not compile any SME assembly.
test-driver-host: $(HOST_DRIVER_TEST)
	./$(HOST_DRIVER_TEST)

$(HOST_TEST): tests/test_reference.cpp $(TEST_HEADERS)
	$(HOST_CXX) $(HOST_CXXFLAGS) -std=c++17 $(REFERENCE_FP_FLAGS) $< -o $@ -lm

$(HOST_DRIVER_TEST): tests/test_fused_driver_host.cpp int8_gemm_fused.cpp $(HOST_DRIVER_HEADERS)
	$(HOST_CXX) $(HOST_CXXFLAGS) -std=c++17 -fopenmp -DCOMPSIZE=1 $(REFERENCE_FP_FLAGS) \
		-Itests/host_stubs -I. tests/test_fused_driver_host.cpp int8_gemm_fused.cpp -o $@ -lm

# Keep the test oracle strict even if production flags change later.
$(TEST_OBJ): $(TEST_SRC) $(TEST_HEADERS) int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 $(REFERENCE_FP_FLAGS) -c $< -o $@

$(STAGE1_TEST_OBJ): $(STAGE1_TEST_SRC) $(TEST_HEADERS) int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 $(REFERENCE_FP_FLAGS) -c $< -o $@

# C++ implementation files.
int8_gemm.o: int8_gemm.cpp int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 -c $< -o $@

int8_gemm_fused.o: int8_gemm_fused.cpp int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 -c $< -o $@

# ---- 每个 .S 文件单独定义 ASMNAME ----
# Included SME sources are listed explicitly so a change in an included file
# cannot silently leave a stale object on the target machine.
SME_ASM_DEPS := assemble/gemm_sme_base.S \
                assemble/gemm_sme_packing.S \
                assemble/gemm_sme_nt.S \
                assemble/gemm_sme_reg_defs.h \
                assemble/int8_gemm_common.S
FUSED_ASM_DEPS := $(SME_ASM_DEPS) assemble/gemm_sme_fused_store.S \
                  assemble/gemm_sme_nn_fused.S

# A矩阵pack -> int8_sme_gemm_itcopy
assemble/gemm_tcopy_zip.o: assemble/gemm_tcopy_zip.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_itcopy -DASMFNAME=int8_sme_gemm_itcopy_ -DNAME=int8_sme_gemm_itcopy_ -DCNAME=int8_sme_gemm_itcopy -c $< -o $@

# B矩阵pack -> int8_sme_gemm_oncopy
assemble/gemm_ncopy_unzip.o: assemble/gemm_ncopy_unzip.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_oncopy -DASMFNAME=int8_sme_gemm_oncopy_ -DNAME=int8_sme_gemm_oncopy_ -DCNAME=int8_sme_gemm_oncopy -c $< -o $@

# NN kernel -> int8_sme_gemm_kernel_nn
assemble/gemm_sme_nn.o: assemble/gemm_sme_nn.S $(SME_ASM_DEPS)
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_kernel_nn -DASMFNAME=int8_sme_gemm_kernel_nn_ -DNAME=int8_sme_gemm_kernel_nn_ -DCNAME=int8_sme_gemm_kernel_nn -c $< -o $@

# Phase-1 shadow NN kernel.  The dedicated wrapper itself defines
# FUSED_INVERSE_SCALING, avoiding a Makefile-only behavior switch.
assemble/gemm_sme_nn_fused.o: assemble/gemm_sme_nn_fused.S $(FUSED_ASM_DEPS)
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_kernel_nn_fused -DASMFNAME=int8_sme_gemm_kernel_nn_fused_ -DNAME=int8_sme_gemm_kernel_nn_fused_ -DCNAME=int8_sme_gemm_kernel_nn_fused -c $< -o $@

# NT kernel -> int8_sme_gemm_kernel_nt
assemble/gemm_sme_nt.o: assemble/gemm_sme_nt.S $(SME_ASM_DEPS)
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_kernel_nt -DASMFNAME=int8_sme_gemm_kernel_nt_ -DNAME=int8_sme_gemm_kernel_nt_ -DCNAME=int8_sme_gemm_kernel_nt -c $< -o $@

clean:
	rm -f $(BASE_CPP_OBJS) $(FUSED_CPP_OBJS) $(BASE_ASM_OBJS) $(FUSED_ASM_OBJS) \
	       $(TARGET) $(TEST_OBJ) $(TEST_TARGET) $(STAGE1_TEST_OBJ) \
	       $(STAGE1_TEST_TARGET) $(HOST_TEST) $(HOST_DRIVER_TEST)

# 快速检查
check: $(TARGET)
	nm -D $(TARGET) | grep -E 'cblas_gemm_s8s8s32($$|_)'
