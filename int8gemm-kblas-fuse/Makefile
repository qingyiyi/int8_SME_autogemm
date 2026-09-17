# ============================================
# INT8 GEMM SME + inverse-scaling epilogue
# ============================================

# ---- 编译器配置 ----
GCC_DIR    := /data1/env/HPCKit_26.0.RC1/HPCKit/latest/compiler/bisheng
CC         := $(GCC_DIR)/bin/BiSheng-clang
AR         := $(GCC_DIR)/bin/ar
CXX        := $(GCC_DIR)/bin/BiSheng-clang++

# ---- 编译选项 ----
CFLAGS     := 	-O2 -fPIC -Wall \
				-mllvm=-disable-lsr \
				-mllvm=-aarch64-insert-extract-base-cost=3 \
				-mllvm=-disable-aggressive-fabs-opt=true \
				-mllvm=-disable-folding-shifts-loads=true \
				-mllvm=-disable-small-spreads-precision-increase=true \
				-mllvm=-ignore-tti-inline-compatible=true

# ARMv9 + SME
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
INCLUDES   := -I. -I./assemble -I./include

# 汇总
ALL_CFLAGS := $(CFLAGS) $(ARCH_FLAGS) $(DEFINES) $(OPENMP) $(INCLUDES)

# ---- 源文件 ----
# The production NN symbol includes the inverse-scaling epilogue.  There is
# intentionally only one driver/kernel path in the final build.
BASE_CPP_SRCS := int8_gemm.cpp
BASE_ASM_SRCS := assemble/gemm_sme_nn.S \
                 assemble/gemm_tcopy_zip.S \
                 assemble/gemm_ncopy_unzip.S

# ---- 目标文件 ----
BASE_CPP_OBJS := $(BASE_CPP_SRCS:%.cpp=%.o)
BASE_ASM_OBJS := $(BASE_ASM_SRCS:assemble/%.S=assemble/%.o)
BASE_OBJS     := $(BASE_CPP_OBJS) $(BASE_ASM_OBJS)

# ---- 输出与测试 ----
TARGET      := libint8gemm.so
TEST_SRC    := test_int8gemm_kblas.cpp
TEST_OBJ    := $(TEST_SRC:%.cpp=%.o)
TEST_TARGET := test_kblas_gemm
PERF_SRC    := test_fused_performance.cpp
PERF_OBJ    := $(PERF_SRC:%.cpp=%.o)
PERF_TARGET := test_fused_gemm

# Local runtime libraries used only by the HBM performance test.  -L chooses
# this project's lib/libhwloc.so at link time; the relative rpath lets the
# executable find the same library at runtime without LD_LIBRARY_PATH.
PROJECT_DIR     := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_LIB_DIR := $(PROJECT_DIR)/lib
PERF_LDFLAGS ?= -L$(PROJECT_LIB_DIR) -Wl,-rpath,'$$ORIGIN/lib'

# test_fused_performance.cpp calls hbw_verify_memory_region(), whose public
# hbwmalloc API is provided by libmemkind on the target image.  Keep this
# overrideable in case the target installs it under a non-default path/name.
PERF_LDLIBS ?= -lnuma -lhwloc -lmemkind

# Portable reference checks never use target-only ARCH_FLAGS or target
# dependencies.  They compile the production C++ driver against strict mocks.
HOST_CXX              ?= g++
HOST_CXXFLAGS         ?= -O2 -Wall -Wextra -Werror
HOST_TEST             := tests/test_reference
PRODUCTION_DRIVER_TEST := tests/test_production_driver_host
HOST_DRIVER_HEADERS   := tests/reference.hpp int8_gemm.hpp \
                        tests/host_stubs/hbwmalloc.h tests/host_stubs/numa.h \
                        tests/host_stubs/numaif.h
TEST_HEADERS          := tests/reference.hpp tests/guarded_matrix.hpp
REFERENCE_FP_FLAGS   := -fno-fast-math -frounding-math -ffp-contract=off
# The only target acceptance shape is the fixed production contract.
TEST_ARGS ?= --m 8192 --n 8192 --threads 32 --modes all --repeat 1 --seed 42

# ============================================
# 编译规则
# ============================================

.PHONY: all clean check test test-host test-driver-host perf verify verify-perf

all: $(TARGET)

# Link the single production API/kernel path.
$(TARGET): $(BASE_OBJS)
	$(CC) -shared $(OPENMP) -o $@ $^ -lm -lpthread -lnuma
	@echo ">>> 生成成功: $@"
	@echo ">>> 导出符号:"
	@nm -D $@ | grep -E "cblas_gemm_s8s8s8|int8_sme_gemm" || echo "警告: 未找到目标符号!"

# Target executable: build only; `verify` runs it.
test: $(TEST_TARGET)

# HBM performance executable; unlike test_kblas_gemm this target emphasizes
# timing/efficiency and follows the original int8/test_unigemm methodology.
perf: $(PERF_TARGET)

$(TEST_TARGET): $(TEST_OBJ) $(BASE_OBJS)
	$(CXX) $(OPENMP) -o $@ $^ -lm -lpthread
	@echo ">>> 生成生产测试程序: $@"

$(PERF_TARGET): $(PERF_OBJ) $(BASE_OBJS)
	$(CXX) $(OPENMP) -o $@ $^ -lm -lpthread $(PERF_LDFLAGS) $(PERF_LDLIBS)
	@echo ">>> 生成 fused 性能测试程序: $@"

verify: $(TEST_TARGET)
	./$(TEST_TARGET) $(TEST_ARGS)

verify-perf: $(PERF_TARGET)
	./$(PERF_TARGET) 8192 8192 2048 32 all 10 3

test-host: $(HOST_TEST) $(PRODUCTION_DRIVER_TEST)
	./$(HOST_TEST)
	./$(PRODUCTION_DRIVER_TEST)

# The host driver test compiles int8_gemm.cpp with ordinary g++ and fake empty
# NUMA/HBM headers.  It verifies production ABI wiring only; no SME assembly is
# executed on the host.
test-driver-host: $(PRODUCTION_DRIVER_TEST)
	./$(PRODUCTION_DRIVER_TEST)

$(HOST_TEST): tests/test_reference.cpp $(TEST_HEADERS)
	$(HOST_CXX) $(HOST_CXXFLAGS) -std=c++17 $(REFERENCE_FP_FLAGS) $< -o $@ -lm

$(PRODUCTION_DRIVER_TEST): tests/test_production_driver_host.cpp int8_gemm.cpp $(HOST_DRIVER_HEADERS)
	$(HOST_CXX) $(HOST_CXXFLAGS) -Wno-error -std=c++17 -fopenmp -DCOMPSIZE=1 $(REFERENCE_FP_FLAGS) \
		-Itests/host_stubs -I. tests/test_production_driver_host.cpp int8_gemm.cpp -o $@ -lm

# Keep the test oracle strict even if production flags change later.
$(TEST_OBJ): $(TEST_SRC) $(TEST_HEADERS) int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 $(REFERENCE_FP_FLAGS) -c $< -o $@

$(PERF_OBJ): $(PERF_SRC) int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 -c $< -o $@

# C++ implementation.
int8_gemm.o: int8_gemm.cpp int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 -c $< -o $@

# ---- 每个 .S 文件单独定义 ASMNAME ----
# Included SME sources are listed explicitly so a change in an included file
# cannot silently leave a stale object on the target machine.
SME_ASM_DEPS := assemble/gemm_sme_base.S \
                assemble/gemm_sme_packing.S \
                assemble/gemm_sme_nt.S \
                assemble/gemm_sme_reg_defs.h \
                assemble/int8_gemm_common.S
NN_ASM_DEPS  := $(SME_ASM_DEPS) assemble/gemm_sme_inverse_scaling.S

# A矩阵pack -> int8_sme_gemm_itcopy
assemble/gemm_tcopy_zip.o: assemble/gemm_tcopy_zip.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_itcopy -DASMFNAME=int8_sme_gemm_itcopy_ -DNAME=int8_sme_gemm_itcopy_ -DCNAME=int8_sme_gemm_itcopy -c $< -o $@

# B矩阵pack -> int8_sme_gemm_oncopy
assemble/gemm_ncopy_unzip.o: assemble/gemm_ncopy_unzip.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_oncopy -DASMFNAME=int8_sme_gemm_oncopy_ -DNAME=int8_sme_gemm_oncopy_ -DCNAME=int8_sme_gemm_oncopy -c $< -o $@

# NN kernel -> int8_sme_gemm_kernel_nn, including the production epilogue.
assemble/gemm_sme_nn.o: assemble/gemm_sme_nn.S $(NN_ASM_DEPS)
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_kernel_nn -DASMFNAME=int8_sme_gemm_kernel_nn_ -DNAME=int8_sme_gemm_kernel_nn_ -DCNAME=int8_sme_gemm_kernel_nn -c $< -o $@

clean:
	rm -f $(BASE_CPP_OBJS) $(BASE_ASM_OBJS) \
	       $(TARGET) $(TEST_OBJ) $(TEST_TARGET) $(PERF_OBJ) $(PERF_TARGET) \
	       $(HOST_TEST) $(PRODUCTION_DRIVER_TEST)
# 快速检查
check: $(TARGET)
	nm -D $(TARGET) | grep -E 'cblas_gemm_s8s8s8($$|_)'
