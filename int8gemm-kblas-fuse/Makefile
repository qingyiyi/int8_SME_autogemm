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
CPP_SRCS     := int8_gemm.cpp

# 需要独立编译的汇编文件（.S 大写 = 先走 C 预处理器）
# gemm_sme_base.S / int8_gemm_common.S / gemm_sme_reg_defs.h 只被 #include，不列在这里
ASM_SRCS := assemble/gemm_sme_nt.S \
            assemble/gemm_sme_nn.S \
            assemble/gemm_tcopy_zip.S \
            assemble/gemm_ncopy_unzip.S

# ---- 目标文件 ----
CPP_OBJS     := $(CPP_SRCS:%.cpp=%.o)
ASM_OBJS   := $(ASM_SRCS:assemble/%.S=assemble/%.o)
ALL_OBJS   := $(CPP_OBJS) $(ASM_OBJS)

# 输出
TARGET     := libint8gemm.so
TEST_SRC   := test_int8gemm_kblas.cpp
TEST_OBJ   := $(TEST_SRC:%.cpp=%.o)
TEST_TARGET:= test_kblas_gemm

# Portable reference checks never use target-only ARCH_FLAGS or target dependencies.
HOST_CXX   ?= g++
HOST_CXXFLAGS ?= -O2 -Wall -Wextra -Werror
HOST_TEST  := tests/test_reference
TEST_HEADERS := tests/reference.hpp tests/guarded_matrix.hpp
REFERENCE_FP_FLAGS := -fno-fast-math -frounding-math -ffp-contract=off
TEST_ARGS  ?= --suite smoke

# ============================================
# 编译规则
# ============================================

.PHONY: all clean check test test-host verify

all: $(TARGET)

# 链接动态库
$(TARGET): $(ALL_OBJS)
	$(CC) -shared $(OPENMP) -o $@ $^ -lm -lpthread -lnuma
	@echo ">>> 生成成功: $@"
	@echo ">>> 导出符号:"
	@nm -D $@ | grep -E "cblas_gemm_s8s8s32|int8_sme_gemm" || echo "警告: 未找到目标符号!"

# 测试可执行文件（直接链接 .o，不走 dlopen）
test: $(TEST_TARGET)

$(TEST_TARGET): $(TEST_OBJ) $(ALL_OBJS)
	$(CXX) $(OPENMP) -o $@ $^ -lm -lpthread
	@echo ">>> 生成测试程序: $@"

# Run the target regression explicitly; `test` remains build-only.
verify: $(TEST_TARGET)
	./$(TEST_TARGET) $(TEST_ARGS)

test-host: $(HOST_TEST)
	./$(HOST_TEST)

$(HOST_TEST): tests/test_reference.cpp $(TEST_HEADERS)
	$(HOST_CXX) $(HOST_CXXFLAGS) -std=c++17 $(REFERENCE_FP_FLAGS) $< -o $@ -lm

# Keep the test oracle strict even if production flags change later.
$(TEST_OBJ): $(TEST_SRC) $(TEST_HEADERS) int8_gemm.hpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 $(REFERENCE_FP_FLAGS) -c $< -o $@

# C++ 文件编译
# ---- 通用 C 规则不变 ----
%.o: %.cpp
	$(CXX) $(CFLAGS) $(ARCH_FLAGS) $(DEFINES) $(OPENMP) $(INCLUDES) -std=c++17 -c $< -o $@

# C++ 测试文件编译
%.o: %.cpp
	$(CXX) $(ALL_CFLAGS) -std=c++17 -c $< -o $@
	
# ---- 每个 .S 文件单独定义 ASMNAME ----

# A矩阵pack -> int8_sme_gemm_incopy
assemble/gemm_tcopy_zip.o: assemble/gemm_tcopy_zip.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_itcopy -DASMFNAME=int8_sme_gemm_itcopy_ -DNAME=int8_sme_gemm_itcopy_ -DCNAME=int8_sme_gemm_itcopy -c $< -o $@

# B矩阵pack -> int8_sme_gemm_oncopy
assemble/gemm_ncopy_unzip.o: assemble/gemm_ncopy_unzip.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_oncopy -DASMFNAME=int8_sme_gemm_oncopy_ -DNAME=int8_sme_gemm_oncopy_ -DCNAME=int8_sme_gemm_oncopy -c $< -o $@

# NN kernel -> int8_sme_gemm_kernel_nn
assemble/gemm_sme_nn.o: assemble/gemm_sme_nn.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_kernel_nn -DASMFNAME=int8_sme_gemm_kernel_nn_ -DNAME=int8_sme_gemm_kernel_nn_ -DCNAME=int8_sme_gemm_kernel_nn -c $< -o $@

# NT kernel -> int8_sme_gemm_kernel_nt
assemble/gemm_sme_nt.o: assemble/gemm_sme_nt.S
	$(CC) $(ALL_CFLAGS) -DASMNAME=int8_sme_gemm_kernel_nt -DASMFNAME=int8_sme_gemm_kernel_nt_ -DNAME=int8_sme_gemm_kernel_nt_ -DCNAME=int8_sme_gemm_kernel_nt -c $< -o $@

# 汇编文件编译
assemble/%.o: assemble/%.S
	$(CC) $(ALL_CFLAGS) -c $< -o $@

clean:
	rm -f $(CPP_OBJS) $(ASM_OBJS) $(TARGET) $(TEST_OBJ) $(TEST_TARGET) $(HOST_TEST)

# 快速检查
check: $(TARGET)
	nm -D $(TARGET) | grep cblas_gemm_s8s8s32