# ============================================================================
# int8 GEMM 工程 Makefile（自编译 GCC 12 + 独立 glibc 2.38）
#
# 用法:
#   make            # 编译 Release 版本（默认）
#   make debug      # 编译 Debug 版本
#   make all        # 同时编译 Release 和 Debug
#   make clean      # 清理所有编译产物
#   make help       # 显示帮助
#
# 注意: 本文件不修改 PATH / LD_LIBRARY_PATH 等环境变量，
#       新工具链全部通过变量显式指定，与系统环境完全隔离。
# ============================================================================

# ------------------------- 工具链与 glibc 路径 -------------------------
GCC_DIR    := /data1/lwx1377379/gcc-12.5.0-install
GLIBC_DIR  := /data1/lwx1377379/glibc-2.38-install
# x86_64 用 ld-linux-x86-64.so.2，ARM64 用 ld-linux-aarch64.so.1
DYN_LINKER := $(GLIBC_DIR)/lib/ld-linux-aarch64.so.1

CC  := $(GCC_DIR)/bin/gcc
CXX := $(GCC_DIR)/bin/g++
AR  := ar

# ------------------------- 项目配置 -------------------------
# TARGET  := int8_gemm_test
TARGET := test_unigemm
# SRCDIRS := ./src
SRCDIRS := .
INCDIRS := ./include
OBJDIR  := obj
BINDIR  := bin

# ------------------------- 编译选项 -------------------------
# 公共选项：C 和 C++ 都使用新 glibc 的头文件
COMMON_CPPFLAGS := $(addprefix -I,$(INCDIRS)) -I$(GLIBC_DIR)/include
COMMON_CFLAGS   := -std=c2x
COMMON_CXXFLAGS := -std=c++20

# 公共链接选项：新 glibc 的动态加载器 + rpath + 第三方库路径
#
# 注意：独立 glibc 的加载器不读系统 /etc/ld.so.cache，
#       所有运行时需要的库目录都必须写进 rpath，不能只写 -L。
#       $ORIGIN 指可执行文件所在目录（bin/release 或 bin/debug），
#       $$ORIGIN/../../lib 即工程根目录下的 ./lib。
PROJECT_ROOT := $(CURDIR)
COMMON_LDFLAGS := \
	-L$(GLIBC_DIR)/lib \
	-Wl,--dynamic-linker=$(DYN_LINKER) \
	-Wl,--rpath=$(GLIBC_DIR)/lib \
	-L/data1/cxz/int8/lib \
	-Wl,--rpath=/data1/cxz/int8/lib \
	-L/opt/HPCKit/latest/kml/gcc/lib/sme/kblas/multi \
	-Wl,--rpath=/opt/HPCKit/latest/kml/gcc/lib/sme/kblas/multi \
	-L/data1/env/HPCKit_26.0.RC1/HPCKit/latest/compiler/gcc/lib64 \
	-Wl,--rpath=/data1/env/HPCKit_26.0.RC1/HPCKit/latest/compiler/gcc/lib64 \
	-L$(GCC_DIR)/lib64 \
	-Wl,--rpath=$(GCC_DIR)/lib64 \
	-L$(PROJECT_ROOT)/lib \
	-Wl,--rpath=$(PROJECT_ROOT)/lib \
	-Wl,--rpath=/usr/lib64 \
	-fopenmp

COMMON_LDLIBS := -lm -lnuma -lhwloc -lmemkind -lunigemm_sme -ldl

# Release / Debug 各自的增量选项
RELEASE_CFLAGS   := -O3 -DNDEBUG
RELEASE_CXXFLAGS := -O3 -DNDEBUG
# RELEASE_LDLIBS   := -lkblas -lunigemm_sme -lgomp
RELEASE_LDLIBS   := -lint8gemm -lunigemm_sme -lgomp

DEBUG_CFLAGS     := -O0 -g -Wall -Wextra -DDEBUG
DEBUG_CXXFLAGS   := -O0 -g -Wall -Wextra -DDEBUG
DEBUG_LDLIBS     :=

# 依赖文件（头文件变化自动重编）
DEPFLAGS := -MMD -MP

# ------------------------- 构建模式（MODE=release 或 debug）-------------------------
MODE ?= release

ifeq ($(MODE),debug)
  MODE_CFLAGS   := $(DEBUG_CFLAGS)
  MODE_CXXFLAGS := $(DEBUG_CXXFLAGS)
  MODE_LDLIBS   := $(DEBUG_LDLIBS)
else
  MODE_CFLAGS   := $(RELEASE_CFLAGS)
  MODE_CXXFLAGS := $(RELEASE_CXXFLAGS)
  MODE_LDLIBS   := $(RELEASE_LDLIBS)
endif

CFLAGS   := $(COMMON_CFLAGS)   $(MODE_CFLAGS)
CXXFLAGS := $(COMMON_CXXFLAGS) $(MODE_CXXFLAGS)
LDFLAGS  := $(COMMON_LDFLAGS)
LDLIBS   := $(COMMON_LDLIBS)   $(MODE_LDLIBS)

# TARGET_PATH := $(BINDIR)/$(MODE)/$(TARGET)
TARGET_PATH := /data1/cxz/int8/$(TARGET)

# ------------------------- 源文件与对象文件 -------------------------
SOURCES := $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.c $(dir)/*.cpp $(dir)/*.cc $(dir)/*.cxx))
OBJECTS := $(patsubst $(SRCDIRS)/%.c,$(OBJDIR)/$(MODE)/%.o, \
           $(patsubst $(SRCDIRS)/%.cpp,$(OBJDIR)/$(MODE)/%.o, \
           $(patsubst $(SRCDIRS)/%.cc,$(OBJDIR)/$(MODE)/%.o, \
           $(patsubst $(SRCDIRS)/%.cxx,$(OBJDIR)/$(MODE)/%.o,$(SOURCES)))))
DEPS    := $(OBJECTS:.o=.d)

# ------------------------- 目标规则 -------------------------
.PHONY: release debug all clean help

# 默认目标：Release
release:
	@$(MAKE) --no-print-directory MODE=release $(TARGET_PATH)
debug:
	@$(MAKE) --no-print-directory MODE=debug $(TARGET_PATH)

all: release debug

help:
	@echo "可用目标:"
	@echo "  make / make release   编译 Release 版本 -> $(BINDIR)/release/$(TARGET)"
	@echo "  make debug            编译 Debug 版本   -> $(BINDIR)/debug/$(TARGET)"
	@echo "  make all              同时编译两个版本"
	@echo "  make clean            清理所有编译产物"

# 链接
$(TARGET_PATH): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
	@echo "==> [$(MODE)] 链接完成: $@"

# 编译规则（release/debug 共用，通过 $(MODE) 区分输出目录）
$(OBJDIR)/$(MODE)/%.o: $(SRCDIRS)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(COMMON_CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)/$(MODE)/%.o: $(SRCDIRS)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)/$(MODE)/%.o: $(SRCDIRS)/%.cc
	@mkdir -p $(dir $@)
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)/$(MODE)/%.o: $(SRCDIRS)/%.cxx
	@mkdir -p $(dir $@)
	$(CXX) $(COMMON_CPPFLAGS) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

# 头文件依赖自动包含
-include $(DEPS)

clean:
	rm -rf $(OBJDIR) $(BINDIR)
	@echo "清理完成。"