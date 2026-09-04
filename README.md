# Kunpeng 1636 GEMM Runtime Wrappers

面向鲲鹏 1636 ARMv9 平台的 GEMM 运行时封装与 SME 内核实验代码。项目将
`kblas` GEMM 内核、SDMA 数据搬运和线程同步接口导出给上层运行时使用，目标是让
MLIR/运行时调用方能够接入 SVE2 和 SME 加速路径。

> 该仓库不是一个可独立构建和发布的通用 BLAS 库。根目录目前没有构建系统或测试
> 入口；它需要集成到已有的鲲鹏运行时工程中，并由该工程提供 SDK、头文件和链接依赖。

## 内容概览

| 路径 | 说明 |
| --- | --- |
| `KunpengRuntimeWrappers.cpp` | 导出的运行时封装：GEMM、矩阵打包、SDMA 传输、同步屏障和调试辅助接口。 |
| `kblas/gemm_sme_packing.S` | 基于 SME 的 GEMM/打包汇编实现。 |
| `kblas/hgemm_sme_nn.S` | 半精度 NN GEMM 的 SME 汇编实现。 |
| `autoGEMM/` | 上游 autoGEMM 工作树，用作可移植 GEMM 自动调优与内核生成参考。 |
| `架构说明.md` | 鲲鹏 1636 的核心、缓存、HBM、SVE 和 SME 硬件说明。 |

## 支持的运行时能力

- 浮点 GEMM：单精度、双精度、复数和 BF16 的打包与计算封装。
- INT8 GEMM：`s8s8`、`s8u8`、`u8s8`、`u8u8` 到 `s32` 的打包和多分块内核分发。
- SME 内核：面向 ARMv9 SME 的浮点和半精度矩阵乘路径。
- 数据与同步：SDMA 异步拷贝、按核心通道选择、futex 和多级屏障接口。

## 目标环境

运行或集成本项目需要以下条件：

- AArch64 Linux 目标环境，以及具备 SVE2/SME 的鲲鹏 1636 或兼容平台。
- 支持 ARM SME 内建头文件的 C++ 编译器。
- 宿主工程提供的 `kblas` 头文件与库，以及 `mdk_sdma.h`、`sdma_utils.h` 等 SDMA 依赖。
- 调用方自己的构建配置，用于编译 `KunpengRuntimeWrappers.cpp`、汇编内核并完成链接。

可用的运行时符号以 `KunpengRuntimeWrappers.cpp` 中的
`MLIR_KUNPENG_WRAPPERS_EXPORT` 导出定义为准。

## 集成方式

1. 将 `KunpengRuntimeWrappers.cpp` 纳入宿主运行时的 C++ 目标。
2. 按目标平台的 ABI 编译 `kblas/` 下的 SME 汇编文件。
3. 为编译目标提供 `kblas`、MDK SDMA 和 Linux 同步相关头文件与库。
4. 在上层运行时中按矩阵数据类型和转置布局调用对应的 GEMM 或打包封装。

根目录没有足以复现完整构建的 Makefile、CMake 配置或测试命令。因此，首次接入时应
先在宿主工程中验证头文件、链接库、SME 编译选项和目标硬件特性。

## autoGEMM 依赖说明

`autoGEMM/` 当前是一个独立 Git 工作树，根仓库记录了其提交，但没有
`.gitmodules` 映射。新克隆根仓库后，如需该目录，需要手动获取上游仓库并检出根仓库
记录的提交：

```bash
git clone https://github.com/wudu98/autoGEMM.git autoGEMM
git -C autoGEMM checkout ebef4a6f5d5308836f89477804dc6419dd472619
```

autoGEMM 自身的构建与实验说明见 [autoGEMM/README.md](autoGEMM/README.md)。

## 硬件参考

鲲鹏 1636 的 SVE/SME 算力、缓存层次、HBM 配置和 I/O 规格见
[架构说明.md](架构说明.md)。

## 许可证

根仓库尚未提供统一许可证。`autoGEMM` 子目录保留其自身的
[LICENSE](autoGEMM/LICENSE)；在复用或分发本项目其余代码前，应先补充并确认根项目
的许可证策略。
