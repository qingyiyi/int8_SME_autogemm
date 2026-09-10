/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 * Description: gemm nt implementations
 * Author: KPL
 * Create: 2022-10-09
 * Notes: NA
 */
#include "gemm_sme_reg_defs.h"
#include "int8_gemm_common.S"
#define OFFSET_A    MIN_M
#define OFFSET_B    MIN_N

#define B_PRFM1_
.macro B_PRFM1
PRFM   PLDL2KEEP, [ptrbp]
.endm

#define B_PRFM2_
.macro B_PRFM2
PRFM   PLDL2KEEP, [ptrbp, #64]
add    ptrbp,ptrbp,128
.endm

#define KERNEL_MM_LOOP_N_PRE_FUNC_
.macro KERNEL_MM_LOOP_N_PRE_FUNC
	mov pAt, origPA
	mov pB0, pBt
    add ptrbp, pBt, #49152  // 49152通过实际测试得到
.endm

#include "gemm_sme_packing.S"