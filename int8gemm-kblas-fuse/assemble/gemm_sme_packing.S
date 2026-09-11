/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: gemm implementations
 * Author: KPL
 * Create: 2022-10-09
 * Notes: NA
 */
#define ASSEMBLER
#include "common.h"

#define EPILOGUE

	//                            *********B*********
	//                            q8   q9   q10   q11
	//                            q12  q13  q14   q15

	//  *********A*********       *********C*********
	//  q0   q4    q0   q4        q16  q20   q24  q28
	//  q1   q5    q1   q5        q17  q21   q25  q29
	//  q2   q6    q2   q6        q18  q22   q26  q30
	//  q3   q7    q3   q7        q19  q23   q27  q31

	//BLASLONG REALNAME(BLASLONG m, BLASLONG n, BLASLONG k,   // X0, X1, X2
	//  double* A, BLASLONG lda,                              // X3, X4
	//  double* B,BLASLONG ldb,                               // X5, X6
	//  double* C,BLASLONG ldc                                // X7, X8
	//FLOAT sgemm_kernel(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT *a, BLASLONG lda, const FLOAT alpha0,
	// FLOAT *b, BLASLONG ldb, const FLOAT beta0, FLOAT *c, BLASLONG ldc);
	// SIMD float passing to vn registers
	// alpha0  s0
	// beta0   s1
#ifndef KERNEL_PARAMS
#define KERNEL_PARAMS
#define origM       x0
#define origN       x1
#define origK       x2
#define origPA      x3
#define origPB      x4
#define pC          x5
#define LDC         x6
#endif
#include "./gemm_sme_reg_defs.h"
#ifndef OFFSET_A
#define OFFSET_A    MIN_M
#endif
#ifndef OFFSET_B
#define OFFSET_B    MIN_N
#endif

#ifndef B_PRFM1_
.macro B_PRFM1
PRFM   PLDL2KEEP, [ptrbp]
add    ptrbp,ptrbp,#64
.endm
#endif

#ifndef B_PRFM2_
.macro B_PRFM2
PRFM   PLDL2KEEP, [ptrbp]
add    ptrbp,ptrbp,#64
.endm
#endif

.macro C_PRFM1
PRFM   PLDL1STRM, [ptrcp, offset_pfc]
add	offset_pfc, offset_pfc, #64
.endm

.macro C_PRFM2

.endm



#ifndef KERNEL_LOAD
.macro LOAD_A0_INT8 a0, Pg0, ldopt=ld1w,ldaopt=ld1w
	\ldaopt     \a0\().s,  \Pg0\()/z, [pA0]
.endm

.macro LOAD_A0A1_INT8 a0, a1, Pg0, Pg1 ldopt=ld1w, ldaopt=ld1w
	\ldaopt     \a0\().s,  \Pg0\()/z, [pA0]
	\ldaopt     \a1\().s,  \Pg1\()/z, [pA0, #1, MUL VL]
.endm

.macro LOAD_B0_INT8 b0, Pg0, ldopt=ld1w, ldaopt=ld1w
	\ldaopt     \b0\().s,  \Pg0\()/z, [pB0]
.endm

.macro LOAD_B0B1_INT8 b0, b1, Pg0, Pg1, ldopt=ld1w, ldaopt=ld1w
	\ldaopt     \b0\().s,  \Pg0\()/z, [pB0]
	\ldaopt     \b1\().s,  \Pg1\()/z, [pB0, #1, MUL VL]
.endm

.macro LOAD_OFFSET_A_B

.endm

.macro SET_SVINDEX

.endm

#define KERNEL_LOAD
#endif

.macro KERNEL_1VLX1VL a0,a1,b0,b1,ldopt=LD1,ldaopt=LD1
    LOAD_A0_INT8 \a0, p1
    LOAD_B0_INT8 \b0, p0
    COMPUTE_MOPA_TYPED	za0.s, Pg_MOPA_1/m, Pg_MOPA_0/m, \a0\().PRECISION, \b0\().PRECISION
	add pB0, pB0, OFFSET_B, LSL #PRECI_WIDTH
	add pA0, pA0, OFFSET_A, LSL #PRECI_WIDTH
.endm
.macro KERNEL_2VLX1VL a0,a1,b0,b1,ldopt=LD1,ldaopt=LD1
    LOAD_A0A1_INT8 \a0, \a1, p1, p2
    LOAD_B0_INT8 \b0, p0
    COMPUTE_MOPA_TYPED	za0.s, Pg_MOPA_1/m, Pg_MOPA_0/m, \a0\().PRECISION, \b0\().PRECISION
    COMPUTE_MOPA_TYPED	za2.s, Pg_MOPA_2/m, Pg_MOPA_0/m, \a1\().PRECISION, \b0\().PRECISION
	add pB0, pB0, OFFSET_B, LSL #PRECI_WIDTH
	add pA0, pA0, OFFSET_A, LSL #PRECI_WIDTH
.endm

.macro KERNEL_1VLX2VL a0,a1,b0,b1,ldopt=LD1,ldaopt=LD1
    LOAD_A0_INT8 \a0, p1
    LOAD_B0B1_INT8 \b0, \b1, p0, p3
    COMPUTE_MOPA_TYPED	za0.s, Pg_MOPA_1/m, Pg_MOPA_0/m, \a0\().PRECISION, \b0\().PRECISION
    COMPUTE_MOPA_TYPED	za1.s, Pg_MOPA_1/m, Pg_MOPA_3/m, \a0\().PRECISION, \b1\().PRECISION
	add pB0, pB0, OFFSET_B, LSL #PRECI_WIDTH
	add pA0, pA0, OFFSET_A, LSL #PRECI_WIDTH
.endm
.macro KERNEL_2VLX2VL a0,a1,b0,b1,ldopt=LD1,ldaopt=LD1
    LOAD_A0A1_INT8 \a0, \a1, p1, p2, \ldopt
    LOAD_B0B1_INT8 \b0, \b1, p0, p3, \ldopt
    COMPUTE_MOPA_TYPED	za0.s, Pg_MOPA_1/m, Pg_MOPA_0/m, \a0\().PRECISION, \b0\().PRECISION
    COMPUTE_MOPA_TYPED	za1.s, Pg_MOPA_1/m, Pg_MOPA_3/m, \a0\().PRECISION, \b1\().PRECISION
    COMPUTE_MOPA_TYPED	za2.s, Pg_MOPA_2/m, Pg_MOPA_0/m, \a1\().PRECISION, \b0\().PRECISION
    COMPUTE_MOPA_TYPED	za3.s, Pg_MOPA_2/m, Pg_MOPA_3/m, \a1\().PRECISION, \b1\().PRECISION
    add pB0, pB0, OFFSET_B, LSL #PRECI_WIDTH
    add pA0, pA0, OFFSET_A, LSL #PRECI_WIDTH
.endm

.macro KERNEL_M0 mvl,nvl
	KERNEL_\mvl\()X\nvl z0,z1,z8,z9
.endm

.macro KERNEL_M1 mvl,nvl
	KERNEL_\mvl\()X\nvl z2,z3,z10,z11
.endm

.macro KERNEL_M2 mvl,nvl
    KERNEL_\mvl\()X\nvl z4,z5,z12,z13
.endm

.macro KERNEL_M3 mvl,nvl
    KERNEL_\mvl\()X\nvl z6,z7,z14,z15
.endm

// 新增M4/5/6/7宏定义
.macro KERNEL_M4 mvl,nvl
    KERNEL_\mvl\()X\nvl z16,z17,z24,z25
.endm
.macro KERNEL_M5 mvl,nvl
    KERNEL_\mvl\()X\nvl z18,z19,z26,z27
.endm
.macro KERNEL_M6 mvl,nvl
    KERNEL_\mvl\()X\nvl z20,z21,z28,z29
.endm
.macro KERNEL_M7 mvl,nvl
    KERNEL_\mvl\()X\nvl z22,z23,z30,z31
.endm
// 新增M4/5/6/7宏定义

.macro KERNEL_LDNT1D_M0 mvl,nvl
	KERNEL_\mvl\()X\nvl z0,z1,z8,z9,LDNT1
.endm

.macro KERNEL_LDNT1D_M1 mvl,nvl
    KERNEL_\mvl\()X\nvl z2,z3,z10,z11,LDNT1
.endm

.macro KERNEL_LDNT1D_M2 mvl,nvl
    KERNEL_\mvl\()X\nvl z4,z5,z12,z13,LDNT1
.endm

.macro KERNEL_LDNT1D_M3 mvl,nvl
    KERNEL_\mvl\()X\nvl z6,z7,z14,z15,LDNT1
.endm

// 新增M4/5/6/7宏定义
.macro KERNEL_LDNT1D_M4 mvl,nvl
    KERNEL_\mvl\()X\nvl z16,z17,z24,z25,LDNT1
.endm
.macro KERNEL_LDNT1D_M5 mvl,nvl
    KERNEL_\mvl\()X\nvl z18,z19,z26,z27,LDNT1
.endm
.macro KERNEL_LDNT1D_M6 mvl,nvl
    KERNEL_\mvl\()X\nvl z20,z21,z28,z29,LDNT1
.endm
.macro KERNEL_LDNT1D_M7 mvl,nvl
    KERNEL_\mvl\()X\nvl z22,z23,z30,z31,LDNT1
.endm
// 新增M4/5/6/7宏定义

.macro KERNEL_LDNT1DAB_M0 mvl,nvl
	KERNEL_\mvl\()X\nvl z0,z1,z8,z9,LDNT1
.endm

.macro KERNEL_LDNT1DAB_M1 mvl,nvl
    KERNEL_\mvl\()X\nvl z2,z3,z10,z11,LDNT1
.endm

.macro KERNEL_LDNT1DAB_M2 mvl,nvl
    KERNEL_\mvl\()X\nvl z4,z5,z12,z13,LDNT1
.endm

.macro KERNEL_LDNT1DAB_M3 mvl,nvl
    KERNEL_\mvl\()X\nvl z6,z7,z14,z15,LDNT1
.endm
// 有预取C
.macro KERNEL_PRFMBC mvl,nvl
    KERNEL_M0 \mvl \nvl
#if !defined(SMALL_SME_GEMM) && !defined(SUP_SME_GEMM)
	PRFM   PLDL1KEEP, [ptrbp]
	PRFM   PLDL1KEEP, [ptrbp,#64]
#endif
	KERNEL_M1 \mvl \nvl
	add  ptrbp, ptrbp, OFFSET_B, LSL #PRECI_WIDTH
	KERNEL_M2 \mvl \nvl
	KERNEL_M3 \mvl \nvl
	//! 测试，将后四次计算直接改成M4/M5/M6/M7
	KERNEL_M4 \mvl \nvl
	KERNEL_M5 \mvl \nvl
#if !defined(SMALL_SME_GEMM) && !defined(SUP_SME_GEMM)
	// PRFM PLDL2KEEP, [ptrcp]  // P=128, Q=512
    add offset_pfc, offset_pfc, #128   // P=128, Q=512
	// PRFM PLDL2KEEP, [ptrcp, #64]
	add ptrcp, ptrcp, LDC
	// PRFM PLDL2KEEP, [ptrcp]
	// PRFM PLDL2KEEP, [ptrcp, #64]
#endif
	KERNEL_M6 \mvl \nvl
	KERNEL_M7 \mvl \nvl
.endm

// 不预取C
.macro KERNEL_3m_PRFMB mvl,nvl
    KERNEL_M0 \mvl \nvl
#if !defined(SMALL_SME_GEMM) && !defined(SUP_SME_GEMM)
	#PRFM   PLDL2KEEP, [ptrbp]
	#PRFM   PLDL2KEEP, [ptrbp,#64]
	#add  ptrbp, ptrbp, OFFSET_B, LSL #PRECI_WIDTH
#endif
	KERNEL_M1 \mvl \nvl

	KERNEL_M2 \mvl \nvl
	KERNEL_M3 \mvl \nvl
	#PRFM   PLDL2KEEP, [ptrbp]
	#PRFM   PLDL2KEEP, [ptrbp,#64]
	#add  ptrbp, ptrbp, OFFSET_B, LSL #PRECI_WIDTH
		
//! 测试，将后四次计算直接改成M4/M5/M6/M7
	KERNEL_M0 \mvl \nvl
	KERNEL_M1 \mvl \nvl
	
	KERNEL_M2 \mvl \nvl
	KERNEL_M3 \mvl \nvl

	
	PRFM   PLDL2KEEP, [ptrbp]
	PRFM   PLDL2KEEP, [ptrbp,#64]
	add  ptrbp, ptrbp, OFFSET_B, LSL #PRECI_WIDTH
		
.endm



.macro KERNEL_LDNTB_PRFMBC mvl,nvl
    KERNEL_LDNT1D_M0 \mvl \nvl
	KERNEL_LDNT1D_M1 \mvl \nvl
#if !defined(SMALL_SME_GEMM) && !defined(SUP_SME_GEMM)
	PRFM   PLDL1KEEP, [ptrbp]
	PRFM   PLDL1KEEP, [ptrbp,#64]
#endif

	KERNEL_LDNT1D_M2 \mvl \nvl

	add  ptrbp, ptrbp, OFFSET_B, LSL #PRECI_WIDTH

	KERNEL_LDNT1D_M3 \mvl \nvl
//! 测试，将后四次计算直接改成M4/M5/M6/M7
	KERNEL_LDNT1D_M4 \mvl \nvl
	KERNEL_LDNT1D_M5 \mvl \nvl
	KERNEL_LDNT1D_M6 \mvl \nvl
	
#if !defined(SMALL_SME_GEMM) && !defined(SUP_SME_GEMM)
	PRFM PLDL1KEEP, [ptrcp]  // P=128, Q=512
    add offset_pfc, offset_pfc, #128   // P=128, Q=512
	PRFM PLDL1KEEP, [ptrcp, #64]
	add ptrcp, ptrcp, LDC
	PRFM PLDL1KEEP, [ptrcp]
	PRFM PLDL1KEEP, [ptrcp, #64]
#endif

	KERNEL_LDNT1D_M7 \mvl \nvl
.endm

.macro KERNEL_LDNTB_PRFMB mvl,nvl
    KERNEL_LDNT1D_M0 \mvl \nvl
	KERNEL_LDNT1D_M1 \mvl \nvl
#if !defined(SMALL_SME_GEMM) && !defined(SUP_SME_GEMM)
	PRFM   PLDL2KEEP, [ptrbp]
	PRFM   PLDL2KEEP, [ptrbp,#64]
#endif
	KERNEL_LDNT1D_M2 \mvl \nvl

	add  ptrbp, ptrbp, OFFSET_B, LSL #PRECI_WIDTH
	
	KERNEL_LDNT1D_M3 \mvl \nvl
//! 测试，将后四次计算直接改成M4/M5/M6/M7
	KERNEL_LDNT1D_M0 \mvl \nvl
	KERNEL_LDNT1D_M1 \mvl \nvl
	KERNEL_LDNT1D_M2 \mvl \nvl
	KERNEL_LDNT1D_M3 \mvl \nvl
#if !defined(SMALL_SME_GEMM) && !defined(SUP_SME_GEMM)
	#PRFM   PLDL2KEEP, [ptrbp]
	#PRFM   PLDL2KEEP, [ptrbp,#64]
#endif
	#add  ptrbp, ptrbp, OFFSET_B, LSL #PRECI_WIDTH
.endm



#ifndef KERNEL_GEMM_START_
.macro KERNEL_GEMM_START
#ifndef PACKING

	ldr     LDC, [sp, #(11*16)]

    // 寄存器不足，借用栈空间存放(origK % 4)的值
    and     TMP_CNT, origK, #3
    str     TMP_CNT_SIN, [sp, #(11*16)]

#endif
	mov pBt,   origPB
	mov counterJ, #0

	PTRUE p0.h, ALL
	SET_SVINDEX

	fmov alpha, s0
	dup alpha0,alpha
	fmov beta, s1
	dup	beta0, beta
	
	// 设置分块大小初始值
	GET_ACTIVE_NUM_S p0, ELE_SIZE
	GET_ACTIVE_NUM_H p0, MIN_N
	GET_ACTIVE_NUM_H p0, MIN_M
	lsl LDC,LDC,#2
	KERNEL_MM_LOOP_N h
.endm
#endif
#include "./gemm_sme_base.S"
#ifdef FUSED_INVERSE_SCALING
#include "./gemm_sme_fused_store.S"
#endif
	/*******************************************************************************
	 * Macro definitions
	 *******************************************************************************/
	// function begin
	PROLOGUE
	// store register into stack
	SAVE_REGS

    START_SME_FEATURE

.SDGEMM_BEGIN:
	KERNEL_GEMM_START

.END_OF_GEMM:
#ifdef FUSED_INVERSE_SCALING
    STOP_SME_FEATURE
    FUSED_INVERSE_SCALING_POSTSTORE
#else
    STOP_SME_FEATURE
#endif
	RESTORE_REGS
	mov w0, wzr
	ret
	EPILOGUE