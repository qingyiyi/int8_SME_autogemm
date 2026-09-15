/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 * Description: gemm sme base implementation
 * Author: KPL
 * Create: 2022-10-22
 * Notes: NA
 */
#define LDNT1 LD1


#undef ZERO
#ifndef SAVE_ZACOL_IMPL_
.macro SAVE_ZACOL pc,off, reg_za,base_idx,idx,nidx,reg_pg,rab0,rc0,t0,t1
	MOVA	\rab0\().s, \reg_pg\()/M, \reg_za\()v.s[\base_idx, \idx]
#ifdef INT8_INVERSE_SCALING_EPILOGUE
	// The production NN path inverse-scales this just-extracted ZA vector and
	// stores C8 directly; it never materializes a C32 output tile.
	INT8_INVERSE_SCALING_DIRECT \pc, \off, \reg_pg, \rab0, \rc0, \t0
#else
	STNT1	 \rab0\().s, \reg_pg, [\pc, \off , MUL VL]
#endif
.endm
#endif

	#ldnt1w	 \rc0\().s, \reg_pg\()/z, [\pc, \off , MUL VL]
	#fmla    \rc0\().s,\reg_pg\()/M,\rab0\().s,alpha0
.macro SAVE_ZACOL_1VL pc,off, za0,za1,za2,za3,base_idx,idx,nidx,reg_pg0,reg_pg1,rab0,rc0,rab1,rc1,t0,t1,t2=z25,t3=z26
	SAVE_ZACOL  \pc, #0, \za0,\base_idx,\idx,\nidx,\reg_pg0,\rab0,\rc0,\t0,\t1
	add \pc, \pc, LDC, LSL #2 // pc = pc  +  4 * ldc
.endm

.macro SAVE_ZACOL_2VL  pc,off, za0,za1,za2,za3,base_idx,idx,nidx,reg_pg0,reg_pg1,rab0,rc0,rab1,rc1,t0,t1,t2=z25,t3=z26
	SAVE_ZACOL  \pc, #0, \za0,\base_idx,\idx,\nidx,\reg_pg0,\rab0,\rc0,\t0,\t1
	SAVE_ZACOL  \pc, #1, \za1,\base_idx,\idx,\nidx,\reg_pg1,\rab1,\rc1,\t0,\t1
	add \pc, \pc, LDC, LSL #2 // pc = pc  +  4 * ldc
.endm

.macro SAVE_ZACOL_3VL  pc,off, reg_za0,reg_za1,reg_za2,reg_za3,base_idx,idx,nidx,reg_pg0,reg_pg1,rab0,rc0,rab1,rc1,t0,t1,t2=z25,t3=z26
	SAVE_ZACOL  \pc, #0, \reg_za0,\base_idx,\idx,\nidx,\reg_pg0,\rab0,\rc0,\t0,\t1
	SAVE_ZACOL  \pc, #1, \reg_za1,\base_idx,\idx,\nidx,\reg_pg0,\rab1,\rc1,\t0,\t1
	SAVE_ZACOL  \pc, #2, \reg_za2,\base_idx,\idx,\nidx,\reg_pg1,\rab0,\rc0,\t0,\t1
	add \pc, \pc, LDC, LSL #2 // pc = pc  +  4 * ldc
.endm

#ifndef SAVE_ZACOL_4VL_IMPL_
.macro SAVE_ZACOL_4VL  pc,off, reg_za0,reg_za1,reg_za2,reg_za3,base_idx,idx,nidx,reg_pg0,reg_pg1,rab0,rc0,rab1,rc1,t0,t1,t2=z25,t3=z26
	SAVE_ZACOL  \pc, #0, \reg_za0,\base_idx,\idx,\nidx,\reg_pg0,\rab0,\rc0,\t0,\t1
	SAVE_ZACOL  \pc, #1, \reg_za1,\base_idx,\idx,\nidx,\reg_pg0,\rab1,\rc1,\t0,\t1
	SAVE_ZACOL  \pc, #2, \reg_za2,\base_idx,\idx,\nidx,\reg_pg0,\t0,\t1,\rab0,\rc0
	SAVE_ZACOL  \pc, #3, \reg_za3,\base_idx,\idx,\nidx,\reg_pg1,\t2,\t3,\rab0,\rc0
	add \pc, \pc, LDC, LSL #2 // pc = pc  +  4 * ldc
.endm
#endif

.macro KERNEL_SAVE_C_VAL label,num,mvl,nvl,pc,c0=za0,c1=za1,c2 = za2,c3 = za3
	mov TMP_CNT,\num
	mov w12, #0
	mov TMP_PTR1,\pc
.loop_save_c_j_\mvl\()_\nvl\()_\label:
	SAVE_ZACOL_\mvl \pc, #0,\c0,\c1,\c2,\c3,w12,#0,#1,p1,p2, z0,z1,z2,z3,z25,z26
	add w12,w12,#SAVE_CNT_STEP
	add TMP_PTR1,TMP_PTR1,LDC
	mov \pc, TMP_PTR1
	subs TMP_CNT,TMP_CNT,#1
	bgt .loop_save_c_j_\mvl\()_\nvl\()_\label
.endm

#ifndef KERNEL_SAVE_C_BASE_1VL_IMPL_
.macro KERNEL_SAVE_C_BASE_1VL_  mvl,nvl,c0=za0,c1=za1,c2 = za2,c3 = za3
#ifdef DOUBLE
	SAVE_ZACOL_\mvl pC0, #0,\c0,\c1,\c2,\c3,w12,#0,#1,p1,p2, z0,z1,z2,z3,z4,z5,z6,z7
	SAVE_ZACOL_\mvl pC1, #0,\c0,\c1,\c2,\c3,w12,#1,#1,p1,p2, z8,z9,z10,z11,z12,z13,z14,z15
	SAVE_ZACOL_\mvl pC2, #0,\c0,\c1,\c2,\c3,w13,#0,#1,p1,p2, z16,z17,z18,z19,z20,z21,z22,z23
	SAVE_ZACOL_\mvl pC3, #0,\c0,\c1,\c2,\c3,w13,#1,#1,p1,p2, z24,z0,z1,z2, z3,z4,z5,z6
	SAVE_ZACOL_\mvl pC0, #0,\c0,\c1,\c2,\c3,w14,#0,#1,p1,p2, z7,z8,z9,z10,z11,z12,z13,z14
    SAVE_ZACOL_\mvl pC1, #0,\c0,\c1,\c2,\c3,w14,#1,#1,p1,p2, z15,z16,z17,z18,z19,z20,z21,z22
    SAVE_ZACOL_\mvl pC2, #0,\c0,\c1,\c2,\c3,w15,#0,#1,p1,p2, z23,z24,z0,z1,z2,z3,z4,z5
    SAVE_ZACOL_\mvl pC3, #0,\c0,\c1,\c2,\c3,w15,#1,#1,p1,p2, z6,z7,z8,z9,z10,z11,z12,z13
#else
	SAVE_ZACOL_\mvl pC0, #0,\c0,\c1,\c2,\c3,w12,#0,#1,p1,p2, z0,z1,z2,z3,z25,z26
	SAVE_ZACOL_\mvl pC1, #0,\c0,\c1,\c2,\c3,w12,#1,#1,p1,p2, z4,z5,z6,z7,z25,z26
	SAVE_ZACOL_\mvl pC2, #0,\c0,\c1,\c2,\c3,w12,#2,#1,p1,p2, z8,z9,z10,z11,z25,z26
	SAVE_ZACOL_\mvl pC3, #0,\c0,\c1,\c2,\c3,w12,#3,#1,p1,p2, z12,z13,z14,z15 ,z25,z26
	SAVE_ZACOL_\mvl pC0, #0,\c0,\c1,\c2,\c3,w13,#0,#1,p1,p2, z0,z1,z2,z3,z25,z26
	SAVE_ZACOL_\mvl pC1, #0,\c0,\c1,\c2,\c3,w13,#1,#1,p1,p2, z4,z5,z6,z7,z25,z26
	SAVE_ZACOL_\mvl pC2, #0,\c0,\c1,\c2,\c3,w13,#2,#1,p1,p2, z8,z9,z10,z11,z25,z26
	SAVE_ZACOL_\mvl pC3, #0,\c0,\c1,\c2,\c3,w13,#3,#1,p1,p2, z12,z13,z14,z15,z25,z26
	SAVE_ZACOL_\mvl pC0, #0,\c0,\c1,\c2,\c3,w14,#0,#1,p1,p2, z0,z1,z2,z3,z25,z26
	SAVE_ZACOL_\mvl pC1, #0,\c0,\c1,\c2,\c3,w14,#1,#1,p1,p2, z4,z5,z6,z7,z25,z26
	SAVE_ZACOL_\mvl pC2, #0,\c0,\c1,\c2,\c3,w14,#2,#1,p1,p2, z8,z9,z10,z11,z25,z26
	SAVE_ZACOL_\mvl pC3, #0,\c0,\c1,\c2,\c3,w14,#3,#1,p1,p2, z12,z13,z14,z15,z25,z26
	SAVE_ZACOL_\mvl pC0, #0,\c0,\c1,\c2,\c3,w15,#0,#1,p1,p2, z0,z1,z2,z3,z25,z26
	SAVE_ZACOL_\mvl pC1, #0,\c0,\c1,\c2,\c3,w15,#1,#1,p1,p2, z4,z5,z6,z7,z25,z26
	SAVE_ZACOL_\mvl pC2, #0,\c0,\c1,\c2,\c3,w15,#2,#1,p1,p2, z8,z9,z10,z11,z25,z26
	SAVE_ZACOL_\mvl pC3, #0,\c0,\c1,\c2,\c3,w15,#3,#1,p1,p2, z12,z13,z14,z15,z25,z26
#endif
.endm
#endif

.macro KERNEL_SAVE_C_BASE_1VL label,mvl,nvl,c0=za0,c1=za1,c2 = za2,c3 = za3
	mov w12, #0
#ifdef DOUBLE
	mov w13, #2
	mov w14, #4
	mov w15, #6
#else
	mov w13, #4
	mov w14, #8
	mov w15, #12
#endif
	ands TMP_CNT_POST, MIN_N, ELENUM
	bne .kernel_save_c_val_\mvl\()_\nvl\()_\label

	add pC1,pC0,LDC
	add pC2,pC0,LDC,LSL #1
	add pC3,pC1,LDC,LSL #1

	KERNEL_SAVE_C_BASE_1VL_ \mvl,\nvl,\c0,\c1,\c2,\c3

	b .kernel_save_c_val_\mvl\()_\nvl\()_\label\()_end
.kernel_save_c_val_\mvl\()_\nvl\()_\label:
	KERNEL_SAVE_C_VAL \label,TMP_CNT_POST,\mvl,\nvl,pC0,\c0,\c1,\c2,\c3
.kernel_save_c_val_\mvl\()_\nvl\()_\label\()_end:
.endm

.macro KERNEL_SAVE_C_1VL_1VL label
	mov TMP_PTR, pC0
	KERNEL_SAVE_C_BASE_1VL \label,1VL, 1VL,za0
	add pC0, TMP_PTR, MIN_M, LSL #PRECI_WIDTH
.endm
.macro KERNEL_SAVE_C_2VL_1VL label
	mov TMP_PTR, pC0
	KERNEL_SAVE_C_BASE_1VL \label,2VL, 1VL,za0,za2
	add pC0, TMP_PTR, MIN_M, LSL #PRECI_WIDTH
.endm
.macro KERNEL_SAVE_C_3VL_1VL label
	mov TMP_PTR, pC0
	KERNEL_SAVE_C_BASE_1VL \label,3VL,1VL, za0,za2,za4
	add pC0, TMP_PTR, MIN_M, LSL #PRECI_WIDTH
.endm
.macro KERNEL_SAVE_C_4VL_1VL label
	mov TMP_PTR, pC0
	KERNEL_SAVE_C_BASE_1VL \label,4VL,1VL,za0,za2,za4,za6
	add pC0, TMP_PTR, MIN_M, LSL #PRECI_WIDTH
.endm

.macro KERNEL_SAVE_C_BASE_2VL label,mvl
    mov w12, #0

	mov w13, #4
	mov w14, #8
	mov w15, #12

	mov TMP_PTR, pC0
	add pC1,pC0,LDC
	add pC2,pC0,LDC,LSL #1
	add pC3,pC1,LDC,LSL #1
	KERNEL_SAVE_C_BASE_1VL_ \mvl,2VL, za0,za2,za4,za6
	KERNEL_SAVE_C_BASE_1VL \label,\mvl, 2VL,za1,za3,za5,za7
	add pC0, TMP_PTR, MIN_M, LSL #PRECI_WIDTH
.endm

.macro KERNEL_SAVE_C_1VL_2VL label
	KERNEL_SAVE_C_BASE_2VL \label,1VL
.endm
.macro KERNEL_SAVE_C_2VL_2VL label
    KERNEL_SAVE_C_BASE_2VL \label,2VL
.endm
.macro KERNEL_SAVE_C_3VL_2VL label
	KERNEL_SAVE_C_BASE_2VL \label,3VL
.endm

#ifndef KERNEL_SAVE_C_4VL_2VL_IMPL_
.macro KERNEL_SAVE_C_4VL_2VL label
	KERNEL_SAVE_C_BASE_2VL \label,4VL
.endm
#endif

.macro UPDATE_PRMC_CNT cnt,off,ptrpc
	add \cnt,\cnt,1
	sub \off,\off,\off
	add \ptrpc,\ptrpc,LDC
.endm

.macro KERNEL_MM_LOOP_KK label,mvl,nvl,kernel_prfmc,kernel_nprfmc,knl_lu,prfpc=ptrcp
.\label\()_\mvl\()_\nvl\()_prfm_c: // evict b from cache at last m block
    cmp wbk, #0
	ble \knl_lu
    \kernel_prfmc \mvl \nvl

	sub	wbk, wbk, #1
//看看C的预取是否已经完成
	// sgemm
	//cmp offset_pfc,#128

	//b.ne .\label\()_\mvl\()_\nvl\()_prfm_c
    //UPDATE_PRMC_CNT PRC_CNT, offset_pfc, \prfpc
    //cmp PRC_CNT, PRFC_NUM
    //b.lt .\label\()_\mvl\()_\nvl\()_prfm_c
// 不在预取C
.\label\()_\mvl\()_\nvl\()_no_prfm_c:
    cmp wbk, #0
	ble \knl_lu
	\kernel_nprfmc \mvl \nvl
	sub	wbk, wbk, #1
	b	.\label\()_\mvl\()_\nvl\()_no_prfm_c
.endm

.macro KERNEL_MM_LOOP_K_LU label,kernel,cnt,mvl,nvl
.\label\()_\mvl\()_\nvl\()_k:
#if defined(SMALL_SME_GEMM)
    cmp \cnt, #0
    beq .L_deal_last_k_\label\()_\mvl\()_\nvl\()

    KERNEL_M0 \mvl \nvl
    sub \cnt,\cnt,#1
	b .\label\()_\mvl\()_\nvl\()_k
#else
    KERNEL_M0 \mvl \nvl
    sub \cnt,\cnt,#1
	cmp \cnt ,#0
	bgt .\label\()_\mvl\()_\nvl\()_k
    b .L_end_\label\()_\mvl\()_\nvl\()_k
#endif

.L_deal_last_k_\label\()_\mvl\()_\nvl\():
    ldr TMP_CNT, [sp, #(11*16)]     // 获取最后小于等于4的部分

    // if TMP_CNT=0 即k被4整除，不做处理
    cmp TMP_CNT, #0
    beq .L_end_\label\()_\mvl\()_\nvl\()_k

    // else TMP_CNT=1/2/3 做最后一次k方向上的计算
    DEAL_LAST_K TMP_CNT
    KERNEL_M0 \mvl \nvl
    sub \cnt,\cnt,\cnt

.L_end_\label\()_\mvl\()_\nvl\()_k:
    
.endm

.macro GET_ACTIVE_NUM_D reg_pg,out
	sub \out,\out,\out
    cntp \out, \reg_pg, \reg_pg\().d
.endm
.macro GET_ACTIVE_NUM_S reg_pg,out
	sub \out,\out,\out
    cntp \out, \reg_pg, \reg_pg\().s
.endm
.macro GET_ACTIVE_NUM_H reg_pg,out
	sub \out,\out,\out
    cntp \out, \reg_pg, \reg_pg\().h
.endm

.macro KERNEL_MM_LOOP_K_LDNTA label,mvl,nvl,k,is,m,min_m,cnt
#if defined(SMALL_SME_GEMM)
    mov TMP_PTR, #0xffffffff
    dup z29.s, TMP_PTR_SIN
#endif
	LSR wbk,\k,#3
	sub \cnt, \m, \is
    cmp \cnt, \min_m
    b.le .k_loop_ldnta_last_m_\mvl\()_\nvl\()_prfm_c
	KERNEL_MM_LOOP_KK k_loop_ldnta_m,\mvl,\nvl,KERNEL_PRFMBC_LDNTA,KERNEL_3m_PRFMB_LDNTA,KERNEL_MM_LU_ldnta_\mvl\()_\nvl\()_K,ptrcp
	KERNEL_MM_LOOP_KK k_loop_ldnta_last_m,\mvl,\nvl,KERNEL_LDNTB_PRFMBC_LDNTA,KERNEL_LDNTB_PRFMB_LDNTA,KERNEL_MM_LU_ldnta_\mvl\()_\nvl\()_K

	KERNEL_MM_LU_ldnta_\mvl\()_\nvl\()_K:
#if !defined(SMALL_SME_GEMM)
	tst origK, #7
	ble .SAVE_C_ldnta_\mvl\()_\nvl
#endif
	ANDS TMP_CNT,origK, #7
	KERNEL_MM_LOOP_K_LU k_mm_m_lu_k_ldnta,KERNEL_M0,TMP_CNT, \mvl,\nvl
.SAVE_C_ldnta_\mvl\()_\nvl:
	KERNEL_SAVE_C_\mvl\()_\nvl \label
	b .end_of_loop_k_\nvl\()_\label
.endm

.macro KERNEL_MM_LOOP_K label,mvl,nvl,k,is,m,min_m,cnt

#if defined(SMALL_SME_GEMM)
    mov TMP_PTR, #0xffffffff
    dup z29.s, TMP_PTR_SIN
#endif
//如果K方向循环展开，这里需要改成#4
	LSR wbk,\k,#3
	sub \cnt, \m, \is
    cmp \cnt, \min_m
    b.le .\label\()_\mvl\()_\nvl\()_last_m_loopk//.k_loop_last_m_\()\label\()_\mvl\()_\nvl\()_prfm_c
	KERNEL_MM_LOOP_KK k_loop_m_\()\label,\mvl,\nvl,KERNEL_PRFMBC,KERNEL_3m_PRFMB,KERNEL_MM_LU_\mvl\()_\nvl\()_K_\label

//! testtest
// b .end_of_loop_k_\nvl\()_\label

.\label\()_\mvl\()_\nvl\()_last_m_loopk:
	MUL TMP_CNT, MIN_N, LDC // \stepNum * ldc * sizeof(double/float)
	LSR TMP_CNT, TMP_CNT, #PC_IMM
	add ptrcp, pC,TMP_CNT
	//M方向最后一次迭代
	KERNEL_MM_LOOP_KK k_loop_last_m_\()\label,\mvl,\nvl,KERNEL_LDNTB_PRFMBC,KERNEL_LDNTB_PRFMB,KERNEL_MM_LU_\mvl\()_\nvl\()_K_\label
//KERNEL_MM_LOOP_KK里面会跳转到这里
KERNEL_MM_LU_\mvl\()_\nvl\()_K_\label:
#if !defined(SMALL_SME_GEMM)
//如果K方向循环展开，这里和下面的#7都要改成#15
	tst origK, #7
	ble .SAVE_C_\mvl\()_\nvl\()_\label
#endif
	ANDS TMP_CNT,origK, #7
	//K方向外层最后一次迭代
	KERNEL_MM_LOOP_K_LU k_mm_m_lu_k_\label,KERNEL_M0,TMP_CNT, \mvl,\nvl
.SAVE_C_\mvl\()_\nvl\()_\label:
	KERNEL_SAVE_C_\mvl\()_\nvl \label
	b .end_of_loop_k_\nvl\()_\label
.endm

.macro UPDATE_PG_BY_CNTR reg_pg,reg_mopa_pg,start,end,step
	whilelt \reg_pg\().s, \start,\end
	add \start,\start,\step
	cntp TMP_OFF, \reg_pg, \reg_pg\().s
	lsl TMP_OFF, TMP_OFF, #2
	whilelt \reg_mopa_pg\().b, XZR, TMP_OFF
	mov	TMP_OFF, #0
.endm
#ifndef KERNEL_MM_LOOP_M_PRE_FUNC_
.macro KERNEL_MM_LOOP_M_PRE_FUNC
	mov pB0, pBt
	mov pA0, pAt

.endm
#endif
#ifndef KERNEL_MM_LOOP_M_POST_FUNC_
.macro KERNEL_MM_LOOP_M_POST_FUNC
//#if !defined(USE_HBM_KERNEL)
	mov pAt, pA0
//#endif
.endm
#endif
#ifndef KERNEL_INIT_
.macro KERNEL_INIT

.endm
#endif
// stepNum : MIN_M, stepNum:MIN_N
.macro KERNEL_MM_LOOP_L2 label,nvl,loopNum,stepNum
	b .cond_of_loops_m_\nvl\()_\label
.loops_of_m_\nvl\()_\label:
	KERNEL_INIT
    ZERO	{za0.B}
	//设置MIN_M（stepNum） 
	sub \stepNum, \stepNum,\stepNum
	cntp \stepNum, p1, p1.h
	KERNEL_MM_LOOP_M_PRE_FUNC

	// sgemm  设置C的预取指针
	add ptrcp, pC0,#128

	sub PRC_CNT,PRC_CNT,PRC_CNT
	sub offset_pfc,offset_pfc,offset_pfc

.loop_m_2vl_\nvl\()_\label:
	cmp \stepNum, ELE_SIZE
	ble .loop_m_1vl_\nvl\()_\label
	sub TMP_CNT, TMP_CNT,TMP_CNT
	UPDATE_PG_BY_CNTR p1, Pg_MOPA_1, TMP_CNT, \stepNum, ELE_SIZE
	UPDATE_PG_BY_CNTR p2, Pg_MOPA_2, TMP_CNT, \stepNum, ELE_SIZE
	// k方向loop
	KERNEL_MM_LOOP_K \label,2VL,\nvl,origK, counterI, \loopNum, \stepNum,TMP_CNT
.loop_m_1vl_\nvl\()_\label:
	sub TMP_CNT, TMP_CNT,TMP_CNT
	UPDATE_PG_BY_CNTR p1, Pg_MOPA_1, TMP_CNT, \stepNum, ELE_SIZE
	GET_ACTIVE_NUM_D p1,TMP_CNT

	KERNEL_MM_LOOP_K \label,1VL,\nvl,origK, counterI, \loopNum, \stepNum,TMP_CNT
.end_of_loop_k_\nvl\()_\label:
//M方向counter+1
	add  counterI , counterI,\stepNum

	KERNEL_MM_LOOP_M_POST_FUNC
.cond_of_loops_m_\nvl\()_\label:
	lsl TMP_CNT,origM,#DIMENSION
	whilelt p1.h, counterI, TMP_CNT
	b.first .loops_of_m_\nvl\()_\label
.endm

.macro KERNEL_MM_LOOP_L2_LDNTA label,nvl,loopNum,stepNum
	b .cond_of_loops_m_\nvl\()_\label
.loops_of_m_\nvl\()_\label:
	KERNEL_INIT
    ZERO	{za0.B}
	sub \stepNum, \stepNum,\stepNum
	cntp \stepNum, p1, p1.h
	KERNEL_MM_LOOP_M_PRE_FUNC
	add ptrcp, pC0, #0
	sub PRC_CNT,PRC_CNT,PRC_CNT
	sub offset_pfc,offset_pfc,offset_pfc

.loop_m_2vl_\nvl\()_\label:
	cmp \stepNum, ELE_SIZE
	ble .loop_m_1vl_\nvl\()_\label
	sub TMP_CNT, TMP_CNT,TMP_CNT
	UPDATE_PG_BY_CNTR p1, Pg_MOPA_1, TMP_CNT, \stepNum, ELE_SIZE
	UPDATE_PG_BY_CNTR p2, Pg_MOPA_2, TMP_CNT, \stepNum, ELE_SIZE
	KERNEL_MM_LOOP_K \label,2VL,\nvl,origK, counterI, \loopNum, \stepNum,TMP_CNT
.loop_m_1vl_\nvl\()_\label:
	sub TMP_CNT, TMP_CNT,TMP_CNT
	UPDATE_PG_BY_CNTR p1, Pg_MOPA_1, TMP_CNT, \stepNum, ELE_SIZE
	GET_ACTIVE_NUM_D p1,TMP_CNT

	KERNEL_MM_LOOP_K \label,1VL,\nvl,origK, counterI, \loopNum, \stepNum,TMP_CNT
.end_of_loop_k_\nvl\()_noldnta_\label:
.end_of_loop_k_\nvl\()_\label:
	add  counterI , counterI,\stepNum

	KERNEL_MM_LOOP_M_POST_FUNC
.cond_of_loops_m_\nvl\()_\label:
	lsl TMP_CNT,origM,#DIMENSION
	whilelt p1.h, counterI, TMP_CNT
	b.first .loops_of_m_\nvl\()_\label
.endm

#ifndef KERNEL_MM_LOOP_N_PRE_FUNC_
.macro KERNEL_MM_LOOP_N_PRE_FUNC
	mov pAt, origPA
	mov pB0, pBt

	MUL TMP_CNT, origK , MIN_N
	// add ptrbp, pBt, TMP_CNT,lsl #PRECI_WIDTH
	// add ptrbp, pBt, origK,lsl #1
	mov ptrbp, pBt

.endm
#endif
#ifndef KERNEL_MM_LOOP_N_POST_FUNC_
.macro KERNEL_MM_LOOP_N_POST_FUNC
	mov pBt, pB0
.endm
#endif

.macro origK_ALIGN
#if !defined(SMALL_SME_GEMM)
    add origK, origK, #3
#endif
	LSR origK, origK, #2
.endm

.macro KERNEL_MM_LOOP_L1 precs,loopNum1,loopNum2,stepNum1,stepNum2
	origK_ALIGN
	b .cond_of_loops_n
.loops_of_n:
	mov pC0, pC
	sub counterI, counterI, counterI
	// 保存原始指针 stepNum1=32（即MIN_N） ELE_SIZE=16
	KERNEL_MM_LOOP_N_PRE_FUNC
	cmp \stepNum1,ELE_SIZE
	ble .loops_of_l1_1vl
.loops_of_l1_2vl:
	sub TMP_CNT, TMP_CNT,TMP_CNT
	UPDATE_PG_BY_CNTR p0, Pg_MOPA_0, TMP_CNT, \stepNum1, ELE_SIZE
	UPDATE_PG_BY_CNTR p3, Pg_MOPA_3, TMP_CNT, \stepNum1, ELE_SIZE

// 判断是否是n方向最后一次迭代
#if defined(INVALID_LAST_N)
	sub TMP_CNT, origN, counterJ
	cmp TMP_CNT, MIN_N
	b.le .loops_l1_2vl_last_n
#endif

	KERNEL_MM_LOOP_L2 k_loop_m,2VL, \loopNum2, \stepNum2

#if defined(INVALID_LAST_N)
	b .loops_of_l1_1vl
// n方向最后一次迭代
.loops_l1_2vl_last_n:
	KERNEL_MM_LOOP_L2_LDNTA k_loop_m_ldnta,2VL,\loopNum2, \stepNum2
#endif

.loops_of_l1_1vl:
	sub TMP_CNT, TMP_CNT,TMP_CNT
	UPDATE_PG_BY_CNTR p0, Pg_MOPA_0, TMP_CNT, \stepNum1, ELE_SIZE
	KERNEL_MM_LOOP_L2 k_loop_m,1VL,\loopNum2,\stepNum2
.end_of_loops_m:


	add counterJ,counterJ,\stepNum1
	KERNEL_MM_LOOP_N_POST_FUNC
	// 偏移C指针
	MUL TMP_CNT, \stepNum1, LDC // \stepNum * ldc * sizeof(double/float)
	LSR TMP_CNT, TMP_CNT, #PC_IMM
	add pC, pC, TMP_CNT
.cond_of_loops_n:
	LSL TMP_CNT, \loopNum1, #DIMENSION  // for complex *2
	whilelt p0.\precs, counterJ,TMP_CNT
	sub \stepNum1, \stepNum1, \stepNum1
	cntp \stepNum1, p0, p0.\precs
	b.first .loops_of_n
.endm
#ifndef KERNEL_MM_LOOP_M_
.macro KERNEL_MM_LOOP_M pre=s
	KERNEL_MM_LOOP_L1 \pre, origM, origN,MIN_M ,MIN_N
.endm
#endif

#ifndef KERNEL_MM_LOOP_N_
.macro KERNEL_MM_LOOP_N pre=s
	KERNEL_MM_LOOP_L1 \pre,origN,origM,MIN_N,MIN_M
.endm
#endif