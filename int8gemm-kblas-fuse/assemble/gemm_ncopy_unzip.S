/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: igemm tcopy implementation
 * Author: KPL
 * Create: 2025-04-01
 * Notes: NA
*/

#define ASSEMBLER
#include "common.h"

/*
    入参：M, N, A00, LDA, B00
    功能：从矩阵A00中取一个大小为M*N的矩阵，并对其packing，packing结果存储到B00中
    参数说明：
            1. M/N：M和N确定本次需要做packing操作的矩阵大小
            2. LDA：与矩阵行列主序无关。即若矩阵是行主序，LDA表示一行的元素个数；若矩阵是列主序，LDA表示一列的元素个数。
            3. M：M <= LDA，M值与LDA值属于同维度。即若矩阵是行主序，则M为一行需要读取的元素个数；若矩阵是列主序，则M为一列需要读取的元素个数
 */
#define     M               x0
#define     N               x1
#define     A00             x2
#define     LDA             x3
#define     LDA_TMP         w3
#define     B00             x4      // packing result is stored in B00
#define     offset          w5
#define     A01             x8
#define     A02             x9
#define     A03             x15
#define     A04             x16
#define     M_tail          x10

#define     I               x11
#define     J               x12
#define     MIN_N           x13

#define     CNT             x14
#define     TMP_CNT         x17
#define     TMP_VAL         x18
#define     TMP_CNT_2       x19

#define     offset_vec      z29
#define     offset_vec3     z28
#define     offset_vec2     z27
#define     offset_vec1     z26

#define     TMP_P0          p0
#define     TMP_P1          p1
#define     TMP_P2          p2
#define     TMP_P6          p6
#define     TMP_Pall        p7
/*******************************************************************************
 * Macro definitions begin
 *******************************************************************************/

.macro COPY1x4VL
    ld1w    {z0.s}, p3/z, [A01, z6.s, UXTW]
    add     z0.b, TMP_P0/m, z0.b, offset_vec.b
    st1w    {z0.s}, p3, [B00]
    add     A01, A01, #4     
    add     B00, B00, MIN_N, lsl #2
.endm

.macro COPY_TAIL_3VL    
    ld1h    {z0.s}, p3/z, [A01, z6.s, UXTW]     // 0  0 10  9
    ld1b    {z1.s}, p3/z, [A02, z6.s, UXTW]     // 0  0  0 11
    lsl     z1.s, z1.s, #16                     // 0 11  0  0 (logic shift left)
    orr     z0.d, z0.d, z1.d                    // 0 11 10  9
    add     z0.b, TMP_P0/m, z0.b, offset_vec3.b
    st1w    {z0.s}, p3, [B00]                   // 9 10 11  0
    add     B00, B00, MIN_N, lsl #2 
.endm

.macro COPY_TAIL_2VL   
    ld1h    {z0.s}, p3/z, [A01, z6.s, UXTW]
    add     z0.b, TMP_P0/m, z0.b, offset_vec2.b
    st1w    {z0.s}, p3, [B00]
    add     B00, B00, MIN_N, lsl #2 
.endm

.macro COPY_TAIL_1VL
    ld1b    {z0.s}, p3/z, [A01, z6.s, UXTW]
    add     z0.b, TMP_P0/m, z0.b, offset_vec1.b
    st1w    {z0.s}, p3, [B00]
    add     B00, B00, MIN_N, lsl #2 
.endm

.macro COPY1x4VLx2
    ld1w    {z0.s}, p3/z, [A01, z6.s, UXTW]
    add     z0.b, TMP_P0/m, z0.b, offset_vec.b
    st1w    {z0.s}, p3, [B00]
    add     A01, A01, #4     

    ld1w    {z2.s}, p4/z, [A03, z6.s, UXTW]
    add     z2.b, TMP_P1/m, z2.b, offset_vec.b
    st1w    {z2.s}, p4, [B00, #1, MUL VL]
    add     A03, A03, #4     
    add     B00, B00, MIN_N, lsl #2
.endm

.macro COPY_TAIL_3VLx2
    ld1h    {z0.s}, p3/z, [A01, z6.s, UXTW]     // 0  0 10  9
    ld1b    {z1.s}, p3/z, [A02, z6.s, UXTW]     // 0  0  0 11
    lsl     z1.s, z1.s, #16                     // 0 11  0  0 (logic shift left)
    orr     z0.d, z0.d, z1.d                    // 0 11 10  9
    add     z0.b, TMP_P0/m, z0.b, offset_vec3.b
    st1w    {z0.s}, p3, [B00]                   // 9 10 11  0

    ld1h    {z2.s}, p4/z, [A03, z6.s, UXTW]
    ld1b    {z3.s}, p4/z, [A04, Z6.s, UXTW]
    lsl     z3.s, z3.s, #16
    orr     z2.d, z2.d, z3.d
    add     z2.b, TMP_P1/m, z2.b, offset_vec3.b
    st1w    {z2.s}, p4, [B00, #1, MUL VL] 

    add     B00, B00, MIN_N, lsl #2
.endm

.macro COPY_TAIL_2VLx2
    ld1h    {z0.s}, p3/z, [A01, z6.s, UXTW]
    add     z0.b, TMP_P0/m, z0.b, offset_vec2.b
    st1w    {z0.s}, p3, [B00]
    ld1h    {z2.s}, p4/z, [A03, z6.s, UXTW]
    add     z2.b, TMP_P1/m, z2.b, offset_vec2.b
    st1w    {z2.s}, p4, [B00, #1, MUL VL]
    add     B00, B00, MIN_N, lsl #2 
.endm

.macro COPY_TAIL_1VLx2
    ld1b    {z0.s}, p3/z, [A01, z6.s, UXTW]
    add     z0.b, TMP_P0/m, z0.b, offset_vec1.b
    st1w    {z0.s}, p3, [B00]
    ld1b    {z2.s}, p4/z, [A03, Z6.s, UXTW]
    add     z2.b, TMP_P1/m, z2.b, offset_vec1.b
    st1w    {z2.s}, p4, [B00, #1, MUL VL]
    add     B00, B00, MIN_N, lsl #2 
.endm

.macro UPDATE_PG_BY_CNTR reg_pg, offset_reg_pg, start,end,step
	whilelt \reg_pg\().s, \start,\end
	add \start,\start,\step
    cntp    TMP_CNT_2, \reg_pg, \reg_pg\().s
    LSL     TMP_CNT_2, TMP_CNT_2, #2
    whilelt \offset_reg_pg\().b, XZR, TMP_CNT_2
.endm

.macro INIT_OFFSET_VEC
    dup     offset_vec.b, offset

    mov     w18, #0xff
    dup     offset_vec1.s, w18
    mov     w18, #0xffff
    dup     offset_vec2.s, w18
    mov     w18, #0xffffff
    dup     offset_vec3.s, w18

    ptrue   TMP_Pall.b, all

    and     offset_vec1.b, TMP_Pall/m, offset_vec1.b, offset_vec.b
    and     offset_vec2.b, TMP_Pall/m, offset_vec2.b, offset_vec.b
    and     offset_vec3.b, TMP_Pall/m, offset_vec3.b, offset_vec.b
.endm
/*******************************************************************************
 * Macro definitions end
 *******************************************************************************/

    PROLOGUE
    .align  5
    SAVE_REGS
    and     M_tail, M, #3                   // M_tail = M % 4
    asr     M, M, #2                        // M /= 4
    index   z6.s, #0, LDA_TMP
    mov     J, N 
    INIT_OFFSET_VEC
    mov     TMP_VAL, #16

.Ligemm_ncopy_BEGIN:
    mov     CNT, #0
    whilelt p5.h, CNT, J  
    
    mov     MIN_N, #0
    cntp    MIN_N, p5, p5.h
    cmp     MIN_N, #0
    
    ble     .Ligemm_ncopy_L999

    mov     A01, A00
    add     A03, A01, LDA, LSL #4
    add     A00, A00, LDA, LSL #5

    mov     TMP_CNT, #0
    UPDATE_PG_BY_CNTR   p3, TMP_P0, TMP_CNT, MIN_N, TMP_VAL
    UPDATE_PG_BY_CNTR   p4, TMP_P1, TMP_CNT, MIN_N, TMP_VAL

    cmp     M, #0
    ble     .Ligemm_ncopy_x2_TAIL_BEGIN        // M <= 3

    cmp     MIN_N, #16
    ble     .Ligemm_ncopy_4VL_BEGIN // N <= 16


/*******************************************************************************
 * 16x2 for N
 *******************************************************************************/ 

.Ligemm_ncopy_4VLx2_BEGIN:                    // M >= 4
    mov     I, M

.Ligemm_ncopy_4VLx2:
    COPY1x4VLx2
    subs    I, I, #1
    bne     .Ligemm_ncopy_4VLx2

.Ligemm_ncopy_x2_TAIL_BEGIN:                   // M <= 3
    cmp     MIN_N, #16
    ble     .Ligemm_ncopy_TAIL_BEGIN
    cmp     M_tail, #0
    beq     .Ligemm_ncopy_TAIL_END
    cmp     M_tail, #2
    blt     .Ligemm_ncopy_1VLx2_TAIL
    beq     .Ligemm_ncopy_2VLx2_TAIL
    bgt     .Ligemm_ncopy_3VLx2_TAIL

.Ligemm_ncopy_3VLx2_TAIL:
    add     A02, A01, #2
    add     A04, A03, #2
    COPY_TAIL_3VLx2
    b       .Ligemm_ncopy_TAIL_END

.Ligemm_ncopy_2VLx2_TAIL:
    COPY_TAIL_2VLx2
    b       .Ligemm_ncopy_TAIL_END

.Ligemm_ncopy_1VLx2_TAIL:
    COPY_TAIL_1VLx2
    b       .Ligemm_ncopy_TAIL_END

/*******************************************************************************
 * 16x1 for N
 *******************************************************************************/ 
/*******************************************************************************
 * Normal condition
 *******************************************************************************/ 

.Ligemm_ncopy_4VL_BEGIN:                    // M >= 4
    mov     I, M

.Ligemm_ncopy_4VL:
    COPY1x4VL
    subs    I, I, #1
    bne     .Ligemm_ncopy_4VL

/*******************************************************************************
 * Boundary condition
 *******************************************************************************/    
        
.Ligemm_ncopy_TAIL_BEGIN:                   // M <= 3
    cmp     M_tail, #0
    beq     .Ligemm_ncopy_TAIL_END
    cmp     M_tail, #2
    blt     .Ligemm_ncopy_1VL_TAIL
    beq     .Ligemm_ncopy_2VL_TAIL
    bgt     .Ligemm_ncopy_3VL_TAIL

.Ligemm_ncopy_3VL_TAIL:
    add     A02, A01, #2
    COPY_TAIL_3VL
    b       .Ligemm_ncopy_TAIL_END

.Ligemm_ncopy_2VL_TAIL:
    COPY_TAIL_2VL
    b       .Ligemm_ncopy_TAIL_END

.Ligemm_ncopy_1VL_TAIL:
    COPY_TAIL_1VL

.Ligemm_ncopy_TAIL_END:
    subs    J, J, MIN_N
    bgt     .Ligemm_ncopy_BEGIN
        
.Ligemm_ncopy_L999:
    mov     x0, #0                        // set return value
    RESTORE_REGS
    ret

    EPILOGUE