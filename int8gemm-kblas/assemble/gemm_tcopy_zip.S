/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: igemm ntcopy implementation
 * Author: KPL
 * Create: 2025-04-01
 * Notes: NA
*/

#define ASSEMBLER
#include "common.h"


/*
    入参：M, N, A00, LDA, B00
    功能：从矩阵A00中取一个大小为N*M(N行, M列)的矩阵，并对其packing，packing结果存储到B00中(n行，m列)
    参数说明：
            1. M/N：M和N确定本次需要做packing操作的矩阵大小
            2. LDA：与矩阵行列主序无关。即若矩阵是行主序，LDA表示一行的元素个数；若矩阵是列主序，LDA表示一列的元素个数。
            3. M：M <= LDA，M值与LDA值属于同维度。即若矩阵是行主序，则M为一行需要读取的元素个数；若矩阵是列主序，则M为一列需要读取的元素个数
 */

#define     N               x0
#define     M               x1
#define     A00             x2
#define     LDA             x3
#define     LDA_TMP         w3
#define     B00             x4      // packing result is stored in B00
#define     offset          w5    
#define     B02             x7
#define     LDB             x8

#define     A01             x9
#define     A02             x9,LDA
#define     A03             x11
#define     A04             x11,LDA

#define     I               x13
#define     J               x14

#define     MIN_M           x15
#define     CNT             x16


#define     N_tail          x17
#define     M_tail          x18
#define     M_M_tail        x19
#define     M_MTP_tail      x20
#define     startP          x21
#define     K               x22
#define     Q               x23
#define     P               x24
#define     S               x25
#define     TMP_VAL         x26
#define     TMP_CNT         x27
#define     ELE_SIZE        x28
#define     offset_vec      z29
/*******************************************************************************
 * Macro definitions begin
 *******************************************************************************/

.macro TCOPY4x1VL
    ld1b    {z0.b}, p0/z, [A01]
    ld1b    {z1.b}, p0/z, [A02]

    ld1b    {z2.b}, p0/z, [A03]
    ld1b    {z3.b}, p0/z, [A04]

    add     z0.b, p0/m, z0.b, offset_vec.b
    add     z1.b, p0/m, z1.b, offset_vec.b
    add     z2.b, p0/m, z2.b, offset_vec.b
    add     z3.b, p0/m, z3.b, offset_vec.b

    ALTER_AOFFSET

    ZIP4REG_ST
.endm

.macro ZIP4REG_ST

    ZIP4REG  z0,z1,z2,z3,z8,z9,z10,z11

    st1b    {z8.b}, p0, [B00]
    st1b    {z9.b}, p0, [B00, #1, MUL VL]
    st1b    {z10.b}, p0, [B02]
    st1b    {z11.b}, p0, [B02, #1, MUL VL]

    add     B00, B00, #128
    add     B02, B00, LDB
.endm


.macro ALTER_BTAIL_OFFSET
    add     B00, B00, LDB
.endm

.macro ALTER_AOFFSET
    add     A01, A01, LDA, lsl #2
    add     A03, A03, LDA, lsl #2
.endm

.macro ZIP4REG in_r0, in_r1, in_r2, in_r3, out_r0, out_r1, out_r2, out_r3

    zip1  z4.b,\in_r0\().b,\in_r1\().b
    zip1 z6.b,\in_r2\().b,\in_r3\().b

    zip2  z5.b, \in_r0\().b, \in_r1\().b
    zip2 z7.b,\in_r2\().b,\in_r3\().b

    zip1 \out_r0\().h, z4.h,z6.h
    zip2 \out_r1\().h, z4.h,z6.h
    zip1 \out_r2\().h, z5.h,z7.h
    zip2 \out_r3\().h, z5.h,z7.h
.endm


//N_tail == 1 补 3 行 0
//N_tail == 2 补 2 行 0
//N_tail == 3 补 1 行 0

.macro TCOPY_N_TAIL_3x1VL
    ld1b    {z0.b}, p0/z, [A01]
    ld1b    {z1.b}, p0/z, [A02]
    ld1b    {z2.b}, p0/z, [A03]
    add     z0.b, p0/m, z0.b, offset_vec.b
    add     z1.b, p0/m, z1.b, offset_vec.b
    add     z2.b, p0/m, z2.b, offset_vec.b

    mov     z3.b, #0       // 为空的那一行补零,A01偏移量不变

    ALTER_AOFFSET

    ZIP4REG_ST
.endm


.macro TCOPY_N_TAIL_2x1VL
    ld1b    {z0.b}, p0/z, [A01]
    ld1b    {z1.b}, p0/z, [A02]
    add     z0.b, p0/m, z0.b, offset_vec.b
    add     z1.b, p0/m, z1.b, offset_vec.b
    mov     z2.b, #0
    mov     z3.b, #0

    ALTER_AOFFSET

    ZIP4REG_ST
.endm


.macro TCOPY_N_TAIL_1x1VL
    ld1b    {z0.b}, p0/z, [A01]
    add     z0.b, p0/m, z0.b, offset_vec.b
    mov     z1.b, #0
    mov     z2.b, #0
    mov     z3.b, #0

    ALTER_AOFFSET

    ZIP4REG_ST
.endm



.macro TCOPY_M_TAIL_BEGIN
    mov     startP, #0
    whilelt p4.b, startP, M_tail  //  M_tail个1，64 - M_tail个0

    ld1b    {z0.b}, p4/z, [A01]
    ld1b    {z1.b}, p4/z, [A02]
    ld1b    {z2.b}, p4/z, [A03]
    ld1b    {z3.b}, p4/z, [A04]
    add     z0.b, p4/m, z0.b, offset_vec.b
    add     z1.b, p4/m, z1.b, offset_vec.b
    add     z2.b, p4/m, z2.b, offset_vec.b
    add     z3.b, p4/m, z3.b, offset_vec.b

    ALTER_AOFFSET
.endm



.macro TCOPY_M_TAIL_4x1VL

    ZIP4REG  z0,z1,z2,z3,z8,z9,z10,z11

    // 交织后的四个寄存器长度和 大于3x64 <4x64 ,放置4行
    st1b    {z8.b}, p1, [B00]
    st1b    {z9.b}, p2, [B00, #1, MUL VL]
    st1b    {z10.b}, p3, [B02]
    st1b    {z11.b}, p5, [B02, #1, MUL VL]

    add     B00, B00, #128
    add     B02, B02, M_MTP_tail, lsl #2

.endm

.macro TCOPY_M_TAIL_3x1VL

    ZIP4REG  z0,z1,z2,z3,z8,z9,z10,z11

    // 交织后的四个寄存器长度和 大于2x64  <= 3x64 ,放置3行
    st1b    {z8.b}, p1, [B00]
    st1b    {z9.b}, p2, [B00, #1, MUL VL]
    st1b    {z10.b}, p3, [B02]
    add     B00, B00, #128
    add     B02, B02, M_MTP_tail, lsl #2
.endm

.macro TCOPY_M_TAIL_2x1VL

    ZIP4REG  z0,z1,z2,z3,z8,z9,z10,z11

    // 交织后的四个寄存器长度和 大于64  <= 2x64 ,放置2行
    st1b    {z8.b}, p1, [B00]
    st1b    {z9.b}, p2, [B00, #1, MUL VL]
    add     B00, B00, M_MTP_tail, lsl #2
.endm

.macro TCOPY_M_TAIL_1x1VL

    ZIP4REG  z0,z1,z2,z3,z8,z9,z10,z11
    // 交织后的四个寄存器长度和 <= 64 ,放置1行
    st1b    {z8.b}, p1, [B00]

    add     B00, B00, M_MTP_tail, lsl #2
.endm


.macro TCOPY_N_M_TAIL_3x1VL
    mov     startP, #0
    whilelt p4.b, startP, M_tail

    ld1b    {z0.b}, p4/z, [A01]
    ld1b    {z1.b}, p4/z, [A02]
    ld1b    {z2.b}, p4/z, [A03]
    add     z0.b, p4/m, z0.b, offset_vec.b
    add     z1.b, p4/m, z1.b, offset_vec.b
    add     z2.b, p4/m, z2.b, offset_vec.b
    mov     z3.b, #0

    ALTER_AOFFSET
.endm



.macro TCOPY_N_M_TAIL_2x1VL
    mov     startP, #0
    whilelt p4.b, startP, M_tail

    ld1b    {z0.b}, p4/z, [A01]
    ld1b    {z1.b}, p4/z, [A02]
    add     z0.b, p4/m, z0.b, offset_vec.b
    add     z1.b, p4/m, z1.b, offset_vec.b
    mov     z2.b, #0
    mov     z3.b, #0

    ALTER_AOFFSET
.endm


.macro TCOPY_N_M_TAIL_1x1VL
    mov     startP, #0
    whilelt p4.b, startP, M_tail

    ld1b    {z0.b}, p4/z, [A01]
    add     z0.b, p4/m, z0.b, offset_vec.b
    mov     z1.b, #0
    mov     z2.b, #0
    mov     z3.b, #0

    ALTER_AOFFSET
.endm


/*******************************************************************************
 * Macro definitions end
 *******************************************************************************/

    PROLOGUE
    .align  5
    SAVE_REGS
    and     M_tail, M, #63                  // M_tail = M % 64
    and     N_tail, N, #3                   // N_tail = N % 4
    asr     N, N, #2                        // N /= 4
    asr     M, M, #6                        // M /= 64
    ptrue   p0.b, all
    mov     ELE_SIZE, #16
    dup     offset_vec.b, offset

igemm_tcopy_zip_start:
    mov     A01, A00
    add     A03, A00, LDA, lsl #1
    cmp     N_tail, #0
    cset    LDB, gt                         // n_tail > 0  --> LDB = 1，否则，LDB = 0
    add     LDB, LDB, N                     // n_tail > 0   -->>  LDB =   N / 4 + 1 ; n_tail = 0   -->>  LDB =   N / 4
    mov     J, LDB
    lsl     J, J, #2

/*******************************************************************************
 * Normal condition
 *******************************************************************************/


.Ligemm_tcopy_loop_BEGIN:
    lsl     LDB, LDB, #7                    // LDB = LDB x 4 * 32 , 最终 LDB 相当于B00 一行存储个数
    mov     Q, M
    mov     I, N
    mov     P, N
    add     TMP_VAL, M_tail, #15
    asr     M_M_tail, TMP_VAL, #4            // M_M_tail =  (M_tail + 15)  / 16
    and     M_MTP_tail, M_tail, #31         // M_MTP_tail =  M_tail  % 32
    cmp     M_tail, 32
    bne     .Ligemm_tcopy_M_loop
    beq     .Lmov32toM_MTP_tail

.Lmov32toM_MTP_tail:
    mov M_MTP_tail, #32

.Ligemm_tcopy_M_loop:
    subs    Q, Q, #1

    add     B02, B00, LDB

    cmp     Q, #0
    blt     .Ligemm_tcopy_M_TAIL_BEGIN      // 处理不规则的M     Q <  0    // Q >= 0规则往下走
    mov     S, I
    cmp     S, #0
    beq     .Ligemm_tcopy_N_TAIL_BEGIN


.Ligemm_tcopy_N_loop:
    TCOPY4x1VL
    subs    S, S, #1
    bne     .Ligemm_tcopy_N_loop                // 处理规则的N  I !=0 I> 0
    beq     .Ligemm_tcopy_N_TAIL_BEGIN          // 处理不规则的N   I = 0


/*******************************************************************************
 * Boundary condition
 *******************************************************************************/
//*****************************不规则的N ***********************************************/
.Ligemm_tcopy_N_TAIL_BEGIN:                   // N_tail  (0,1,2,3)
    cmp     N_tail, #0
    beq     .Ligemm_tcopy_N_TAIL_END
    cmp     N_tail, #2
    blt     .Ligemm_tcopy_1VL_N_TAIL
    beq     .Ligemm_tcopy_2VL_N_TAIL
    bgt     .Ligemm_tcopy_3VL_N_TAIL

.Ligemm_tcopy_3VL_N_TAIL:
    TCOPY_N_TAIL_3x1VL
    b       .Ligemm_tcopy_N_TAIL_END

.Ligemm_tcopy_2VL_N_TAIL:
    TCOPY_N_TAIL_2x1VL
    b       .Ligemm_tcopy_N_TAIL_END

.Ligemm_tcopy_1VL_N_TAIL:
    TCOPY_N_TAIL_1x1VL

.Ligemm_tcopy_N_TAIL_END:
    mov     K, J
    //add     A01, A01, #64
    //add     A03, A03, #64
    ALTER_BTAIL_OFFSET                           // 当加载至A矩阵最后一行时，B00矩阵需换4行存储
    add     A00, A00, #64
    mov     A01, A00
    add     A03, A00, LDA, lsl #1
    b     .Ligemm_tcopy_M_loop
    //b       .Ligemm_tcopy_alter_Aoffset


// 更新A00 的偏移量，移至行首
.Ligemm_tcopy_alter_Aoffset:
    subs    A01, A01, LDA
    subs    A03, A03, LDA
    subs    K, K, #1
    bne     .Ligemm_tcopy_alter_Aoffset      // K > 0
    beq     .Ligemm_tcopy_M_loop            // K = 0

//****************************不规则的M ******************************************************/
.macro UPDATE_PG_BY_CNTR reg_pg,start,end,step
	whilelt \reg_pg\().s, \start,\end
    cntp    TMP_CNT, \reg_pg, \reg_pg\().s
    LSL     TMP_CNT, TMP_CNT, #2
    whilelt \reg_pg\().b, XZR, TMP_CNT
	add \start,\start,\step
.endm

.Ligemm_tcopy_M_TAIL_BEGIN:
    cmp     M_tail, #0
    beq     .Ligemm_tcopy_L999

    mov     TMP_VAL, #0

    UPDATE_PG_BY_CNTR p1, TMP_VAL, M_tail, ELE_SIZE
    UPDATE_PG_BY_CNTR p2, TMP_VAL, M_tail, ELE_SIZE
    UPDATE_PG_BY_CNTR p3, TMP_VAL, M_tail, ELE_SIZE
    UPDATE_PG_BY_CNTR p5, TMP_VAL, M_tail, ELE_SIZE

    // 处理最后一个不规则的M-N
    cmp     I, #0
    beq     .Ligemm_tcopy_N_M_TAIL_BEGIN


    TCOPY_M_TAIL_BEGIN

    //(0,1,2,3,4 四种结果,比如 M_tail = 61 --->  M_M_tail = 4,拆分为4行;M_tail = 15 --->  M_M_tail = 1,拆分为1行),自动尾部放置
    cmp     M_M_tail, #0
    beq     .Ligemm_tcopy_M_M_TAIL_END
    cmp     M_M_tail, #1
    beq     .Ligemm_tcopy_1VL_M_TAIL
    cmp     M_M_tail, #3
    blt     .Ligemm_tcopy_2VL_M_TAIL
    beq     .Ligemm_tcopy_3VL_M_TAIL
    bgt     .Ligemm_tcopy_4VL_M_TAIL


.Ligemm_tcopy_4VL_M_TAIL:
    TCOPY_M_TAIL_4x1VL
    b       .Ligemm_tcopy_M_M_TAIL_END

.Ligemm_tcopy_3VL_M_TAIL:
    TCOPY_M_TAIL_3x1VL
    b       .Ligemm_tcopy_M_M_TAIL_END

.Ligemm_tcopy_2VL_M_TAIL:
    TCOPY_M_TAIL_2x1VL
    b       .Ligemm_tcopy_M_M_TAIL_END

.Ligemm_tcopy_1VL_M_TAIL:
    TCOPY_M_TAIL_1x1VL


.Ligemm_tcopy_M_M_TAIL_END:
    subs    P, P, #1
    cmp     P, #0
    blt     .Ligemm_tcopy_L999              //  结束     P <  0
    bgt     .Ligemm_tcopy_M_TAIL_BEGIN      // P > 0  处理下四行 M


//**********************P=0时，处理最后一个不规则的M-N *************************************************************/
.Ligemm_tcopy_N_M_TAIL_BEGIN:
    cmp     N_tail, #0
    beq     .Ligemm_tcopy_L999
    cmp     N_tail, #2
    blt     .Ligemm_tcopy_1VL_N_M_TAIL
    beq     .Ligemm_tcopy_2VL_N_M_TAIL
    bgt     .Ligemm_tcopy_3VL_N_M_TAIL

//先从A00 ld1b 数据
.Ligemm_tcopy_3VL_N_M_TAIL:
    TCOPY_N_M_TAIL_3x1VL
    b       .Ligemm_tcopy_N_M_TAIL_END

.Ligemm_tcopy_2VL_N_M_TAIL:
    TCOPY_N_M_TAIL_2x1VL
    b       .Ligemm_tcopy_N_M_TAIL_END

.Ligemm_tcopy_1VL_N_M_TAIL:
    TCOPY_N_M_TAIL_1x1VL

//再st1b 数据到B00
.Ligemm_tcopy_N_M_TAIL_END:
    cmp     M_M_tail, #0
    beq       .Ligemm_tcopy_L999
    cmp     M_M_tail, #1
    beq     .Ligemm_tcopy_N_M_1VL_TAIL
    cmp     M_M_tail, #3
    blt     .Ligemm_tcopy_N_M_2VL_TAIL
    beq     .Ligemm_tcopy_N_M_3VL_TAIL
    bgt     .Ligemm_tcopy_N_M_4VL_TAIL

.Ligemm_tcopy_N_M_4VL_TAIL:
    TCOPY_M_TAIL_4x1VL
    b       .Ligemm_tcopy_L999

.Ligemm_tcopy_N_M_3VL_TAIL:
    TCOPY_M_TAIL_3x1VL
    b       .Ligemm_tcopy_L999

.Ligemm_tcopy_N_M_2VL_TAIL:
    TCOPY_M_TAIL_2x1VL
    b       .Ligemm_tcopy_L999

.Ligemm_tcopy_N_M_1VL_TAIL:
    TCOPY_M_TAIL_1x1VL


.Ligemm_tcopy_L999:
    mov     x0, #0                        // set return value
    RESTORE_REGS
    ret

    EPILOGUE


