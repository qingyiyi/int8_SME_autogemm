/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: gemm sme registers definitions
 * Author: KPL
 * Create: 2022-09-22
 * Notes: NA
 */
#ifndef GEMM_SME_REG_DEFS
    #define GEMM_SME_REG_DEFS
    #ifndef KERNEL_PARAMS
        #define KERNEL_PARAMS
        #define origM          x0
        #define origN          x1
        #define origK          x2
        #define origPA         x3
        #ifndef PACKING
            #define LDA            x4
            #define origPB         x5
            #define LDB            x6
            #define pC             x7
            #define LDC            x8
            #define pB             x5
            #define pA_OFFSET      x5
            #define pA1            x26  // int8 sme kernel just need pA1, do not need pA2 pA3 (pAn)
        #else
            #define origPB         x4
            #define pC             x5
            #define LDC            x6
            #define pB             x4
        #endif
    #endif

#define MIN_M          x9
#define MIN_N          x10

#define counterL       x14
#define counterI       x18
#define counterJ       x16

#define pA0            x24
#define pAt            x29
#define pB0            x19
#define pB1            x15
#define pBt            x11
#define ptrcp          x17
#define ptrbp          x25
#define wbk            x23
#define offset_pfc     x28

#define pC0            x20
#define pC1            x21
#define pC2            x22
#define pC3            x23

#define TMP_OFF        x28
#define TMP_VL         x5
#define TMP_VL_SIN     w5
#define TMP_CNT_POST   x23
#define TMP_PTR        x17
#define TMP_PTR_SIN    w17
#define TMP_PTR1       x21
#define ELE_SIZE       x27
#define StepCNT        x26
#define PRC_CNT        x22
#define TMP_CNT        x14
#define TMP_CNT_SIN    w14
#define MSTEP3VL       #24
#define MSTEP2VL       #16

#define Pg_MOPA_0    p4
#define Pg_MOPA_1    p5
#define Pg_MOPA_2    p6
#define Pg_MOPA_3    p7


#define PRECISION      b
#define LD1            ld1b
#define LDNT1          ldnt1b
#define ST1            st1w
#define STNT1          stnt1w
#define alpha          w26
#define alpha0         z30.s
#define beta           w26
#define beta0          z31.s
#define ELENUM         #15


#define DIMENSION      0
#define PC_IMM         0
#define SAVE_CNT_STEP  1


#define PRECI_WIDTH    2
#define PRFC_NUM       #16

#endif /* GEMM_SME_REG_DEFS */