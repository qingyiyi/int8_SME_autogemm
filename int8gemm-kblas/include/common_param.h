/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: common parameter header
 * Author: KPL
 * Create: 2020-11-05
 * Notes: NA
 */

#ifndef COMMON_PARAM_H
#define COMMON_PARAM_H

#include "include/param.h"

#ifndef ASSEMBLER

#define GEMM_OFFSET_A GEMM_DEFAULT_OFFSET_A
#define GEMM_OFFSET_B GEMM_DEFAULT_OFFSET_B
#define GEMM_ALIGN GEMM_DEFAULT_ALIGN
#define DTB_ENTRIES DTB_DEFAULT_ENTRIES

#ifndef HAVE_EXCLUSIVE_CACHE
#define HAVE_EX_L2 0
#else
#define HAVE_EX_L2 1
#endif

#define SGEMM_UNROLL_M SGEMM_DEFAULT_UNROLL_M
#define SGEMM_UNROLL_N SGEMM_DEFAULT_UNROLL_N
#ifndef SGEMM_DEFAULT_UNROLL_MN
#if defined(PARAM_CONF_KP920B)
#define SGEMM_UNROLL_MN (SGEMM_UNROLL_M == 16 ? 48 : MAX((SGEMM_UNROLL_M), (SGEMM_UNROLL_N)))
#elif defined(PARAM_CONF_KP920F_SVE)
#define SGEMM_UNROLL_MN (SGEMM_UNROLL_M == 32 ? 96 : MAX((SGEMM_UNROLL_M), (SGEMM_UNROLL_N)))
#else
#define SGEMM_UNROLL_MN (SGEMM_UNROLL_M == 12 ? 24 : MAX((SGEMM_UNROLL_M), (SGEMM_UNROLL_N)))
#endif
#else
#define SGEMM_UNROLL_MN SGEMM_DEFAULT_UNROLL_MN
#endif
#define SGEMM_P SGEMM_DEFAULT_P
#define SGEMM_Q SGEMM_DEFAULT_Q
#define SGEMM_R SGEMM_DEFAULT_R

#define DGEMM_UNROLL_M DGEMM_DEFAULT_UNROLL_M
#define DGEMM_UNROLL_N DGEMM_DEFAULT_UNROLL_N
#ifndef DGEMM_DEFAULT_UNROLL_MN
#if defined(PARAM_CONF_KP920B) || defined(PARAM_CONF_KP920F_SVE)
#define DGEMM_UNROLL_MN (DGEMM_UNROLL_N == 8 ? 24 : MAX((DGEMM_UNROLL_M), (DGEMM_UNROLL_N)))
#else
#define DGEMM_UNROLL_MN (DGEMM_UNROLL_N == 6 ? 24 : MAX((DGEMM_UNROLL_M), (DGEMM_UNROLL_N)))
#endif
#else
#define DGEMM_UNROLL_MN DGEMM_DEFAULT_UNROLL_MN
#endif
#define DGEMM_P DGEMM_DEFAULT_P
#define DGEMM_Q DGEMM_DEFAULT_Q
#define DGEMM_R DGEMM_DEFAULT_R

#define CGEMM_UNROLL_M CGEMM_DEFAULT_UNROLL_M
#define CGEMM_UNROLL_N CGEMM_DEFAULT_UNROLL_N
#define CGEMM_P CGEMM_DEFAULT_P
#define CGEMM_Q CGEMM_DEFAULT_Q
#define CGEMM_R CGEMM_DEFAULT_R

#if defined(CGEMM_DEFAULT_UNROLL_MN)
#define CGEMM_UNROLL_MN CGEMM_DEFAULT_UNROLL_MN
#else
#define CGEMM_UNROLL_MN MAX((CGEMM_UNROLL_M), (CGEMM_UNROLL_N))
#endif

#define ZGEMM_UNROLL_M ZGEMM_DEFAULT_UNROLL_M
#define ZGEMM_UNROLL_N ZGEMM_DEFAULT_UNROLL_N
#define ZGEMM_P ZGEMM_DEFAULT_P
#define ZGEMM_Q ZGEMM_DEFAULT_Q
#define ZGEMM_R ZGEMM_DEFAULT_R

#ifdef ZGEMM_DEFAULT_UNROLL_MN
#define ZGEMM_UNROLL_MN ZGEMM_DEFAULT_UNROLL_MN
#else
#define ZGEMM_UNROLL_MN MAX((ZGEMM_UNROLL_M), (ZGEMM_UNROLL_N))
#endif

#if defined(CGEMM3M_DEFAULT_UNROLL_N)
#define CGEMM3M_UNROLL_M CGEMM3M_DEFAULT_UNROLL_M
#define CGEMM3M_UNROLL_N CGEMM3M_DEFAULT_UNROLL_N
#define CGEMM3M_UNROLL_MN MAX((CGEMM3M_UNROLL_M), (CGEMM3M_UNROLL_N))
#define CGEMM3M_P CGEMM3M_DEFAULT_P
#define CGEMM3M_Q CGEMM3M_DEFAULT_Q
#define CGEMM3M_R CGEMM3M_DEFAULT_R
#else
#define CGEMM3M_UNROLL_M SGEMM_DEFAULT_UNROLL_M
#define CGEMM3M_UNROLL_N SGEMM_DEFAULT_UNROLL_N
#define CGEMM3M_UNROLL_MN MAX((CGEMM_UNROLL_M), (CGEMM_UNROLL_N))
#define CGEMM3M_P SGEMM_DEFAULT_P
#define CGEMM3M_Q SGEMM_DEFAULT_Q
#define CGEMM3M_R SGEMM_DEFAULT_R
#endif

#ifdef ZGEMM3M_DEFAULT_UNROLL_N
#define ZGEMM3M_UNROLL_M ZGEMM3M_DEFAULT_UNROLL_M
#define ZGEMM3M_UNROLL_N ZGEMM3M_DEFAULT_UNROLL_N
#define ZGEMM3M_UNROLL_MN MAX((ZGEMM_UNROLL_M), (ZGEMM_UNROLL_N))
#define ZGEMM3M_P ZGEMM3M_DEFAULT_P
#define ZGEMM3M_Q ZGEMM3M_DEFAULT_Q
#define ZGEMM3M_R ZGEMM3M_DEFAULT_R
#else
#define ZGEMM3M_UNROLL_M DGEMM_DEFAULT_UNROLL_M
#define ZGEMM3M_UNROLL_N DGEMM_DEFAULT_UNROLL_N
#define ZGEMM3M_UNROLL_MN MAX((ZGEMM_UNROLL_M), (ZGEMM_UNROLL_N))
#define ZGEMM3M_P DGEMM_DEFAULT_P
#define ZGEMM3M_Q DGEMM_DEFAULT_Q
#define ZGEMM3M_R DGEMM_DEFAULT_R
#endif
#endif

#if !(defined(HGEMM) || defined(BGEMM))
#ifndef COMPLEX
#if defined(DOUBLE)
#define GEMM_UNROLL_M DGEMM_UNROLL_M
#define GEMM_UNROLL_N DGEMM_UNROLL_N
#define GEMM_UNROLL_MN DGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M DGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N DGEMM_DEFAULT_UNROLL_N
#define GEMM_P DGEMM_P
#define GEMM_Q DGEMM_Q
#define GEMM_R DGEMM_R
#define GEMM_DEFAULT_P DGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q DGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R DGEMM_DEFAULT_R
#elif defined(INT8_SME)
// INT8_SME除PQR外，其余值使用SGEMM的默认值
#define GEMM_UNROLL_M SGEMM_UNROLL_M
#define GEMM_UNROLL_N SGEMM_UNROLL_N
#define GEMM_UNROLL_MN SGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M SGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N SGEMM_DEFAULT_UNROLL_N
#define GEMM_P 192
#define GEMM_Q 1380
#define GEMM_R 8192
#define GEMM_DEFAULT_P SGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q SGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R SGEMM_DEFAULT_R
#else
#define GEMM_UNROLL_M SGEMM_UNROLL_M
#define GEMM_UNROLL_N SGEMM_UNROLL_N
#define GEMM_UNROLL_MN SGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M SGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N SGEMM_DEFAULT_UNROLL_N
#define GEMM_P SGEMM_P
#define GEMM_Q SGEMM_Q
#define GEMM_R SGEMM_R
#define GEMM_DEFAULT_P SGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q SGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R SGEMM_DEFAULT_R
#endif
#else
#if defined(DOUBLE)
#if defined(GEMM3M)
#define GEMM_UNROLL_M ZGEMM3M_UNROLL_M
#define GEMM_UNROLL_N ZGEMM3M_UNROLL_N
#define GEMM_UNROLL_MN ZGEMM3M_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M ZGEMM3M_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N ZGEMM3M_DEFAULT_UNROLL_N
#define GEMM_P ZGEMM3M_P
#define GEMM_Q ZGEMM3M_Q
#define GEMM_R ZGEMM3M_R
#define GEMM_DEFAULT_P ZGEMM3M_DEFAULT_P
#define GEMM_DEFAULT_Q ZGEMM3M_DEFAULT_Q
#define GEMM_DEFAULT_R ZGEMM3M_DEFAULT_R
#else
#define GEMM_UNROLL_M ZGEMM_UNROLL_M
#define GEMM_UNROLL_N ZGEMM_UNROLL_N
#define GEMM_UNROLL_MN ZGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M ZGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N ZGEMM_DEFAULT_UNROLL_N
#define GEMM_P ZGEMM_P
#define GEMM_Q ZGEMM_Q
#define GEMM_R ZGEMM_R
#define GEMM_DEFAULT_P ZGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q ZGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R ZGEMM_DEFAULT_R
#endif
#else
#if defined(GEMM3M)
#define GEMM_UNROLL_M CGEMM3M_UNROLL_M
#define GEMM_UNROLL_N CGEMM3M_UNROLL_N
#define GEMM_UNROLL_MN CGEMM3M_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M CGEMM3M_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N CGEMM3M_DEFAULT_UNROLL_N
#define GEMM_P CGEMM3M_P
#define GEMM_Q CGEMM3M_Q
#define GEMM_R CGEMM3M_R
#define GEMM_DEFAULT_P CGEMM3M_DEFAULT_P
#define GEMM_DEFAULT_Q CGEMM3M_DEFAULT_Q
#define GEMM_DEFAULT_R CGEMM3M_DEFAULT_R
#else
#define GEMM_UNROLL_M CGEMM_UNROLL_M
#define GEMM_UNROLL_N CGEMM_UNROLL_N
#define GEMM_UNROLL_MN CGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M CGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N CGEMM_DEFAULT_UNROLL_N
#define GEMM_P CGEMM_P
#define GEMM_Q CGEMM_Q
#define GEMM_R CGEMM_R
#define GEMM_DEFAULT_P CGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q CGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R CGEMM_DEFAULT_R
#endif /* GEMM3M */
#endif /* DOUBLE */
#endif /* COMPLEX */
#else
#ifdef HGEMM
#ifndef COMPLEX
#define GEMM_UNROLL_M HGEMM_UNROLL_M
#define GEMM_UNROLL_N HGEMM_UNROLL_N
#define GEMM_UNROLL_MN HGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M HGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N HGEMM_DEFAULT_UNROLL_N
#define GEMM_DEFAULT_P HGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q HGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R HGEMM_DEFAULT_R
#define GEMM_P HGEMM_DEFAULT_P
#define GEMM_Q HGEMM_DEFAULT_Q
#define GEMM_R HGEMM_DEFAULT_R
#else
#define GEMM_UNROLL_M CHGEMM_UNROLL_M
#define GEMM_UNROLL_N CHGEMM_UNROLL_N
#define GEMM_UNROLL_MN CHGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M CHGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N CHGEMM_DEFAULT_UNROLL_N
#define GEMM_DEFAULT_P CHGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q CHGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R CHGEMM_DEFAULT_R
#define GEMM_P CHGEMM_DEFAULT_P
#define GEMM_Q CHGEMM_DEFAULT_Q
#define GEMM_R CHGEMM_DEFAULT_R
#endif
#else
#ifndef COMPLEX
#define GEMM_UNROLL_M BGEMM_UNROLL_M
#define GEMM_UNROLL_N BGEMM_UNROLL_N
#define GEMM_UNROLL_MN BGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M BGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N BGEMM_DEFAULT_UNROLL_N
#define GEMM_DEFAULT_P BGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q BGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R BGEMM_DEFAULT_R
#define GEMM_P BGEMM_DEFAULT_P
#define GEMM_Q BGEMM_DEFAULT_Q
#define GEMM_R BGEMM_DEFAULT_R
#else
#define GEMM_UNROLL_M CBGEMM_UNROLL_M
#define GEMM_UNROLL_N CBGEMM_UNROLL_N
#define GEMM_UNROLL_MN CBGEMM_UNROLL_MN
#define GEMM_DEFAULT_UNROLL_M CBGEMM_DEFAULT_UNROLL_M
#define GEMM_DEFAULT_UNROLL_N CBGEMM_DEFAULT_UNROLL_N
#define GEMM_DEFAULT_P CBGEMM_DEFAULT_P
#define GEMM_DEFAULT_Q CBGEMM_DEFAULT_Q
#define GEMM_DEFAULT_R CBGEMM_DEFAULT_R
#define GEMM_P CBGEMM_DEFAULT_P
#define GEMM_Q CBGEMM_DEFAULT_Q
#define GEMM_R CBGEMM_DEFAULT_R
#endif
#endif
#endif

#ifndef DOUBLE
#define GEMM3M_UNROLL_M CGEMM3M_UNROLL_M
#define GEMM3M_UNROLL_N CGEMM3M_UNROLL_N
#else
#define GEMM3M_UNROLL_M ZGEMM3M_UNROLL_M
#define GEMM3M_UNROLL_N ZGEMM3M_UNROLL_N
#endif
// Need to be multiple in syr2k,syrk, but gemm3m is irrelevant
#ifndef GEMM3M
#if ((GEMM_P * GEMM_R * GEMM_UNROLL_M * GEMM_UNROLL_N) != 0)
#if ((GEMM_P % (GEMM_UNROLL_M)) != 0) || ((GEMM_R % (GEMM_UNROLL_M)) != 0)
#error "GEMM_P,GEMM_R must be multiple of GEMM_UNROLL_M"
#endif
#if ((GEMM_P % (GEMM_UNROLL_N)) != 0) || ((GEMM_R % (GEMM_UNROLL_N)) != 0)
#error "GEMM_P,GEMM_R must be multiple of GEMM_UNROLL_N"
#endif
#endif
#endif // GEMM3M

#if !defined(GEMM_THREAD)
#define GEMM_THREAD gemm_thread_n
#endif

#ifndef SGEMM_DEFAULT_R
#define SGEMM_DEFAULT_R                                                                                      \
    (((BUFFER_SIZE - ((SGEMM_DEFAULT_P * SGEMM_DEFAULT_Q * 4 + GEMM_DEFAULT_OFFSET_A + GEMM_DEFAULT_ALIGN) & \
        ~GEMM_DEFAULT_ALIGN)) /                                                                              \
        (SGEMM_DEFAULT_Q * 4) -                                                                              \
        15) &                                                                                                \
        ~15)
#endif

#ifndef DGEMM_DEFAULT_R
#define DGEMM_DEFAULT_R                                                                                      \
    (((BUFFER_SIZE - ((DGEMM_DEFAULT_P * DGEMM_DEFAULT_Q * 8 + GEMM_DEFAULT_OFFSET_A + GEMM_DEFAULT_ALIGN) & \
        ~GEMM_DEFAULT_ALIGN)) /                                                                              \
        (DGEMM_DEFAULT_Q * 8) -                                                                              \
        15) &                                                                                                \
        ~15)
#endif

#ifndef CGEMM_DEFAULT_R
#define CGEMM_DEFAULT_R                                                                                      \
    (((BUFFER_SIZE - ((CGEMM_DEFAULT_P * CGEMM_DEFAULT_Q * 8 + GEMM_DEFAULT_OFFSET_A + GEMM_DEFAULT_ALIGN) & \
        ~GEMM_DEFAULT_ALIGN)) /                                                                              \
        (CGEMM_DEFAULT_Q * 8) -                                                                              \
        15) &                                                                                                \
        ~15)
#endif

#ifndef ZGEMM_DEFAULT_R
#define ZGEMM_DEFAULT_R                                                                                       \
    (((BUFFER_SIZE - ((ZGEMM_DEFAULT_P * ZGEMM_DEFAULT_Q * 16 + GEMM_DEFAULT_OFFSET_A + GEMM_DEFAULT_ALIGN) & \
        ~GEMM_DEFAULT_ALIGN)) /                                                                               \
        (ZGEMM_DEFAULT_Q * 16) -                                                                              \
        15) &                                                                                                 \
        ~15)
#endif

#ifndef DNUMOPT
#define DNUMOPT 2
#endif

#ifndef SNUMOPT
#define SNUMOPT 2
#endif

#ifndef QNUMOPT
#define QNUMOPT 1
#endif

#ifndef GEMM3M_P
#ifndef DOUBLE
#define GEMM3M_P CGEMM3M_P
#else
#define GEMM3M_P ZGEMM3M_P
#endif
#endif

#ifndef GEMM3M_Q
#ifndef DOUBLE
#define GEMM3M_Q CGEMM3M_Q
#else
#define GEMM3M_Q ZGEMM3M_Q
#endif
#endif

#ifndef GEMM3M_R
#ifndef DOUBLE
#define GEMM3M_R CGEMM3M_R
#else
#define GEMM3M_R ZGEMM3M_R
#endif
#endif

#define UNROLL_MASK (GEMM_UNROLL_MN - 1)

#define GENERATE_PQR_VARS_FOR_GEMM(precision)             \
    BLAS_API_LOCAL BLASLONG g_##precision##gemm_single_p, \
                            g_##precision##gemm_single_q, \
                            g_##precision##gemm_single_r, \
                            g_##precision##gemm_multi_p,  \
                            g_##precision##gemm_multi_q,  \
                            g_##precision##gemm_multi_r

#define DECLARE_PQR_VARS(precision)               \
    extern BLASLONG g_##precision##gemm_single_p, \
                    g_##precision##gemm_single_q, \
                    g_##precision##gemm_single_r, \
                    g_##precision##gemm_multi_p,  \
                    g_##precision##gemm_multi_q,  \
                    g_##precision##gemm_multi_r

#ifndef ASSEMBLER
typedef struct {
    bool sgemmPEnvFlag, sgemmQEnvFlag, sgemmREnvFlag;
    bool dgemmPEnvFlag, dgemmQEnvFlag, dgemmREnvFlag;
    bool cgemmPEnvFlag, cgemmQEnvFlag, cgemmREnvFlag;
    bool zgemmPEnvFlag, zgemmQEnvFlag, zgemmREnvFlag;
} PqrFlags;

extern PqrFlags g_pqrFlag;

DECLARE_PQR_VARS(s);
DECLARE_PQR_VARS(d);
DECLARE_PQR_VARS(c);
DECLARE_PQR_VARS(z);
DECLARE_PQR_VARS(h);
DECLARE_PQR_VARS(b);
DECLARE_PQR_VARS(ch);
DECLARE_PQR_VARS(cb);
#endif

#if !(defined(HGEMM) || defined(BGEMM))
#ifndef COMPLEX
#if defined(DOUBLE)
#define GEMM_SINGLE_P g_dgemm_single_p
#define GEMM_SINGLE_Q g_dgemm_single_q
#define GEMM_SINGLE_R g_dgemm_single_r
#define GEMM_MULTI_P g_dgemm_multi_p
#define GEMM_MULTI_Q g_dgemm_multi_q
#define GEMM_MULTI_R g_dgemm_multi_r
#define P_FLAG g_pqrFlag.dgemmPEnvFlag
#define Q_FLAG g_pqrFlag.dgemmQEnvFlag
#define R_FLAG g_pqrFlag.dgemmREnvFlag

#else
#define GEMM_SINGLE_P g_sgemm_single_p
#define GEMM_SINGLE_Q g_sgemm_single_q
#define GEMM_SINGLE_R g_sgemm_single_r
#define GEMM_MULTI_P g_sgemm_multi_p
#define GEMM_MULTI_Q g_sgemm_multi_q
#define GEMM_MULTI_R g_sgemm_multi_r
#define P_FLAG g_pqrFlag.sgemmPEnvFlag
#define Q_FLAG g_pqrFlag.sgemmQEnvFlag
#define R_FLAG g_pqrFlag.sgemmREnvFlag
#endif

#else
#if defined(DOUBLE)
#define GEMM_SINGLE_P g_zgemm_single_p
#define GEMM_SINGLE_Q g_zgemm_single_q
#define GEMM_SINGLE_R g_zgemm_single_r
#define GEMM_MULTI_P g_zgemm_multi_p
#define GEMM_MULTI_Q g_zgemm_multi_q
#define GEMM_MULTI_R g_zgemm_multi_r
#define P_FLAG g_pqrFlag.zgemmPEnvFlag
#define Q_FLAG g_pqrFlag.zgemmQEnvFlag
#define R_FLAG g_pqrFlag.zgemmREnvFlag

#else
#define GEMM_SINGLE_P g_cgemm_single_p
#define GEMM_SINGLE_Q g_cgemm_single_q
#define GEMM_SINGLE_R g_cgemm_single_r
#define GEMM_MULTI_P g_cgemm_multi_p
#define GEMM_MULTI_Q g_cgemm_multi_q
#define GEMM_MULTI_R g_cgemm_multi_r
#define P_FLAG g_pqrFlag.cgemmPEnvFlag
#define Q_FLAG g_pqrFlag.cgemmQEnvFlag
#define R_FLAG g_pqrFlag.cgemmREnvFlag
#endif /* DOUBLE */
#endif /* COMPLEX */

#else
#ifdef HGEMM
#ifndef COMPLEX
#define GEMM_SINGLE_P g_hgemm_single_p
#define GEMM_SINGLE_Q g_hgemm_single_q
#define GEMM_SINGLE_R g_hgemm_single_r
#define GEMM_MULTI_P g_hgemm_multi_p
#define GEMM_MULTI_Q g_hgemm_multi_q
#define GEMM_MULTI_R g_hgemm_multi_r

#else
#define GEMM_SINGLE_P g_chgemm_single_p
#define GEMM_SINGLE_Q g_chgemm_single_q
#define GEMM_SINGLE_R g_chgemm_single_r
#define GEMM_MULTI_P g_chgemm_multi_p
#define GEMM_MULTI_Q g_chgemm_multi_q
#define GEMM_MULTI_R g_chgemm_multi_r
#endif

#else
#ifndef COMPLEX
#define GEMM_SINGLE_P g_bgemm_single_p
#define GEMM_SINGLE_Q g_bgemm_single_q
#define GEMM_SINGLE_R g_bgemm_single_r
#define GEMM_MULTI_P g_bgemm_multi_p
#define GEMM_MULTI_Q g_bgemm_multi_q
#define GEMM_MULTI_R g_bgemm_multi_r

#else
#define GEMM_SINGLE_P g_cbgemm_single_p
#define GEMM_SINGLE_Q g_cbgemm_single_q
#define GEMM_SINGLE_R g_cgemm_single_r
#define GEMM_MULTI_P g_cbgemm_multi_p
#define GEMM_MULTI_Q g_cbgemm_multi_q
#define GEMM_MULTI_R g_cbgemm_multi_r
#endif
#endif
#endif

#ifdef SMP
#define GLOBAL_GEMM_P GEMM_MULTI_P
#define GLOBAL_GEMM_Q GEMM_MULTI_Q
#define GLOBAL_GEMM_R GEMM_MULTI_R
#else
#define GLOBAL_GEMM_P GEMM_SINGLE_P
#define GLOBAL_GEMM_Q GEMM_SINGLE_Q
#define GLOBAL_GEMM_R GEMM_SINGLE_R
#endif

#endif