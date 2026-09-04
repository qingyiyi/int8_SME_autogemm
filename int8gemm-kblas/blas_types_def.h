/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: multithread variables and routines
 * Author: KPL
 * Create: 2022-04-21
 * Notes: NA
 */
#ifndef BLAS_TYPES_DEF_H
#define BLAS_TYPES_DEF_H

#ifndef BLAS_TYPES_DEF
#define BLAS_TYPES_DEF

#ifndef ASSEMBLER
#include <stdint.h>
#include <stddef.h>
typedef int8_t BLASINT8;

typedef long BLASLONG;
typedef unsigned long BLASULONG;
#ifdef USE64BITINT
typedef BLASLONG  BLASINT;
#define blasabs(x) labs(x)
#else
typedef int BLASINT;
#define blasabs(x) abs(x)
#endif // USE64BITINT

/* C99 supports complex floating numbers natively, which GCC also offers as an
   extension since version 3.0.  If neither are available, use a compatible
   structure as fallback (see Clause 6.2.5.13 of the C99 standard). */
#if ((defined(__STDC_IEC_559_COMPLEX__) || __STDC_VERSION__ >= 199901L || \
    (__GNUC__ >= 3 && !defined(__cplusplus))) && \
    !(defined(FORCE_BLAS_COMPLEX_STRUCT)))
#define BLAS_COMPLEX_C99
#ifndef __cplusplus
#include <complex.h>
#endif
typedef float _Complex blas_complex_float;
typedef double _Complex blas_complex_double;
#define blas_make_complex_float(real, imag) ((real) + ((imag) * _Complex_I))
#define blas_make_complex_double(real, imag) ((real) + ((imag) * _Complex_I))
#else
#error("Error:required:__STDC_IEC_559_COMPLEX__ || __STDC_VERSION__ >= 199901L || __GNUC__ >= 3")
#endif
#endif // ASSEMBLER
#endif /* BLAS_TYPES_DEF */

#ifdef HGEMM
#ifdef MATRIXC32
#define FLOAT __fp16
#define FLOATC float
#else
#define FLOAT __fp16
#define FLOATC __fp16
#endif
#define SIZE 2
#elif defined(BGEMM)
#ifdef MATRIXC32
#define FLOAT __bf16
#define FLOATC float
#else
#define FLOAT __bf16
#define FLOATC __bf16
#endif
#define SIZE 2
#else
#ifndef DOUBLE
#ifdef HALF
// half precision
#define FLOAT __fp16
#define BASE_SHIFT 1
#define ZBASE_SHIFT 2
#define SIZE 3
#else
// float precision
#define FLOAT float
#define BASE_SHIFT 2
#define ZBASE_SHIFT 3
#define SIZE 4
#endif // HALF
#else
#define FLOAT double
#define BASE_SHIFT 3
#define ZBASE_SHIFT 4
#define SIZE 8
#endif
#define FLOATC FLOAT
#endif

#ifdef COMPLEX
#define COMPSIZE 2
#else
#define COMPSIZE 1
#endif

#ifdef INT8_SME
#undef FLOAT
#undef FLOATC
#define FLOAT BLASINT8
#define FLOATC  float
#endif

#ifndef BLAS_API_LOCAL
#define BLAS_API_LOCAL __attribute__((visibility("hidden")))
#endif

#endif /* BLAS_TYPES_DEF_H */