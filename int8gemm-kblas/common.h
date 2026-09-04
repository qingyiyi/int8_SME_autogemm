/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: reference macro
 * Author: KML
 * Create: 2020-11-05
 * Notes: NA
 */

#ifndef COMMON_H
#define COMMON_H

#if !defined(__USE_XOPEN)
#define __USE_XOPEN
#endif

#if defined(SMP_SERVER) || defined(SMP_ONDEMAND)
#define SMP
#endif

#if !defined(ASSEMBLER) && !defined(NOINCLUDE)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <stdbool.h>

#ifdef OS_LINUX
#include <malloc.h>
#include <sched.h>
#endif

#ifdef NUMA_AFFINITY
#if defined(PARAM_CONF_KP920F_SVE) || defined(PARAM_CONF_KP920F_SME)
#define MAX_NODES 16
#else
#define MAX_NODES 4
#endif
#endif

#include <sys/mman.h>
#ifndef NO_SYSV_IPC
#include <sys/shm.h>
#endif
#include <sys/time.h>
#include <math.h>
#if defined(SMP) || defined(USE_LOCKING)
#include <pthread.h>
#endif
#endif

#include "config.h"
#include "blas_types_def.h"

#undef DEBUG_INFO
#undef MALLOC_DEBUG
#undef SMP_ALLOC_DEBUG
#ifndef ZERO
#ifndef DOUBLE
#define ZERO 0.e0f
#else
#define ZERO 0.e0
#endif
#endif

#ifndef ONE
#ifndef DOUBLE
#define ONE 1.e0f
#else
#define ONE 1.e0
#endif
#endif

#ifndef MINUS_ONE
#ifndef DOUBLE
#define MINUS_ONE (-1.e0f)
#else
#define MINUS_ONE (-1.e0)
#endif
#endif

#ifndef FLOAT_IS_ZERO
#if defined(BGEMM) && !defined(MATRIXC32)
#define FLOAT_IS_ZERO(x) (vcvtah_f32_bf16(x) == ZERO)
#else
#define FLOAT_IS_ZERO(x) ((x) == ZERO)
#endif
#endif

#ifndef FLOAT_IS_ONE
#if defined(BGEMM) && !defined(MATRIXC32)
#define FLOAT_IS_ONE(x) (vcvtah_f32_bf16(x) == ONE)
#else
#define FLOAT_IS_ONE(x) ((x) == ONE)
#endif
#endif

#ifndef FLOAT_IS_MINUS_ONE
#if defined(BGEMM) && !defined(MATRIXC32)
#define FLOAT_IS_MINUS_ONE(x) (vcvtah_f32_bf16(x) == MINUS_ONE)
#else
#define FLOAT_IS_MINUS_ONE(x) ((x) == MINUS_ONE)
#endif
#endif

#define ALLOCA_ALIGN 63UL
#if defined(PARAM_CONF_KP920F_SVE) || defined(PARAM_CONF_KP920F_SME)
#define BUFFER_ALIGN 0x1FFFFFUL // align 2M
#else
#define BUFFER_ALIGN 63UL
#endif

#define NUM_BUFFERS MAX(50, (MAX_CPU_NUMBER * 2 * MAX_PARALLEL_NUMBER))

#ifndef NEEDBUNDERSCORE
#define BLASFUNC(FUNC) FUNC
#else
#define BLASFUNC(FUNC) FUNC##_
#endif

#undef USE_PTHREAD_LOCK
#undef USE_PTHREAD_SPINLOCK

#if defined(USE_PTHREAD_SPINLOCK) && defined(USE_PTHREAD_LOCK)
#error "You can't specify both LOCK operation!"
#endif

#if defined(SMP) || defined(USE_LOCKING)
#define USE_PTHREAD_LOCK
#undef USE_PTHREAD_SPINLOCK
#endif

#ifdef USE_PTHREAD_LOCK
#define BLAS_LOCK(x) pthread_mutex_lock(x)
#define BLAS_UNLOCK(x) pthread_mutex_unlock(x)
#elif defined(USE_PTHREAD_SPINLOCK)
#if !defined(ASSEMBLER)
typedef volatile int pthread_spinlock_t;
int pthread_spin_lock(pthread_spinlock_t *pthreadLock);
int pthread_spin_unlock(pthread_spinlock_t *pthreadLock);
#endif
#define BLAS_LOCK(x) pthread_spin_lock(x)
#define BLAS_UNLOCK(x) pthread_spin_unlock(x)
#else
#define BLAS_LOCK(x) BlasLock(x)
#define BLAS_UNLOCK(x) BlasUnlock(x)
#endif

#ifndef MAX_CPU_NUMBER
#define MAX_CPU_NUMBER 2
#endif

#ifndef SWITCH_RATIO
#define SWITCH_RATIO 2
#endif

#if defined(ARMV9) || defined(ARMV8) || defined(ARMV7) || defined(ARMV6) || defined(ARMV5)
#define YIELDING asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;\n")
#endif

#if !defined(YIELDING)
#define YIELDING sched_yield()
#endif

#ifndef BLAS3_MEM_ALLOC_THRESHOLD
#define BLAS3_MEM_ALLOC_THRESHOLD 160
#endif

#ifdef CBLAS
#include "kblas.h"
#endif

#ifdef ARCH_ARM64
#include "include/common_arm64.h"
#endif

#ifndef ASSEMBLER

#if !defined(RPCC_DEFINED) && !defined(OS_WINDOWS)
#if defined(_POSIX_MONOTONIC_CLOCK)
/* cut the if condition if two lines, otherwise will fail at __GLIBC_PREREQ(2, 17) */
#if defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 17) // don't require -lrt
#define USE_MONOTONIC
#endif
#endif
#endif

/* use similar scale as arm rdtsc for timeouts to work correctly */
static inline unsigned long long rpcc(void)
{
#if defined(USE_MONOTONIC)
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (unsigned long long)t.tv_sec * 1000000000ull + t.tv_nsec;
#else
    struct timeval t;
    gettimeofday(&t, NULL);
    return (unsigned long long)t.tv_sec * 1000000000ull + t.tv_usec * 1000; // 1000: time unit transfer
#endif
}
#define RPCC_DEFINED
#define RPCC64BIT
#endif

#if defined(__GNUC__) && !defined(BLAS_LOCK_DEFINED)
#define BLAS_LOCK_DEFINED
static void inline BlasLock(volatile BLASULONG *address)
{
    do {
        while (*address) {
            YIELDING;
        }
    } while (!__sync_bool_compare_and_swap(address, 0, 1));
}
#endif

#if !defined(RPCC_DEFINED)
#error "rpcc() implementation is missing for your platform"
#endif
#if !defined(BLAS_LOCK_DEFINED)
#error "BlasLock() implementation is missing for your platform"
#endif
#endif

#if defined(OS_LINUX)
#include "include/common_linux.h"
#endif

#define MMAP_ACCESS (PROT_READ | PROT_WRITE)
#define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)

#include "include/common_param.h"

#if defined(DOUBLE)
#define FLOATRET FLOAT
#else
#ifndef NEED_F2CCONV
#define FLOATRET float
#else
#define FLOATRET double
#endif
#endif

#ifndef ASSEMBLER

#if defined(DOUBLE)
#define BLAS_COMPLEX_FLOAT blas_complex_double
#define BLAS_MAKE_COMPLEX_FLOAT(r, i) blas_make_complex_double(r, i)
#else
#define BLAS_COMPLEX_FLOAT blas_complex_float
#define BLAS_MAKE_COMPLEX_FLOAT(r, i) blas_make_complex_float(r, i)
#endif

#ifndef BLAS_COMPLEX_STRUCT
#define CREAL __real__
#define CIMAG __imag__
#else
#define CREAL(Z) ((Z).real)
#define CIMAG(Z) ((Z).imag)
#endif

#endif // ASSEMBLER

#if !defined(ASSEMBLER)

#ifndef MIN
#define MIN(mm, nn) (((mm) > (nn)) ? (nn) : (mm))
#endif

#ifndef MAX
#define MAX(mm, nn) (((mm) < (nn)) ? (nn) : (mm))
#endif

#define TOUPPER(mm)        \
    do {                   \
        if ((mm) > 0x60) { \
            (mm) -= 0x20;  \
        }                  \
    } while (0)

#include "include/common_interface.h"
#ifdef SANITY_CHECK
#include "include/common_reference.h"
#endif
#include "include/common_macro.h"
#include "include/common_level1.h"
#include "include/common_level2.h"
#include "include/common_level3.h"
#include "include/common_stackalloc.h"

#if defined(SMP_SERVER) && defined(SMP_ONDEMAND)
#error Both SMP_SERVER and SMP_ONDEMAND are specified.
#endif

#if defined(SMP_SERVER) || defined(SMP_ONDEMAND)
#include "include/common_thread.h"
#endif

#if (__STDC_VERSION__ >= 201112L)
#include <stdatomic.h>
#endif

#ifdef __cplusplus
extern "C" {
/* Assume C declarations for C++ */
#endif /* __cplusplus */

/* Common Memory Management Routine */
BLAS_API_LOCAL int GetMaxSysCpuNum(void);
BLAS_API_LOCAL int UpdateNumThreads(void);
BLAS_API_LOCAL void *BlasMemoryAlloc(size_t length);
BLAS_API_LOCAL void BlasMemoryFree(const void *freeArea);
BLAS_API_LOCAL void *BlasMemoryAllocNoLock(void); // use malloc without BlasLock
BLAS_API_LOCAL void BlasMemoryFreeNoLock(void *mapAddress);
BLAS_API_LOCAL int GetNumProcs(void);

#ifdef __ELF__
int omp_get_num_procs(void) __attribute__((weak));
int omp_in_parallel(void) __attribute__((weak));
#endif

static inline void BlasUnlock(volatile BLASULONG *address)
{
    MB;
    *address = 0;
}
#ifndef BLAS_FABS
#ifdef XDOUBLE
#define BLAS_FABS fabsl
#elif defined DOUBLE
#define BLAS_FABS fabs
#else
#define BLAS_FABS fabsf
#endif
#endif

#ifndef UNIT
#define GET_COMPLEX_INV(ar, ai, b0, b1) do {                    \
    FLOAT ratio, den;                                           \
    ratio = (ai) / (ar);                                        \
    den = (FLOAT)(ONE / ((ar) * (ONE + (ratio) * (ratio))));    \
    (b0) = den;                                                 \
    (b1) = (ratio) * (den);                                     \
} while (0)

#define COMPINV(b, ar, ai) do {                                 \
    FLOAT b0, b1;                                               \
    if ((BLAS_FABS(ar)) >= (BLAS_FABS(ai))) {                   \
        GET_COMPLEX_INV(ar, ai, b0, b1);                        \
    } else {                                                    \
        GET_COMPLEX_INV(ai, ar, b1, b0);                        \
    }                                                           \
    (b)[0] = b0;                                                \
    (b)[1] = -(b1);                                             \
} while (0)
#else
#define COMPINV(b, ar, ai) do {                                 \
    (b)[0] = ONE;                                               \
    (b)[1] = ZERO;                                              \
} while (0)
#endif
static inline int BlasReadIntEnvParam(char *env)
{
    if (env != NULL) {
        char *p = getenv(env);
        if ((p != NULL) && (strlen(p) < 10)) { // 10: make sure str len less then int32(-2147483648 - 2147483647)
            return strtol(p, NULL, 10);
        }
    }

    return 0;
}

static inline char* BlasReadCharEnvParam(char *env)
{
    if (env != NULL) {
        char *p = getenv(env);
        return p;
    }

    return NULL;
}
#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* ifndef ASSEMBLER */

#endif