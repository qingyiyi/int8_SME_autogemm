#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <sched.h>
#include <dlfcn.h>
#include <hbwmalloc.h>
#include <cmath>

#include <numa.h>
#include <numaif.h>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

typedef int8_t BLASINT8;
typedef int BLASINT;
typedef long BLASLONG;
typedef unsigned long BLASULONG;

#ifndef INT8_SME
#define INT8_SME
#endif



#ifndef BUFFERSIZE
#define BUFFER_SIZE (36 << 20)
#else
#define BUFFER_SIZE (32 << BUFFERSIZE)
#endif

#ifndef INT8_GEMM_H
#define INT8_GEMM_H    
    #ifndef LIKELY
        #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #endif
    #ifndef UNLIKELY
        #define UNLIKELY(x) __builtin_expect(!!(x), 0)
    #endif
    #ifndef MMAP_ERROR_HANDLE
        #define MMAP_ERROR_HANDLE(err, msg) \
            do { fprintf(stderr, "%s (errno=%d)\n", msg, err); } while(0)
    #endif
    #ifndef ALLOC_FROM_OS
        #define ALLOC_FROM_OS 1
    #endif
    
    #define MIN(a, b) ((a) > (b) ? (b) : (a))
    #define MAX(a, b) ((a) > (b) ? (a) : (b))
    #define LHS_INT
    #define RHS_INT
    #define COMP_SV_LEN 512
    #define SMP
    #define CBLAS
    #define REGION_ALIGN_SIZE 4
    #define MAX_CPU_NUMBER 36
    #define FLOAT BLASINT8
    #define FLOATC  int32_t

    #define GEMM_PA_MASK 0x01
    #define GEMM_PB_MASK 0x02
    #define MMAP_ACCESS (PROT_READ | PROT_WRITE)
    #define MMAP_POLICY (MAP_PRIVATE | MAP_ANONYMOUS)
    #define ALLOC_FROM_POOL 0
    #define L1_DATA_LINESIZE 64
    #define MEM_HEAD_SIZE   MAX(sizeof(AllocHead), L1_DATA_LINESIZE) // keep cacheline aligned
    #define BUFFER_ALIGN 0x1FFFFFUL
#endif

#ifndef BLAS_API_LOCAL
#define BLAS_API_LOCAL __attribute__((visibility("hidden")))
#endif
#ifndef PACK_A
#define PACK_A(M, N, A, LDA, X, Y, BUFFER, OFFSET) \
    int8_sme_gemm_itcopy(M, N, (FLOAT *)(A) + ((Y) + (X) * (LDA)) * COMPSIZE, LDA, BUFFER, OFFSET)
#endif

#ifndef PACK_B
#define PACK_B(M, N, A, LDA, X, Y, BUFFER, OFFSET) \
    int8_sme_gemm_oncopy(M, N, (FLOAT *)(A) + ((X) + (Y) * (LDA)) * COMPSIZE, LDA, BUFFER, OFFSET)
#endif

#ifdef __cplusplus
extern "C" {
#endif
void int8_sme_gemm_incopy(const BLASLONG m, const BLASLONG n, const BLASINT8 *a, const BLASLONG lda, BLASINT8 *b, BLASINT8 offset);
void int8_sme_gemm_oncopy(const BLASLONG m, const BLASLONG n, const BLASINT8 *a, const BLASLONG lda, BLASINT8 *b, BLASINT8 offset);
void int8_sme_gemm_itcopy(const BLASLONG m, const BLASLONG n, const BLASINT8 *a, const BLASLONG lda, BLASINT8 *b, BLASINT8 offset);
void int8_sme_gemm_otcopy(const BLASLONG m, const BLASLONG n, const BLASINT8 *a, const BLASLONG lda, BLASINT8 *b, BLASINT8 offset);
#ifdef __cplusplus
}
#endif

/************** cblas.h *******************/
typedef enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_ORDER;
typedef CBLAS_ORDER CBLAS_LAYOUT;
typedef enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113, // conjugate transpose
    CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;
typedef enum CBLAS_OFFSET {
    CblasRowOffset = 171,
    CblasColOffset = 172,
    CblasFixOffset = 173
} CBLAS_OFFSET;
/************** cblas.h *******************/


// Phase-1 fused-store ABI.  The fused kernel receives a pointer to this
// structure through its existing 11th (buf) argument.  Keep this layout
// stable: the assembly uses the byte offsets below after the normal 176-byte
// register save area.
struct Int8FusedStoreParams {
    int32_t *c32;       // column-major tile origin
    int8_t *c8;         // column-major INT8 tile origin
    int64_t ldc32;      // C32 leading dimension, in elements
    int64_t ldc8;       // C8 leading dimension, in elements
    int64_t rows;       // valid rows in this kernel call
    int64_t cols;       // valid columns in this kernel call
    int32_t modulus;    // 0: low byte; otherwise centered remainder modulus
    int32_t reserved;
};

static_assert(offsetof(Int8FusedStoreParams, c32) == 0, "fused ABI c32 offset");
static_assert(offsetof(Int8FusedStoreParams, c8) == 8, "fused ABI c8 offset");
static_assert(offsetof(Int8FusedStoreParams, ldc32) == 16, "fused ABI ldc32 offset");
static_assert(offsetof(Int8FusedStoreParams, ldc8) == 24, "fused ABI ldc8 offset");
static_assert(offsetof(Int8FusedStoreParams, rows) == 32, "fused ABI rows offset");
static_assert(offsetof(Int8FusedStoreParams, cols) == 40, "fused ABI cols offset");
static_assert(offsetof(Int8FusedStoreParams, modulus) == 48, "fused ABI modulus offset");
static_assert(sizeof(Int8FusedStoreParams) == 56, "fused ABI size");

extern "C" {
    void int8_sme_gemm_kernel_nn(
        BLASLONG m, BLASLONG n, BLASLONG k,
        void *sa, BLASLONG lda, float alpha,
        void *sb, BLASLONG ldb,
        int32_t *c, BLASLONG ldc, void *buf
    );

    void int8_sme_gemm_kernel_nn_fused(
        BLASLONG m, BLASLONG n, BLASLONG k,
        void *sa, BLASLONG lda, float alpha,
        void *sb, BLASLONG ldb,
        int32_t *c, BLASLONG ldc, void *buf
    );

    void cblas_gemm_s8s8s32(
        const CBLAS_LAYOUT layout,
        const CBLAS_TRANSPOSE transa,
        const CBLAS_TRANSPOSE transb,
        const CBLAS_OFFSET offsetc,
        const BLASINT m,
        const BLASINT n,
        const BLASINT k,
        const float alpha,
        void *a_,
        const BLASINT lda,
        const BLASINT8 oa,
        void *b_,
        const BLASINT ldb,
        const BLASINT8 ob,
        const float beta,
        int32_t *c,
        const BLASINT ldc,
        const int32_t *oc,
        int8_t *sa,
        int8_t *sb,
        int8_t *C8i_j, 
        size_t ldc8i,
        unsigned num_moduli
    );

    // Phase 1 shadow path: same public arguments as the existing API, but
    // inverse scaling is performed by the fused kernel after each K=2048 tile.
    // For unsupported K values the implementation falls back to the original
    // API so the existing contract is preserved.
    void cblas_gemm_s8s8s32_fused(
        const CBLAS_LAYOUT layout,
        const CBLAS_TRANSPOSE transa,
        const CBLAS_TRANSPOSE transb,
        const CBLAS_OFFSET offsetc,
        const BLASINT m,
        const BLASINT n,
        const BLASINT k,
        const float alpha,
        void *a_,
        const BLASINT lda,
        const BLASINT8 oa,
        void *b_,
        const BLASINT ldb,
        const BLASINT8 ob,
        const float beta,
        int32_t *c,
        const BLASINT ldc,
        const int32_t *oc,
        int8_t *sa,
        int8_t *sb,
        int8_t *C8i_j,
        size_t ldc8i,
        unsigned num_moduli
    );

}



typedef struct {
    const void *a;
    void *b;
    void *c;
    void *d;
    const void *alpha;
    const void *beta;
    BLASLONG m, n, k;
    BLASLONG lda, ldb, ldc, ldd;
    BLASLONG offset;
    BLASINT8 oa;
    BLASINT8 ob;
    const int32_t *oc;
    BLASLONG computeMode;
#if defined(SMP)
    void *common;
    BLASLONG nthreads;
    bool isDoSupKernel;
    BLASLONG smpThreshold;
    BLASLONG unrollSz;
    int threadIdx; // needed for L1 routines which output some reduced value, e.g. dot function
    unsigned int transa;
    unsigned int transb;
    bool isDoSmallKernel;
    BLASLONG kDirectionOption;
    void *cTotal;
    BLASLONG nthreadsM;
    BLASLONG nthreadsN;
#endif
    FLOAT alphaR;
    FLOAT alphaI;
    BLASLONG lastMthreads;
    BLASLONG mblockNumLast;
    BLASULONG mask;
} BlasArgs;


typedef struct BlasQueue_ {
    void *routine;
    BLASLONG position;
    BLASLONG assigned;
    size_t bufSize;

    const BlasArgs *args;
    const void *rangeM;
    const void *rangeN;
    void *sa, *sb;

    struct BlasQueue_ *next;
    volatile int** job_t;

    unsigned int mode;
    int status;
}BlasQueue;



/******************** memory.c *********************/
int sched_getcpu(void);