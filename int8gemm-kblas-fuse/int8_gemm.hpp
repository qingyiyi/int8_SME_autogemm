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
/************** cblas.h *******************/


// Production inverse-scaling store ABI.  The NN kernel receives a pointer to
// this structure through its existing 11th (buf) argument.  `c8` is the stable
// base of the current output tile and the fused kernel C/ldc argument slots
// begin at the same tile in direct C8 byte coordinates (`ldc8` is unscaled).
// The assembly retains `c8` as the fused-store enable/sentinel; it no longer
// reconstructs a byte address from a virtual C32 coordinate plane.
struct alignas(16) Int8FusedStoreParams {
    int8_t *c8;         // column-major INT8 tile origin
    int32_t modulus;    // 0: low byte; otherwise centered remainder modulus
    uint32_t reserved;  // kept for ABI spacing and future use
    double inv_p;       // 1.0 / modulus; zero for low-byte mode
    double neg_p;       // -modulus; zero for low-byte mode
};

static_assert(offsetof(Int8FusedStoreParams, c8) == 0, "fused ABI c8 offset");
static_assert(offsetof(Int8FusedStoreParams, modulus) == 8, "fused ABI modulus offset");
static_assert(offsetof(Int8FusedStoreParams, reserved) == 12,
              "fused ABI reserved offset");
static_assert(offsetof(Int8FusedStoreParams, inv_p) == 16,
              "fused ABI reciprocal offset");
static_assert(offsetof(Int8FusedStoreParams, neg_p) == 24,
              "fused ABI negative-modulus offset");
static_assert(sizeof(Int8FusedStoreParams) == 32, "fused ABI size");

extern "C" {
    // These two output arguments are the direct C8 output cursor and its
    // unscaled C8 byte stride in the fused NN build.
    void int8_sme_gemm_kernel_nn(
        BLASLONG m, BLASLONG n, BLASLONG k,
        void *sa, BLASLONG lda, float alpha,
        void *sb, BLASLONG ldb,
        int8_t *c8, BLASLONG ldc8, void *store_params
    );

    // Fixed-contract fused API: S8 x S8 GEMM followed by inverse scaling to
    // the final S8 output.  There is intentionally no C32 output pointer,
    // C32 leading dimension, or C32 offset vector.
    void cblas_gemm_s8s8s8(
        const CBLAS_LAYOUT layout,
        const CBLAS_TRANSPOSE transa,
        const CBLAS_TRANSPOSE transb,
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
        int8_t *sa,
        int8_t *sb,
        int8_t *c8,
        size_t ldc8,
        unsigned num_moduli
    );

}



/******************** memory.c *********************/
int sched_getcpu(void);
