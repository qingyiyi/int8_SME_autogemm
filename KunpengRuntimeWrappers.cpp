
#include <cstddef>
#include <cstdio>
#include <sched.h>
#include <cassert>
#include "mdk_sdma.h"
#include "sdma_utils.h"
#include <linux/futex.h>  // futex 相关定义
#include <sys/syscall.h>  // syscall 函数
#include <unistd.h>       // 系统调用号（如 SYS_futex）
#include <iostream>
#include <kblas.h>
#include <complex>
#include <stdint.h>
#include <stddef.h>
#include <arm_sme.h>
#define INT_MAX 2147483647

#define YIELDING asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;\n")
#define MLIR_KUNPENG_WRAPPERS_EXPORT __attribute__((visibility("default")))
#define TASK_STATE_LOOP_DONE 99999
#define MB __asm__ __volatile__("dmb  ish" : : : "memory")
#define WMB __asm__ __volatile__("dmb  ishst" : : : "memory")
#define RMB __asm__ __volatile__("dmb  ishld" : : : "memory")

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void print_value(int8_t val)
{
    printf("print a int8_t value in mlir: %d\n", val);
}

extern "C" float sgemm_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);
extern "C" float sgemm_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);
/*extern "C" float sgemm_custom_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float
alpha0, float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf); extern "C" float sgemm_custom_kernel_nn(BLASLONG m,
BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0, float *b, BLASLONG ldb, float *c, BLASLONG ldc,
float *buf);
*/
extern "C" float sgemm_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);
extern "C" float sgemm_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);
extern "C" float sgemm_small_kernel_b0_nn(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda,
    const float alpha0, float *b, BLASLONG ldb, float *c, BLASLONG ldc);
extern "C" float sgemm_kernel_npa_npb(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf, BLASLONG stepCnt);
extern "C" float dgemm_kernel_npa_npb(BLASLONG m, BLASLONG n, BLASLONG k, double *a, BLASLONG lda, const double alpha0,
    double *b, BLASLONG ldb, double *c, BLASLONG ldc, double *buf, BLASLONG stepCnt);
extern "C" float dgemm_kernel_npa_npb_alpha1(BLASLONG m, BLASLONG n, BLASLONG k, double *a, BLASLONG lda,
    const double alpha0, double *b, BLASLONG ldb, double *c, BLASLONG ldc, double *buf, BLASLONG stepCnt);

extern "C" void sgemm_incopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);
extern "C" void sgemm_itcopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);
extern "C" void sgemm_oncopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);
extern "C" void sgemm_otcopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);

extern "C" void cgemm_incopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);
extern "C" void cgemm_itcopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);
extern "C" void cgemm_oncopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);
extern "C" void cgemm_otcopy(const BLASLONG m, const BLASLONG n, const float *a, const BLASLONG lda, float *b);

extern "C" float cgemm_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    const float alpha1, float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);
extern "C" float cgemm_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    const float alpha1, float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);

extern "C" float cgemm_kernel_npa_npb(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    const float alpha1, float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf, BLASLONG stepStart);

extern "C" float cgemm_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    const float alpha1, float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);
extern "C" float cgemm_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, float *a, BLASLONG lda, const float alpha0,
    const float alpha1, float *b, BLASLONG ldb, float *c, BLASLONG ldc, float *buf);

extern "C" void zgemm_incopy(const BLASLONG m, const BLASLONG n, const double *a, const BLASLONG lda, double *b);
extern "C" void zgemm_itcopy(const BLASLONG m, const BLASLONG n, const double *a, const BLASLONG lda, double *b);
extern "C" void zgemm_oncopy(const BLASLONG m, const BLASLONG n, const double *a, const BLASLONG lda, double *b);
extern "C" void zgemm_otcopy(const BLASLONG m, const BLASLONG n, const double *a, const BLASLONG lda, double *b);

extern "C" double zgemm_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, double *a, BLASLONG lda, const double alpha0,
    const double alpha1, double *b, BLASLONG ldb, double *c, BLASLONG ldc, double *buf);
extern "C" double zgemm_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, double *a, BLASLONG lda, const double alpha0,
    const double alpha1, double *b, BLASLONG ldb, double *c, BLASLONG ldc, double *buf);
extern "C" double zgemm_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, double *a, BLASLONG lda, const double alpha0,
    const double alpha1, double *b, BLASLONG ldb, double *c, BLASLONG ldc, double *buf);
extern "C" double zgemm_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, double *a, BLASLONG lda, const double alpha0,
    const double alpha1, double *b, BLASLONG ldb, double *c, BLASLONG ldc, double *buf);

extern "C" __bf16 bgemm_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, __bf16 *a, BLASLONG lda, const __bf16 alpha0,
    __bf16 *b, BLASLONG ldb, __bf16 *c, BLASLONG ldc, __bf16 *buf);
// extern "C" __bf16 bgemm_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, __bf16 *a, BLASLONG lda, const __bf16 alpha0,
//     __bf16 *b, BLASLONG ldb, __bf16 *c, BLASLONG ldc, __bf16 *buf);
extern "C" __bf16 bgemm_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, __bf16 *a, BLASLONG lda, const __bf16 alpha0,
    __bf16 *b, BLASLONG ldb, __bf16 *c, BLASLONG ldc, __bf16 *buf);
extern "C" __bf16 bgemm_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, __bf16 *a, BLASLONG lda, const __bf16 alpha0,
    __bf16 *b, BLASLONG ldb, __bf16 *c, BLASLONG ldc, __bf16 *buf);

extern "C" void bgemm_incopy(const BLASLONG m, const BLASLONG n, const __bf16 *a, const BLASLONG lda, __bf16 *b);
extern "C" void bgemm_itcopy(const BLASLONG m, const BLASLONG n, const __bf16 *a, const BLASLONG lda, __bf16 *b);
extern "C" void bgemm_oncopy(const BLASLONG m, const BLASLONG n, const __bf16 *a, const BLASLONG lda, __bf16 *b);
extern "C" void bgemm_otcopy(const BLASLONG m, const BLASLONG n, const __bf16 *a, const BLASLONG lda, __bf16 *b);
extern "C" void pack_b_s8_n(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_s8_n(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);
extern "C" void pack_b_s8_t(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_s8_t(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);

// u8 pack kernels
extern "C" void pack_b_u8_n(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_u8_n(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);
extern "C" void pack_b_u8_t(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_u8_t(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);

// s8 pack kernels
extern "C" void pack_b_for_sme_s8_n(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_for_sme_s8_n(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);
extern "C" void pack_b_for_sme_s8_t(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_for_sme_s8_t(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);

// u8 pack kernels
extern "C" void pack_b_for_sme_u8_n(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_for_sme_u8_n(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);
extern "C" void pack_b_for_sme_u8_t(
    void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size, size_t ldb, const BLASINT8 ob);
extern "C" void pack_a_for_sme_u8_t(
    void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size, size_t lda, const BLASINT8 oa);
// extern "C" void gemm_kernel_s8s8s32_1x1(const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc,
//     int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_1x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_1x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_1x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_1x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_2x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_2x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_2x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_2x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_3x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_3x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_3x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_3x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_4x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_4x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_4x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8s8s32_4x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_1x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_1x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_1x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_1x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_2x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_2x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_2x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_2x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_3x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_3x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_3x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_3x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_4x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_4x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_4x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_s8u8s32_4x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_1x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_1x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_1x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_1x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_2x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_2x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_2x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_2x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_3x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_3x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_3x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_3x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_4x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_4x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_4x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8s8s32_4x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_1x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_1x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_1x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_1x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_2x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_2x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_2x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_2x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_3x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_3x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_3x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_3x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_4x1(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_4x2(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_4x3(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);
extern "C" void gemm_kernel_u8u8s32_4x4(
    const void *lhs_ptr, const void *rhs_ptr, int32_t *accum_ptr, size_t ldc, int64_t k_depth, int64_t sv_len);

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t SdmaICopyWrap(void *phandle, void *srcPtr, void *dstPtr,
    int64_t srcOffset, int64_t srcStride, int64_t i, int64_t j, int64_t dstOffset, int64_t dstStride, int64_t k,
    int64_t l, int64_t numElements, int64_t numEltPerStride, int64_t qos, void *req, void *sdmaMgr)
{
    // printf("enter sdmaIcopy!\n");
    sdma_sqe_task_t sqeTask;
    auto sdmaMgrReal = reinterpret_cast<SdmaMgr *>(sdmaMgr);
    sqeTask.src_addr = reinterpret_cast<uint64_t>(srcPtr) + (srcStride * i + j) + srcOffset;
    sqeTask.dst_addr = reinterpret_cast<uint64_t>(dstPtr) + (dstStride * k + l) + dstOffset;
    sqeTask.src_process_id = sdmaMgrReal->srcPasid;
    sqeTask.dst_process_id = sdmaMgrReal->dstPasid;
    sqeTask.src_stride_len = srcStride - numEltPerStride;
    // sqeTask.src_stride_len = 0;
    sqeTask.dst_stride_len = dstStride - numEltPerStride;
    // sqeTask.dst_stride_len = 0;
    sqeTask.stride_num = numElements / numEltPerStride;
    sqeTask.length = numEltPerStride;
    sqeTask.opcode = 0;
    sqeTask.qos = qos;
    sqeTask.next_sqe = NULL;
    auto request = reinterpret_cast<sdma_request_t *>(req);
    int ret = SDMA_SUCCESS;
    int retry = 0;
    // printf("src_stride_len: %d, dst_stride_len: %d, stride_num: %d, length: %d, i: %ld, j: %ld, k: %ld, l: %ld\n",
    //     sqeTask.src_stride_len,
    //     sqeTask.dst_stride_len,
    //     sqeTask.stride_num,
    //     sqeTask.length,
    //     i,
    //     j,
    //     k,
    //     l);
    while (retry < 10) {
        ret = sdma_icopy_data(phandle, &sqeTask, 1, request);
        if (ret != SDMA_SUCCESS) {
            retry++;
            continue;
        }
        if (ret != SDMA_SUCCESS) {
            printf("sdma start error!");
        }
        break;
    }
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t sdmaCopyWrap(void *allocPtrSrc, void *alignPtrSrc, int64_t offsetSrc,
    int64_t sizeSrc0, int64_t sizeSrc1, int64_t strideSrc0, int64_t strideSrc1, void *allocPtrDst, void *alignPtrDst,
    int64_t offsetDst, int64_t sizeDst0, int64_t sizeDst1, int64_t strideDst0, int64_t strideDst1, int64_t i, int64_t j,
    int64_t k, int64_t l, int64_t sliceSize0, int64_t sliceSize1, void *phandle, void *req, void *sdmaMgr, int64_t qos)
{
    // printf("enter sdmaIcopy!\n");
    // std::cout << "sdmaIcopy req: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req))
    //           << ", sdmaIcopy handle: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(phandle)) << std::endl;
    float *alignPtrSrcFloat = reinterpret_cast<float *>(alignPtrSrc);
    float *alignPtrDstFloat = reinterpret_cast<float *>(alignPtrDst);
    auto src_addr = reinterpret_cast<uint64_t>(alignPtrSrcFloat + i * strideSrc0 + j * strideSrc1 + offsetSrc);
    auto dst_addr = reinterpret_cast<uint64_t>(alignPtrDstFloat + k * strideDst0 + l * strideDst1 + offsetDst);
    auto sdmaMgrReal = reinterpret_cast<SdmaMgr *>(sdmaMgr);

    auto strideNum = sliceSize0;
    auto length = sliceSize1 * sizeof(float);
    auto src_stride_len = strideSrc0 * sizeof(float) - length;
    auto dst_stride_len = strideDst0 * sizeof(float) - length;
    if (strideSrc0 == 1) {
        strideNum = sliceSize1;
        length = sliceSize0 * sizeof(float);
        src_stride_len = strideSrc1 * sizeof(float) - length;
        dst_stride_len = strideDst1 * sizeof(float) - length;
    }
    sdma_sqe_task_t sqeTask;
    sqeTask.src_addr = src_addr;
    sqeTask.dst_addr = dst_addr;
    sqeTask.src_process_id = sdmaMgrReal->srcPasid;
    sqeTask.dst_process_id = sdmaMgrReal->dstPasid;
    sqeTask.src_stride_len = src_stride_len;
    // sqeTask.src_stride_len = 0;
    sqeTask.dst_stride_len = dst_stride_len;
    // sqeTask.dst_stride_len = 0;
    sqeTask.stride_num = strideNum;
    sqeTask.length = length;
    sqeTask.opcode = 0;
    sqeTask.qos = qos;
    sqeTask.next_sqe = NULL;
    auto request = reinterpret_cast<sdma_request_t *>(req);
    int ret = SDMA_SUCCESS;
    int retry = 0;
    //     printf("src_stride_len: %d, dst_stride_len: %d, stride_num: %d, length: %d, i: %ld, j: %ld, k: %ld, l: %ld, "
    //        "mthreadNum: %ld, nThreadNum: %ld, src_addr: %ld, dst_addr: %ld, req: %ld\n",
    //     sqeTask.src_stride_len,
    //     sqeTask.dst_stride_len,
    //     sqeTask.stride_num,
    //     sqeTask.length,
    //     i,
    //     j,
    //     k,
    //     l,
    //     -1,
    //     -1,
    //     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.src_addr)),
    //     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.dst_addr)),
    //     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(request)));
    // std::cout << "src: " <<  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.src_addr)) << std::endl;
    // std::cout << "dst: " <<  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.dst_addr)) << std::endl;
    while (retry < 10) {
        ret = sdma_icopy_data(phandle, &sqeTask, 1, request);
        if (ret != SDMA_SUCCESS) {
            retry++;
            continue;
        }
        if (ret != SDMA_SUCCESS) {
            printf("sdma start error!");
        }
        break;
    }
    // printf("out sdmaIcopy!\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t sdmaCopyDoubleWrap(void *allocPtrSrc, void *alignPtrSrc,
    int64_t offsetSrc, int64_t sizeSrc0, int64_t sizeSrc1, int64_t strideSrc0, int64_t strideSrc1, void *allocPtrDst,
    void *alignPtrDst, int64_t offsetDst, int64_t sizeDst0, int64_t sizeDst1, int64_t strideDst0, int64_t strideDst1,
    int64_t i, int64_t j, int64_t k, int64_t l, int64_t sliceSize0, int64_t sliceSize1, void *phandle, void *req,
    void *sdmaMgr, int64_t qos, int64_t mTid, int64_t nTid)
{
    // printf("enter sdmaIcopy!\n");
    // std::cout << "sdmaIcopy req: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req))
    //           << ", sdmaIcopy handle: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(phandle)) << std::endl;
    double *alignPtrSrcFloat = reinterpret_cast<double *>(alignPtrSrc);
    double *alignPtrDstFloat = reinterpret_cast<double *>(alignPtrDst);
    auto src_addr = reinterpret_cast<uint64_t>(alignPtrSrcFloat + i * strideSrc0 + j * strideSrc1 + offsetSrc);
    auto dst_addr = reinterpret_cast<uint64_t>(alignPtrDstFloat + k * strideDst0 + l * strideDst1 + offsetDst);
    auto sdmaMgrReal = reinterpret_cast<SdmaMgr *>(sdmaMgr);

    auto strideNum = sliceSize0;
    auto length = sliceSize1 * sizeof(double);
    auto src_stride_len = strideSrc0 * sizeof(double) - length;
    auto dst_stride_len = strideDst0 * sizeof(double) - length;
    if (strideSrc0 == 1) {
        strideNum = sliceSize1;
        length = sliceSize0 * sizeof(double);
        src_stride_len = strideSrc1 * sizeof(double) - length;
        dst_stride_len = strideDst1 * sizeof(double) - length;
    }
    sdma_sqe_task_t sqeTask;
    sqeTask.src_addr = src_addr;
    sqeTask.dst_addr = dst_addr;
    sqeTask.src_process_id = sdmaMgrReal->srcPasid;
    sqeTask.dst_process_id = sdmaMgrReal->dstPasid;
    sqeTask.src_stride_len = src_stride_len;
    // sqeTask.src_stride_len = 0;
    sqeTask.dst_stride_len = dst_stride_len;
    // sqeTask.dst_stride_len = 0;
    sqeTask.stride_num = strideNum;
    sqeTask.length = length;
    sqeTask.opcode = 0;
    sqeTask.qos = qos;
    sqeTask.next_sqe = NULL;
    auto request = reinterpret_cast<sdma_request_t *>(req);
    int ret = SDMA_SUCCESS;
    int retry = 0;
    // printf("src_stride_len: %d, dst_stride_len: %d, stride_num: %d, length: %d, i: %ld, j: %ld, k: %ld, l: %ld, "
    //    "mthreadNum: %ld, nThreadNum: %ld, src_addr: %ld, dst_addr: %ld, req: %ld\n",
    // sqeTask.src_stride_len,
    // sqeTask.dst_stride_len,
    // sqeTask.stride_num,
    // sqeTask.length,
    // i,
    // j,
    // k,
    // l,
    // -1,
    // -1,
    // static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.src_addr)),
    // static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.dst_addr)),
    // static_cast<uint64_t>(reinterpret_cast<uintptr_t>(request)));
    // std::cout << "src: " <<  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.src_addr)) << std::endl;
    // std::cout << "dst: " <<  static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sqeTask.dst_addr)) << std::endl;
    while (retry < 10) {
        ret = sdma_icopy_data(phandle, &sqeTask, 1, request);
        if (ret != SDMA_SUCCESS) {
            retry++;
            continue;
        }
        if (ret != SDMA_SUCCESS) {
            printf("sdma start error!");
        }
        break;
    }
    // printf("out sdmaIcopy!\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t sdmaCopyComplexWrap(void *allocPtrSrc, void *alignPtrSrc,
    int64_t offsetSrc, int64_t sizeSrc0, int64_t sizeSrc1, int64_t strideSrc0, int64_t strideSrc1, void *allocPtrDst,
    void *alignPtrDst, int64_t offsetDst, int64_t sizeDst0, int64_t sizeDst1, int64_t strideDst0, int64_t strideDst1,
    int64_t i, int64_t j, int64_t k, int64_t l, int64_t sliceSize0, int64_t sliceSize1, void *phandle, void *req,
    void *sdmaMgr, int64_t qos, int64_t mTid, int64_t nTid)
{
    // printf("enter sdmaIcopy!\n");
    // std::cout << "sdmaIcopy req: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req))
    //           << ", sdmaIcopy handle: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(phandle)) << std::endl;
    std::complex<float> *alignPtrSrcFloat = reinterpret_cast<std::complex<float> *>(alignPtrSrc);
    std::complex<float> *alignPtrDstFloat = reinterpret_cast<std::complex<float> *>(alignPtrDst);
    auto src_addr = reinterpret_cast<uint64_t>(alignPtrSrcFloat + i * strideSrc0 + j * strideSrc1 + offsetSrc);
    auto dst_addr = reinterpret_cast<uint64_t>(alignPtrDstFloat + k * strideDst0 + l * strideDst1 + offsetDst);
    auto sdmaMgrReal = reinterpret_cast<SdmaMgr *>(sdmaMgr);

    auto strideNum = sliceSize0;
    auto length = sliceSize1 * sizeof(std::complex<float>);
    auto src_stride_len = strideSrc0 * sizeof(std::complex<float>) - length;
    auto dst_stride_len = strideDst0 * sizeof(std::complex<float>) - length;
    if (strideSrc0 == 1) {
        strideNum = sliceSize1;
        length = sliceSize0 * sizeof(std::complex<float>);
        src_stride_len = strideSrc1 * sizeof(std::complex<float>) - length;
        dst_stride_len = strideDst1 * sizeof(std::complex<float>) - length;
    }
    sdma_sqe_task_t sqeTask;
    sqeTask.src_addr = src_addr;
    sqeTask.dst_addr = dst_addr;
    sqeTask.src_process_id = sdmaMgrReal->srcPasid;
    sqeTask.dst_process_id = sdmaMgrReal->dstPasid;
    sqeTask.src_stride_len = src_stride_len;
    // sqeTask.src_stride_len = 0;
    sqeTask.dst_stride_len = dst_stride_len;
    // sqeTask.dst_stride_len = 0;
    sqeTask.stride_num = strideNum;
    sqeTask.length = length;
    sqeTask.opcode = 0;
    sqeTask.qos = qos;
    sqeTask.next_sqe = NULL;
    auto request = reinterpret_cast<sdma_request_t *>(req);
    int ret = SDMA_SUCCESS;
    int retry = 0;
    while (retry < 10) {
        ret = sdma_icopy_data(phandle, &sqeTask, 1, request);
        if (ret != SDMA_SUCCESS) {
            retry++;
            continue;
        }
        if (ret != SDMA_SUCCESS) {
            printf("sdma start error!");
        }
        break;
    }
    // printf("out sdmaIcopy!\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t sdmaIWaitWrap(void *phandle, void *req)
{
    // printf("enter sdmawait!\n");
    // std::cout << "sdmawait req: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(req))
    //           << ", sdmawait handle: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(phandle)) << std::endl;

    auto sdmaRequest = reinterpret_cast<sdma_request_t *>(req);
    int ret = SDMA_SUCCESS;

    int retry = 0;
    // printf("enter sdmaIWaitWrap! \n");
    while (true) {
        ret = sdma_iquery_chn(phandle, sdmaRequest);
        if (ret == SDMA_RNDCNT_ERR) {
            YIELDING;
            continue;
        }

        if (ret == SDMA_LOCK_TIMEOUT) {
            if (retry < 10) {
                retry++;
                printf("\n wait failed ,and retry for %d \n", retry);
                continue;
            }
            return ret;
        }
        break;
    }
    if (ret != SDMA_SUCCESS) {
        printf("\n wait error! %ld\n", static_cast<int64_t>(ret));
    }
    // printf("out sdmawait!\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void *GetChannelIdByCoreId(void *sdmaMgr)
{
    unsigned int cpuid = 0;
    unsigned int nodeid = 0;

    getcpu(&cpuid, &nodeid);
    unsigned int channelId = cpuid % MAX_SDMA_CHNL;
    auto sdmaMgrReal = reinterpret_cast<SdmaMgr *>(sdmaMgr);
    void *chnl = sdmaMgrReal->channels[channelId];
    if (chnl == NULL) {
        printf("%s: chnl is null at %d\n", __func__, channelId);
        assert(chnl != NULL);
    }
    return chnl;
}

struct SdmaInfoExt {
    void *channelHandle;
    void *req;
    int64_t qos;
    int64_t loadIdx = 0;
    int64_t waitIdx = 0;
    int64_t sqaNum = 0;
    int64_t gemmIdx = 0;
    int64_t cachePrefetchIdxPtr = 0;
};

// priority=1->high, priority=0->low
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void UpdateChannelIdAndQosByCoreId(void *sdmaMgr, void *sdmaInfoPtr,
    int64_t chlType, int64_t priority, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum)
{
    unsigned int cpuid = 0;
    unsigned int nodeid = 0;

    // getcpu(&cpuid, &nodeid);
    unsigned int channelId = (mTid * nThreadNum + nTid) % MAX_SDMA_CHNL;
    if (chlType == 1) {
        channelId = nTid + 36;
    }
    auto sdmaMgrReal = reinterpret_cast<SdmaMgr *>(sdmaMgr);
    void *chnl = sdmaMgrReal->channels[channelId];
    if (chnl == NULL) {
        printf("%s: chnl is null at %d\n", __func__, channelId);
        assert(chnl != NULL);
    }
    SdmaInfoExt *typedSdmaInfoPtr = reinterpret_cast<SdmaInfoExt *>(sdmaInfoPtr);
    typedSdmaInfoPtr->channelHandle = chnl;
    // if (chlType >= 2) {
    //     typedSdmaInfoPtr->qos = 7;
    // } else if (priority == 1) {
    //     typedSdmaInfoPtr->qos = 5;
    // } else {
    //     typedSdmaInfoPtr->qos = 3;
    // }
    if (chlType == 3) {
        typedSdmaInfoPtr->qos = 4;
    } else if (chlType == 2) {
        typedSdmaInfoPtr->qos = 7;
    } else if (chlType == 1) {
        typedSdmaInfoPtr->qos = 12;
    } else if (chlType == 0) {
        typedSdmaInfoPtr->qos = 7;
    }
    /*
    printf("chlType: %ld, channelId: %ld, mTid: %ld, nTid: %ld, mThreadNum: %ld, nThreadNum: %ld\n",
        chlType,
        channelId,
        mTid,
        nTid,
        mThreadNum,
        nThreadNum);*/
    return;
}

struct BarrierInt {
    int phase;
    int initCount;
    int left;
    pthread_mutex_t *mutex;
};

static int futex(int *uaddr, int futex_op, int val, const struct timespec *timeout)
{
    return syscall(SYS_futex, uaddr, futex_op, val, timeout, NULL, 0);
}

int futexWait(int *futex_ptr, int expected_val)
{
    // 检查 futex_ptr 的值是否等于 expected_val
    // 如果相等，则阻塞线程，等待唤醒
    return futex(futex_ptr, FUTEX_WAIT, expected_val, NULL);
}

int futexWake(int *futex_ptr, int num_to_wake)
{
    // 唤醒最多 num_to_wake 个等待在 futex_ptr 上的线程
    return futex(futex_ptr, FUTEX_WAKE, num_to_wake, NULL);
}

#define YIELDING asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;\n")
#define MB __asm__ __volatile__("dmb  ish" : : : "memory")

/*

1025 * 256
257 256
256 256
256 256
256 256


for k in range(0, 512, 256)
    for m in range(0, 512, 256):
        corePack A
        wait()
        for n in range(0, 512, 256):
            corepack C
            wait()
            tt = 0
        setDoneC
        waitSync1  //使用waitDoneC // 只阻塞了循环提前结束的，而其他线程未阻塞，循环提前结束的线程可能会卡死
        setZeroC
        waitSync1

for k in range(0, 523, 256)
    for m in range(0, 513, 256):
        for n in range(0, 513, 256):
            compute()


sync1Cnt = [0]*38
sync2Cnt = [0]*38
TASK_STATE_LOOP_DONE = 99999
for k in range(0, 512, 256)
    for m in range(0, 512, 256):
        for n in range(0, 512, 256):
            compute()
            key = addOneSync1()
            waitSync1(key)
        setDoneSync1(TASK_STATE_LOOP_DONE)
        key = addOneSync2()
        waitSync2(key)
        setZeroSync1()
        key = addOneSync2()
        waitSync2(key)
    setDoneSync2(TASK_STATE_LOOP_DONE+1)
    setDoneSync1(TASK_STATE_LOOP_DONE+1)
    waitSync1(TASK_STATE_LOOP_DONE+1)
    setZeroSync2()
    key = addOneSync1()
    waitSync1(key)
    key = addSync2()
    waitSync2(key)
    setZeroSync1()
    key = addSync2()
    waitSync2(key)




ASK_STATE_LOOP_DONE = 99999
for k in range(0, 512, 256)
    for m in range(0, 512, 256):
        if (masterA): corePackA
        key = addASync1()
        waitASync1(key)
        for n in range(0, 512, 256):
            if (masterB): corePackB
            key = addBSync1()
            waitBSync1(key)
            compute()
            key = addOneSync1()
            waitSync1(key)
        setDoneSync1(TASK_STATE_LOOP_DONE)
        setDoneBSync1(TASK_STATE_LOOP_DONE)
        key = addOneSync2()
        waitSync2(key)
        setZeroSync1()
        setZeroBSync1()
        key = addOneSync2()
        waitSync2(key)
    setDoneSync2(TASK_STATE_LOOP_DONE+1)
    setDoneSync1(TASK_STATE_LOOP_DONE+1)
    setDoneBSync1(TASK_STATE_LOOP_DONE+1)
    setDoneASync1(TASK_STATE_LOOP_DONE)
    waitSync1(TASK_STATE_LOOP_DONE+1)
    setZeroSync2()
    key = addOneSync1()
    waitSync1(key)
    key = addSync2()
    waitSync2(key)
    setZeroSync1()
    setZeroBSync1()
    setZeroASync1()
    key = addSync2()
    waitSync2(key)




    setDoneASync1(TASK_STATE_LOOP_DONE)
    key = addOneASync2()
    waitASync2(key)
    setZeroASync1()
    key = addOneASync2()
    waitASync2(key)



for m in range(0, 512, 256):
    for k in range(0, 512, 256)
        for n in range(0, 512, 256):
            compute()
            waitSync1()
        setDoneSync1(TASK_STATE_LOOP_DONE)
        key = addOneSync2()
        waitSync2(key)
        setZeroSync1()
        key = addOneSync2()
        waitSync2(key)
setDoneSync2(TASK_STATE_LOOP_DONE+1)
setDoneSync1(TASK_STATE_LOOP_DONE+1)
waitSync1(TASK_STATE_LOOP_DONE+1)
setZeroSync2()
key = addOneSync1()
waitSync1(key)
key = addSync2()
waitSync2(key)
setZeroSync1()
key = addSync2()
waitSync2(key)



*/

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void print_tmp(int64_t Asize0, int64_t Asize1, int64_t Bsize0, int64_t Bsize1,
    int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum)
{
    /*
     printf("-----------Asize0: %ld, Asize1: %ld, Bsize0: %ld, Bsize1: %ld, mTid: %ld, nTid: %ld, mthreadNum: %ld, "
            "nThreadNum: %ld\n",
         Asize0,
         Asize1,
         Bsize0,
         Bsize1,
         mTid,
         nTid,
         mThreadNum,
         nThreadNum);*/
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void print_tmp1(
    int64_t m, int64_t n, int64_t k, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum)
{
    printf("##########m: %ld, n: %ld, k: %ld, mTid: %ld, nTid: %ld, mthreadNum: %ld, nThreadNum: %ld\n",
        m,
        n,
        k,
        mTid,
        nTid,
        mThreadNum,
        nThreadNum);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void print_tmp2(
    int64_t m, int64_t n, int64_t k, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum)
{
    printf("@@@@@@@@@@@@m: %ld, n: %ld, k: %ld,, mTid: %ld, nTid: %ld, mthreadNum: %ld, nThreadNum: %ld\n",
        m,
        n,
        k,
        mTid,
        nTid,
        mThreadNum,
        nThreadNum);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void print_tmp3(
    int64_t m, int64_t n, int64_t k, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum)
{
    printf("!!!!!!!!!m: %ld, n: %ld, k: %ld,, mTid: %ld, nTid: %ld, mthreadNum: %ld, nThreadNum: %ld\n",
        m,
        n,
        k,
        mTid,
        nTid,
        mThreadNum,
        nThreadNum);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void print_tmp4(int64_t m, int64_t n, int64_t k, int64_t mTid, int64_t nTid,
    int64_t mThreadNum, int64_t nThreadNum, int64_t offset0, int64_t offset1, int64_t size0, int64_t size1)
{
    printf("kkkkkm: %ld, n: %ld, k: %ld,, mTid: %ld, nTid: %ld, mthreadNum: %ld, nThreadNum: %ld, offset0: %ld, "
           "offset1: %ld, size0: %ld, size1: %ld\n",
        m,
        n,
        k,
        mTid,
        nTid,
        mThreadNum,
        nThreadNum,
        offset0,
        offset1,
        size0,
        size1);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void print_tmp5(int64_t m, int64_t n, int64_t k, int64_t mTid, int64_t nTid,
    int64_t mThreadNum, int64_t nThreadNum, int64_t offset0, int64_t offset1, int64_t size0, int64_t size1)
{
    printf("oooom: %ld, n: %ld, k: %ld,, mTid: %ld, nTid: %ld, mthreadNum: %ld, nThreadNum: %ld, offset0: %ld, "
           "offset1: %ld, size0: %ld, size1: %ld\n",
        m,
        n,
        k,
        mTid,
        nTid,
        mThreadNum,
        nThreadNum,
        offset0,
        offset1,
        size0,
        size1);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t BarrierWait(
    void *ptr, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum, int64_t barType)
{
    // printf("enter bar!\n");
    volatile int64_t *ptrInt64 = reinterpret_cast<volatile int64_t *>(ptr);
    // if (mTid == 0 && nTid == 0) {
    //     std::cout << "BarrierWait: ";
    //     for (int64_t i = 0; i < 16; i += 1) {
    //         std::cout << ptrInt64[i] << " ";
    //     }
    //     std::cout << std::endl;
    //     printf("barPtr: %ld\n", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
    // }
    ptrInt64[mTid * nThreadNum + nTid]++;
    WMB;

    int64_t curCnt = ptrInt64[mTid * nThreadNum + nTid];

    int64_t groupStart = 0;
    int64_t groupEnd = mThreadNum * nThreadNum;
    int64_t step = 1;

    if (barType == 0) {
        groupStart = mTid * nThreadNum;
        groupEnd = groupStart + nThreadNum;
        step = 1;
    } else if (barType == 1) {
        groupStart = nTid;
        groupEnd = mThreadNum * nThreadNum;
        step = nThreadNum;
    }
    // printf("curCnt: %ld, groupStart: %ld, groupEnd: %ld, step: %ld, barType: %ld, barPtr: %ld, mTid: %ld, nTid: %ld,
    // "
    //        "mThreadNum: %ld, nThreadNum: %ld\n",
    //     curCnt,
    //     groupStart,
    //     groupEnd,
    //     step,
    //     barType,
    //     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)),
    //     mTid,
    //     nTid,
    //     mThreadNum,
    //     nThreadNum);
    for (int64_t i = groupStart; i < groupEnd; i += step) {
        int64_t cnt = 0;
        while (ptrInt64[i] < curCnt) {
            // printf("i: %ld, v: %ld, curCnt: %ld\n", i, ptrInt64[i], curCnt);
            YIELDING;
            MB;
            cnt += 1;
            if (cnt > 500000000) {
                printf("curCnt: %ld, mTid: %ld, nTid: %ld, mthreadNum: %ld, nThreadNum: %ld, step: %ld, barType: %ld, "
                       "barPtr: %ld\n",
                    curCnt,
                    mTid,
                    nTid,
                    mThreadNum,
                    nThreadNum,
                    step,
                    barType,
                    static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));

                std::string out = "BarrierWait: ";
                for (int64_t i = 0; i < 16; i += 1) {
                    out += (std::to_string(ptrInt64[i]) + " ");
                }
                out += std::to_string(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
                std::cout << out << std::endl;
                int64_t tmp_cnt = 500000000;
                while (tmp_cnt > 0) {
                    YIELDING;
                    tmp_cnt--;
                }
                std::cout << "###############################" << std::endl;
                out = "BarrierWait: ";
                for (int64_t i = 0; i < 16; i += 1) {
                    out += (std::to_string(ptrInt64[i]) + " ");
                }
                out += std::to_string(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
                std::cout << out << std::endl;
                while (1) {
                }
            }
        }
    }
    // printf("out bar!\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t BarrierWait2(void *ptr, int64_t mTid, int64_t nTid, int64_t mThreadNum,
    int64_t nThreadNum, int64_t barType, void *dataReady, void *dataReady1)
{
    // printf("enter bar!\n");
    volatile int64_t *ptrInt64 = reinterpret_cast<volatile int64_t *>(ptr);
    // if (mTid == 0 && nTid == 0) {
    //     std::cout << "BarrierWait: ";
    //     for (int64_t i = 0; i < 16; i += 1) {
    //         std::cout << ptrInt64[i] << " ";
    //     }
    //     std::cout << std::endl;
    //     printf("barPtr: %ld\n", static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
    // }
    ptrInt64[mTid * nThreadNum + nTid]++;
    WMB;

    int64_t curCnt = ptrInt64[mTid * nThreadNum + nTid];

    int64_t groupStart = 0;
    int64_t groupEnd = mThreadNum * nThreadNum;
    int64_t step = 1;

    // if (barType == 0) {
    //     groupStart = mTid * nThreadNum;
    //     groupEnd = groupStart + nThreadNum;
    //     step = 1;
    // } else if (barType == 1) {
    //     groupStart = nTid;
    //     groupEnd = mThreadNum * nThreadNum;
    //     step = nThreadNum;
    // }
    // volatile int64_t *ptrInt64_tmp = reinterpret_cast<volatile int64_t *>(dataReady);
    // volatile int64_t *ptr2Int64 = reinterpret_cast<volatile int64_t *>(dataReady);
    // volatile int64_t *ptr3Int64 = reinterpret_cast<volatile int64_t *>(dataReady1);
    // int64_t ptr2cnt = ptr2Int64[mTid * nThreadNum + nTid];

    // printf("curCnt: %ld, groupStart: %ld, groupEnd: %ld, step: %ld, barType: %ld, barPtr: %ld\n",
    //     curCnt,
    //     groupStart,
    //     groupEnd,
    //     step,
    //     barType,
    //     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
    for (int64_t i = groupStart; i < groupEnd; i += step) {
        int64_t cnt = 0;
        while (ptrInt64[i] < curCnt) {
            // printf("i: %ld, v: %ld, curCnt: %ld\n", i, ptrInt64[i], curCnt);
            YIELDING;
            MB;
            // cnt += 1;
            // if (cnt > 500000000) {

            //     int tmp_idx = 29 * (mTid * nThreadNum + nTid + 1) + ptr2cnt * 4;
            //     printf("curCnt: %ld, mTid: %ld, nTid: %ld, step: %ld, barType: %ld, barPtr: %ld, "
            //            "dataReadyCnt1: %ld, ptr1: %ld, dataReadyCnt2: %ld, ptr2: %ld, mNum: %ld, nNum: %ld, bar4:
            //            %ld, " "bar5: %ld, " "bar6: %ld, bar7: %ld\n",
            //         curCnt,
            //         mTid,
            //         nTid,
            //         step,
            //         barType,
            //         static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)),
            //         ptr2Int64[mTid * nThreadNum + nTid],
            //         static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dataReady)),
            //         ptr3Int64[mTid * nThreadNum + nTid],
            //         static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dataReady1)),
            //         mThreadNum,
            //         nThreadNum,
            //         ptr2Int64[tmp_idx + 0],
            //         ptr2Int64[tmp_idx + 1],
            //         ptr2Int64[tmp_idx + 2],
            //         ptr2Int64[tmp_idx + 3]);

            //     std::string out = "BarrierWait: ";
            //     for (int64_t i = 0; i < 16; i += 1) {
            //         out += (std::to_string(ptrInt64[i]) + " ");
            //     }
            //     out += std::to_string(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
            //     std::cout << out << std::endl;
            //     int64_t tmp_cnt = 500000000;
            //     while (tmp_cnt > 0) {
            //         YIELDING;
            //         tmp_cnt--;
            //     }
            //     std::cout << "###############################" << std::endl;
            //     out = "BarrierWait: ";
            //     for (int64_t i = 0; i < 16; i += 1) {
            //         out += (std::to_string(ptrInt64[i]) + " ");
            //     }
            //     out += std::to_string(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
            //     std::cout << out << std::endl;
            //     while (1) {
            //     }
            // }
        }
    }
    // int64_t idx = 29 * (mTid * nThreadNum + nTid + 1) + barType - 4 + (ptr2cnt * 4);
    // // std::cout << "!!" << ptr2cnt <<" " << idx << " ***" << std::endl;
    // ptr2Int64[idx] = curCnt;
    // printf("out bar!\n");
    return 0;
}

// 77 64 64
// 64 64 64 13
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t SyncToOneCore(
    void *ptr, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum, int64_t barType)
{
    // printf("enter SyncToOneCore bar!\n");
    volatile int64_t *ptrInt64 = reinterpret_cast<volatile int64_t *>(ptr);
    ptrInt64[mTid * nThreadNum + nTid]++;
    WMB;
    int64_t curCnt = ptrInt64[mTid * nThreadNum + nTid];

    int64_t groupStart = 0;

    if (barType == 0) {
        groupStart = mTid * nThreadNum;
    } else if (barType == 1) {
        groupStart = nTid;
    }
    // printf("SyncToOneCore, curCnt: %ld, groupStart: %ld, barType: %ld, barPtr: %ld, mTid: %ld, nTid: %ld, "
    //                "mThreadNum: %ld, "
    //                "nThreadNum: %ld\n",
    //             curCnt,
    //             groupStart,
    //             barType,
    //             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)),
    //             mTid,
    //             nTid,
    //             mThreadNum,
    //             nThreadNum);
    int cnt = 0;
    while (ptrInt64[groupStart] < curCnt) {
        YIELDING;
        MB;
        cnt += 1;
        if (cnt > 500000000) {
            printf("SyncToOneCore hang! curCnt: %ld, groupStart: %ld, barType: %ld, barPtr: %ld, mTid: %ld, nTid: %ld, "
                   "mThreadNum: %ld, "
                   "nThreadNum: %ld\n",
                curCnt,
                groupStart,
                barType,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)),
                mTid,
                nTid,
                mThreadNum,
                nThreadNum);
            while (1) {
            }
        }
    }
    // printf("out SyncToOneCore bar!\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t SetSyncVal(
    void *ptr, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum, int64_t val)
{
    volatile int64_t *ptrInt64 = reinterpret_cast<volatile int64_t *>(ptr);
    ptrInt64[mTid * nThreadNum + nTid] = val;
    MB;
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t ResetSyncLevel1(void *ptrBarCompute, void *ptrBarDataReady,
    void *ptrBarAssist, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum)
{
    // printf("enter ResetSyncLevel1\n");
    // return 0;
    // volatile int64_t *barCompute = reinterpret_cast<volatile int64_t *>(ptrBarCompute);
    // volatile int64_t *barDataReady = reinterpret_cast<volatile int64_t *>(ptrBarDataReady);
    // volatile int64_t *ptrBarAssist = reinterpret_cast<volatile int64_t *>(ptrBarAssist);
    // std::cout << "ResetSyncLevel1 ptrBarDataReady bar ptr: " <<
    // static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptrBarDataReady)) << std::endl;
    SetSyncVal(ptrBarCompute, mTid, nTid, mThreadNum, nThreadNum, TASK_STATE_LOOP_DONE);
    // SetSyncVal(ptrBarDataReady, mTid, nTid, mThreadNum, nThreadNum, TASK_STATE_LOOP_DONE);
    // if (mTid == 0 && nTid == 0) {
    //     std::cout<<"ResetSyncLevel1: 1 ptrBarAssist" << std::endl;
    // }
    BarrierWait(ptrBarAssist, mTid, nTid, mThreadNum, nThreadNum, 3);

    SetSyncVal(ptrBarCompute, mTid, nTid, mThreadNum, nThreadNum, 0);
    SetSyncVal(ptrBarDataReady, mTid, nTid, mThreadNum, nThreadNum, 0);
    // if (mTid == 0 && nTid == 0) {
    //     std::cout<<"ResetSyncLevel1: 2 ptrBarAssist" << std::endl;
    // }
    BarrierWait(ptrBarAssist, mTid, nTid, mThreadNum, nThreadNum, 3);
    WMB;
    // printf("out ResetSyncLevel1\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT int64_t ResetSyncLevel2(void *ptrBarCompute, void *ptrBarDataReady1,
    void *ptrBarDataReady2, void *ptrBarAssist, int64_t mTid, int64_t nTid, int64_t mThreadNum, int64_t nThreadNum)
{
    // volatile int64_t *barCompute = reinterpret_cast<volatile int64_t *>(ptrBarCompute);
    // volatile int64_t *barDataReady1 = reinterpret_cast<volatile int64_t *>(ptrBarDataReady1);
    // volatile int64_t *barDataReady2 = reinterpret_cast<volatile int64_t *>(ptrBarDataReady2);
    // volatile int64_t *barAssist = reinterpret_cast<volatile int64_t *>(ptrBarAssist);
    // std::cout << "ResetSyncLevel2 ptrBarDataReady1 bar ptr: " <<
    // static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptrBarDataReady1)) << std::endl; std::cout << "ResetSyncLevel2
    // ptrBarDataReady2 bar ptr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptrBarDataReady2)) << std::endl;
    // return 0;
    // printf("Enter ResetSyncLevel2\n");
    SetSyncVal(ptrBarAssist, mTid, nTid, mThreadNum, nThreadNum, TASK_STATE_LOOP_DONE + 1);  // 9990
    // SetSyncVal(ptrBarDataReady1, mTid, nTid, mThreadNum, nThreadNum, TASK_STATE_LOOP_DONE + 1);
    // SetSyncVal(ptrBarDataReady2, mTid, nTid, mThreadNum, nThreadNum, TASK_STATE_LOOP_DONE + 1);
    SetSyncVal(ptrBarCompute, mTid, nTid, mThreadNum, nThreadNum, TASK_STATE_LOOP_DONE + 1);
    // if (mTid == 0 && nTid == 0) {
    //     std::cout<<"ResetSyncLevel2: 1 ptrBarCompute" << std::endl;
    // }
    BarrierWait(ptrBarCompute, mTid, nTid, mThreadNum, nThreadNum, 4);

    SetSyncVal(ptrBarAssist, mTid, nTid, mThreadNum, nThreadNum, 0);  // 0号线程 set没有更新L1
    // if (mTid == 0 && nTid == 0) {
    //     std::cout<<"ResetSyncLevel2: 2 ptrBarCompute" << std::endl;
    // }
    BarrierWait(ptrBarCompute, mTid, nTid, mThreadNum, nThreadNum, 5);  // 0号线程还在这边查询，其他线程都过了
    // if (mTid == 0 && nTid == 0) {
    //     std::cout<<"ResetSyncLevel2: 3 ptrBarAssist" << std::endl;
    // }
    BarrierWait(ptrBarAssist, mTid, nTid, mThreadNum, nThreadNum, 6);  // 9999
    SetSyncVal(ptrBarCompute, mTid, nTid, mThreadNum, nThreadNum, 0);  // 某个线程置为0, 刷到L1
    SetSyncVal(ptrBarDataReady1, mTid, nTid, mThreadNum, nThreadNum, 0);
    SetSyncVal(ptrBarDataReady2, mTid, nTid, mThreadNum, nThreadNum, 0);
    // if (mTid == 0 && nTid == 0) {
    //     std::cout<<"ResetSyncLevel2: 4 ptrBarAssist" << std::endl;
    // }
    BarrierWait(ptrBarAssist, mTid, nTid, mThreadNum, nThreadNum, 7);  //
                                                                       // 6 7 9
    WMB;
    // printf("Out ResetSyncLevel2\n");
    return 0;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_incopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("sgemm_incopy_wrap:  kstep: %ld, mstep: %ld, lda: %ld\n", size0, size1, srcStride0);
    // printf("sgemm_incopy_wrap:  kstep: %ld, mstep: %ld, lda: %ld\n",
    //     size0,
    //     size1,
    //     srcStride0);
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtr)) << std::endl;
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtr);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;

    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);

    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;
    sgemm_incopy(size0, size1, srcAlignPtrFloat, srcStride1, dstAlignPtrFloat);
    // std::cout << "end sgemm_itcopy" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_itcopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{

    // printf("sgemm_itcopy_wrap:  kstep: %ld, mstep: %ld, lda: %ld,srcOffset:%ld, dstOffset:%ld,  offset0: %ld,
    // offset1: "
    //        "%ld \n",
    //     size0,
    //     size1,
    //     srcStride0,
    //     srcOffset,
    //     dstOffset,
    //     offset0,
    //     offset1);
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtr);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtr);
    srcAlignPtrFloat += (offset1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat += dstOffset;

    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;

    sgemm_itcopy(size0, size1, srcAlignPtrFloat, srcStride0, dstAlignPtrFloat);
    // std::cout << "end sgemm_itcopy" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_oncopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("sgemm_oncopy_wrap:  offset0: %ld, offset1: %ld, size0: %ld, size1: %ld, srcStride0: %ld, srcStride1: %ld
    // "
    //        "srcOffset: %ld, dstOffset: %ld\n",
    //     offset0,
    //     offset1,
    //     size0,
    //     size1,
    //     srcStride0,
    //     srcStride1,
    //     srcOffset,
    //     dstOffset);
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtr);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtr);
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat += dstOffset;
    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;
    sgemm_oncopy(size0, size1, srcAlignPtrFloat, srcStride1, dstAlignPtrFloat);
    // std::cout << "end sgemm_otcopy" << std::endl;
    // 33
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_otcopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("sgemm_otcopy_wrap:  offset0: %ld, offset1: %ld, size0: %ld, size1: %ld, srcStride0: %ld, srcStride1: %ld
    // "
    //            "srcOffset: %ld, dstOffset: %ld\n",
    //         offset0,
    //         offset1,
    //         size0,
    //         size1,
    //         srcStride0,
    //         srcStride1,
    //         srcOffset,
    //         dstOffset);
    // printf("sgemm_otcopy_wrap:  offset0: %ld, offset1: %ld, size0: %ld, size1: %ld, srcStride0: %ld\n",
    //     offset0,
    //     offset1,
    //     size0,
    //     size1,
    //     srcStride0);
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtr)) << std::endl;
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtr);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;
    // srcAlignPtrChar += ((offset1 + offset0 * srcStride0) * (sizeof(float) / sizeof(char)));
    srcAlignPtrFloat += (offset1 + offset0 * srcStride0 + srcOffset);
    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrChar)) <<
    // std::endl;
    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;
    sgemm_otcopy(size0, size1, srcAlignPtrFloat, srcStride0, dstAlignPtrFloat);
    // std::cout << "end sgemm_otcopy" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_nt_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    // printf("sgemm_nt_wrap:  offset0: %ld, offset1: %ld, m: %ld, n: %ld, k: %ld, strideA1: %ld, strideB1: %ld, "
    //        "strideC1: %ld, strideC2: %ld, lda: %ld, ldb: %ld, ldc: %ld\n",
    //     offset0,
    //     offset1,
    //     m,
    //     n,
    //     k,
    //     strideA1,
    //     strideB1,
    //     strideC1,
    //     strideC2,
    //     lda,
    //     ldb,
    //     strideC2);
    // printf("sgemm_nt_wrap:  offset0: %ld, offset1: %ld, m: %ld, n: %ld, k: %ld, strideA1: %ld, strideB1: %ld, "
    //        "strideC1: %ld, strideC2: %ld, lda: %ld, ldb: %ld, ldc: %ld, offsetC: %ld, offsetA: %ld, offsetB: %ld\n",
    //     offset0,
    //     offset1,
    //     m,
    //     n,
    //     k,
    //     strideA1,
    //     strideB1,
    //     strideC1,
    //     strideC2,
    //     lda,
    //     ldb,
    //     strideC2,
    //     offsetC,
    //     offsetA,
    //     offsetB);
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    // mlir c matrix is rowmajor expression，while in
    // alignPtrCChar += ((offset0 + offset1 * strideC1) * (sizeof(float) / sizeof(char)));
    // alignPtrCChar += ((offset0 *strideC1 + offset1 * strideC2) * (sizeof(float) / sizeof(char)));
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrA)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrB)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCChar)) << std::endl;
    // // if (offset0 > 0) {
    //     return;
    // }
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
    // sgemm_kernel_nt(m, n, k, alignPtrAFloat, lda, alpha, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
    //  printf("alignPtrCChar[0]: %f\n", ((float *)alignPtrCChar)[0]);
    //  std::cout << "end sgemm_kernel_nt" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_nn_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    // lda = 128;
    // ldb = 128;
    // strideC2 = 128;
    // printf("sgemm_nn_wrap:  offset0: %ld, offset1: %ld, m: %ld, n: %ld, k: %ld, strideA1: %ld, strideB1: %ld, "
    //        "strideC1: %ld, strideC2: %ld, lda: %ld, ldb: %ld, ldc: %ld, offsetC: %ld, offsetA: %ld, offsetB: %ld,
    //        alpha: %f\n",
    //     offset0,
    //     offset1,
    //     m,
    //     n,
    //     k,
    //     strideA1,
    //     strideB1,
    //     strideC1,
    //     strideC2,
    //     lda,
    //     ldb,
    //     strideC2,
    //     offsetC,
    //     offsetA,
    //     offsetB,
    //     alpha);

    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
    // if (m < 128 || n < 128) {
    //     return;
    // }
    // // if (offset0 > 0) {
    //     return;
    // }
    // printf("before A[0]: %f, B[0]: %f, C[0]: %f\n", reinterpret_cast<float *>(alignPtrA)[0],reinterpret_cast<float
    // *>(alignPtrB)[0], alignPtrCFloat[0] );
    sgemm_kernel_nn(m, n, k, alignPtrAFloat, lda, alpha, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
    // printf("A[0]: %f, B[0]: %f, C[0]: %f\n", alignPtrAFloat[0],alignPtrBFloat[0], alignPtrCFloat[0] );
    // printf("alignPtrCChar[0]: %f\n", alignPtrCFloat[0]);
    // std::cout<< "end sgemm_nn_wrap" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_tn_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    // printf("sgemm_tn_wrap:  offset0: %ld, offset1: %ld, m: %ld, n: %ld, k: %ld, strideA1: %ld, strideB1: %ld, "
    //        "strideC1: %ld, strideC2: %ld, lda: %ld, ldb: %ld, ldc: %ld, offsetC: %ld, offsetA: %ld, offsetB: %ld\n",
    //     offset0,
    //     offset1,
    //     m,
    //     n,
    //     k,
    //     strideA1,
    //     strideB1,
    //     strideC1,
    //     strideC2,
    //     lda,
    //     ldb,
    //     strideC2,
    //     offsetC,
    //     offsetA,
    //     offsetB);

    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    // mlir c matrix is rowmajor expression，while in
    // alignPtrCChar += ((offset0 + offset1 * strideC1) * (sizeof(float) / sizeof(char)));
    // alignPtrCChar += ((offset0 *strideC1 + offset1 * strideC2) * (sizeof(float) / sizeof(char)));
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrA)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrB)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCChar)) << std::endl;
    // // if (offset0 > 0) {
    //     return;
    // }
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
    sgemm_kernel_tn(m, n, k, alignPtrAFloat, lda, alpha, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
    // printf("alignPtrCChar[0]: %f\n", ((float *)alignPtrCChar)[0]);
    // std::cout << "end sgemm_kernel_nt" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sgemm_tt_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    //    printf("sgemm_tt_wrap:  offset0: %ld, offset1: %ld, m: %ld, n: %ld, k: %ld, strideA1: %ld, strideB1: %ld, "
    //            "strideC1: %ld, strideC2: %ld, lda: %ld, ldb: %ld, ldc: %ld, offsetC: %ld, offsetA: %ld, offsetB:
    //            %ld\n",
    //         offset0,
    //         offset1,
    //         m,
    //         n,
    //         k,
    //         strideA1,
    //         strideB1,
    //         strideC1,
    //         strideC2,
    //         lda,
    //         ldb,
    //         strideC2,
    //         offsetC,
    //         offsetA,
    //         offsetB);
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);

    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
    sgemm_kernel_tt(m, n, k, alignPtrAFloat, lda, alpha, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
    // printf("alignPtrCChar[0]: %f\n", ((float *)alignPtrCChar)[0]);
    // std::cout << "end sgemm_kernel_nt" << std::endl;
    return;
}

// extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void small_sgemm_nn_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
//     int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
//     int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
//     void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
//     int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
// {
//     float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
//     float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
//     float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
//     alignPtrAFloat += offsetA;
//     alignPtrBFloat += offsetB;
//     alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
//     // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
//     // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
//     // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
//     // std::cout << "lda: " << lda << ", ldb: " << ldb << ", ldc: " << strideC2 << std::endl;
//     sgemm_small_kernel_b0_nn(
//         m, n, k, alignPtrAFloat, lda, alpha, alignPtrBFloat, ldb, alignPtrCFloat, strideC2);
// }

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sdma_sgemm_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2, float alpha)
{
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += offsetC;
    // alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
    // std::cout << "lda: " << lda << ", ldb: " << ldb << ", ldc: " << strideC2 << std::endl;
    // for (int i = 0; i < 1000; i++) {
    //     YIELDING;
    // }

    // printf("lda: %ld, ldb: %ld, ldc: %ld, m: %ld, n: %ld, k: %ld \n", strideA1, strideB1, strideC2, sizeC1, sizeC2,
    // sizeA1); printf("A: %f, %f", alignPtrAFloat[0], alignPtrAFloat[strideA1]); printf("B: %f, %f",
    // alignPtrBFloat[0+32], alignPtrBFloat[strideA1+32]); printf("A: %f, %f", alignPtrAFloat[0], alignPtrAFloat[32]);
    // printf("B: %f, %f", alignPtrBFloat[64], alignPtrBFloat[96]);
    sgemm_kernel_npa_npb(sizeC1,
        sizeC2,
        sizeA1,
        alignPtrAFloat,
        strideA1,
        alpha,
        alignPtrBFloat,
        strideB1,
        alignPtrCFloat,
        strideC2,
        nullptr,
        16);
    // A: -0.570837, -1.186221
    // B: -0.546275, 1.198876
    // A: -1.186221, -0.829297
    // B: 0.608079, 1.198876
    // -0.546275 * -1.186221 + 1.198876 * -0.829297
    // -1.186221*0.608079+-0.829297*1.198876
    /*
    -0.546275 * -1.186221 + 1.198876 * -0.829297
     A: -0.110058, -0.429407
     B: 0.888066, 0.171210
     A: -0.110058, -2.071593
     B: -1.703914, 0.171210
    */

    // sgemm_kernel_npa_npb(sizeC2,
    //     sizeC1,
    //     sizeA1,
    //     alignPtrAFloat,
    //     sizeC1,
    //     alpha,
    //     alignPtrBFloat,
    //     sizeC1,
    //     alignPtrCFloat,
    //     sizeC1,
    //     nullptr,
    //     0);
    // for (int i = 0; i < 1; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         int tmp = i + j * strideC1;
    //         printf("gemm: %d, %f\n", j, alignPtrCFloat[tmp]);
    //     }
    // }
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sdma_cgemm_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2, float alpha0,
    float alpha1)
{
    /*float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += offsetC;
*/
    std::complex<float> *alignPtrAFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrA);
    std::complex<float> *alignPtrBFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrB);
    std::complex<float> *alignPtrCFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrC);
    alignPtrAFloat_complex += offsetA;
    alignPtrBFloat_complex += offsetB;
    alignPtrCFloat_complex += offsetC;
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrAFloat_complex);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrBFloat_complex);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrCFloat_complex);
    cgemm_kernel_npa_npb(sizeC1,
        sizeC2,
        sizeA1,
        alignPtrAFloat,
        strideA1,
        alpha0,
        alpha1,
        alignPtrBFloat,
        strideB1,
        alignPtrCFloat,
        strideC2,
        nullptr,
        16);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sdma_dgemm_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2, double alpha,
    int64_t mTid, int64_t nTid)
{
    double *alignPtrAFloat = reinterpret_cast<double *>(alignPtrA);
    double *alignPtrBFloat = reinterpret_cast<double *>(alignPtrB);
    double *alignPtrCFloat = reinterpret_cast<double *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += offsetC;
    // // alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
    // std::cout << "lda: " << lda << ", ldb: " << ldb << ", ldc: " << strideC2 << std::endl;
    // for (int i = 0; i < 1000; i++) {
    //     YIELDING;
    // }

    // printf("lda: %ld, ldb: %ld, ldc: %ld, m: %ld, n: %ld, k: %ld \n", strideA1, strideB1, strideC2, sizeC1, sizeC2,
    // sizeA1);
    if (alpha - 1.0f < 1e-5) {
        dgemm_kernel_npa_npb_alpha1(sizeC1,
            sizeC2,
            sizeA1,
            alignPtrAFloat,
            strideA1,
            alpha,
            alignPtrBFloat,
            strideB1,
            alignPtrCFloat,
            strideC2,
            nullptr,
            32);
    } else {
        dgemm_kernel_npa_npb(sizeC1,
            sizeC2,
            sizeA1,
            alignPtrAFloat,
            strideA1,
            alpha,
            alignPtrBFloat,
            strideB1,
            alignPtrCFloat,
            strideC2,
            nullptr,
            32);
    }

    // for (int i = 0; i < 1; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         int tmp = i + j * strideC1;
    //         printf("gemm: %d, %lf\n", j, alignPtrCFloat[tmp]);
    //     }
    // }
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void *PrintTidAndSdmaInfoPtr(
    int64_t mTid, int64_t nTid, int64_t groupId, void *sdmaPtr)
{
    // printf("PrintTidAndSdmaInfoPtr mTid: %ld, nTid: %ld, groupId: %ld, sdmaPtr: %ld \n",
    //     mTid,
    //     nTid,
    //     groupId,
    //     static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sdmaPtr)));
    return nullptr;
}

/************************BGEMM ******************************/

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void bgemm_incopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("sgemm_incopy_wrap:  kstep: %ld, mstep: %ld, lda: %ld\n", size0, size1, srcStride0);
    // printf("sgemm_incopy_wrap:  kstep: %ld, mstep: %ld, lda: %ld\n",
    //     size0,
    //     size1,
    //     srcStride0);
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtr)) << std::endl;
    __bf16 *srcAlignPtrFloat = reinterpret_cast<__bf16 *>(srcAlignPtr);
    __bf16 *dstAlignPtrFloat = reinterpret_cast<__bf16 *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;

    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);

    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;
    bgemm_incopy(size0, size1, srcAlignPtrFloat, srcStride1, dstAlignPtrFloat);
    // std::cout << "end sgemm_itcopy" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void bgemm_itcopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{

    // printf("sgemm_itcopy_wrap:  kstep: %ld, mstep: %ld, lda: %ld,srcOffset:%ld, dstOffset:%ld,  offset0: %ld,
    // offset1: "
    //        "%ld \n",
    //     size0,
    //     size1,
    //     srcStride0,
    //     srcOffset,
    //     dstOffset,
    //     offset0,
    //     offset1);
    __bf16 *srcAlignPtrFloat = reinterpret_cast<__bf16 *>(srcAlignPtr);
    __bf16 *dstAlignPtrFloat = reinterpret_cast<__bf16 *>(dstAlignPtr);
    srcAlignPtrFloat += (offset1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat += dstOffset;

    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;

    bgemm_itcopy(size0, size1, srcAlignPtrFloat, srcStride0, dstAlignPtrFloat);
    // std::cout << "end sgemm_itcopy" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void bgemm_oncopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("sgemm_oncopy_wrap:  offset0: %ld, offset1: %ld, size0: %ld, size1: %ld, srcStride0: %ld, srcStride1: %ld
    // "
    //        "srcOffset: %ld, dstOffset: %ld\n",
    //     offset0,
    //     offset1,
    //     size0,
    //     size1,
    //     srcStride0,
    //     srcStride1,
    //     srcOffset,
    //     dstOffset);
    __bf16 *srcAlignPtrFloat = reinterpret_cast<__bf16 *>(srcAlignPtr);
    __bf16 *dstAlignPtrFloat = reinterpret_cast<__bf16 *>(dstAlignPtr);
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat += dstOffset;
    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;
    bgemm_oncopy(size0, size1, srcAlignPtrFloat, srcStride1, dstAlignPtrFloat);
    // std::cout << "end sgemm_otcopy" << std::endl;
    // 33
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void bgemm_otcopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("sgemm_otcopy_wrap:  offset0: %ld, offset1: %ld, size0: %ld, size1: %ld, srcStride0: %ld, srcStride1: %ld
    // "
    //            "srcOffset: %ld, dstOffset: %ld\n",
    //         offset0,
    //         offset1,
    //         size0,
    //         size1,
    //         srcStride0,
    //         srcStride1,
    //         srcOffset,
    //         dstOffset);
    // printf("sgemm_otcopy_wrap:  offset0: %ld, offset1: %ld, size0: %ld, size1: %ld, srcStride0: %ld\n",
    //     offset0,
    //     offset1,
    //     size0,
    //     size1,
    //     srcStride0);
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtr)) << std::endl;
    __bf16 *srcAlignPtrFloat = reinterpret_cast<__bf16 *>(srcAlignPtr);
    __bf16 *dstAlignPtrFloat = reinterpret_cast<__bf16 *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;
    // srcAlignPtrChar += ((offset1 + offset0 * srcStride0) * (sizeof(float) / sizeof(char)));
    srcAlignPtrFloat += (offset1 + offset0 * srcStride0 + srcOffset);
    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrChar)) <<
    // std::endl;
    // std::cout << "srcAlignPtrChar: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(srcAlignPtrFloat))
    //           << std::endl;
    // std::cout << "dstAlignPtr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dstAlignPtrFloat)) <<
    // std::endl;
    bgemm_otcopy(size0, size1, srcAlignPtrFloat, srcStride0, dstAlignPtrFloat);
    // std::cout << "end sgemm_otcopy" << std::endl;
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void bgemm_nt_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, __bf16 alpha)
{

    __bf16 *alignPtrAFloat = reinterpret_cast<__bf16 *>(alignPtrA);
    __bf16 *alignPtrBFloat = reinterpret_cast<__bf16 *>(alignPtrB);
    __bf16 *alignPtrCFloat = reinterpret_cast<__bf16 *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);

    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void bgemm_nn_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, __bf16 alpha)
{

    __bf16 *alignPtrAFloat = reinterpret_cast<__bf16 *>(alignPtrA);
    __bf16 *alignPtrBFloat = reinterpret_cast<__bf16 *>(alignPtrB);
    __bf16 *alignPtrCFloat = reinterpret_cast<__bf16 *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    // std::cout << "a addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrAFloat)) << std::endl;
    // std::cout << "b addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrBFloat)) << std::endl;
    // std::cout << "c addr: " << static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alignPtrCFloat)) << std::endl;
    // if (m < 128 || n < 128) {
    //     return;
    // }
    // // if (offset0 > 0) {
    //     return;
    // }
    // printf("before A[0]: %f, B[0]: %f, C[0]: %f\n", reinterpret_cast<float *>(alignPtrA)[0],reinterpret_cast<float
    // *>(alignPtrB)[0], alignPtrCFloat[0] );
    bgemm_kernel_nn(m, n, k, alignPtrAFloat, lda, alpha, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
    // printf("A[0]: %f, B[0]: %f, C[0]: %f\n", alignPtrAFloat[0],alignPtrBFloat[0], alignPtrCFloat[0] );
    // printf("alignPtrCChar[0]: %f\n", alignPtrCFloat[0]);
    // std::cout<< "end sgemm_nn_wrap" << std::endl;
    return;
}

/************************CGEMM ******************************/

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_incopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("cgemm_incopy_wrap is called\n");
    std::complex<float> *srcAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(srcAlignPtr);
    std::complex<float> *dstAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(dstAlignPtr);
    srcAlignPtrFloat_complex += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat_complex += dstOffset;
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtrFloat_complex);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtrFloat_complex);
    cgemm_incopy(size0, size1, srcAlignPtrFloat, srcStride1, dstAlignPtrFloat);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_itcopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    std::complex<float> *srcAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(srcAlignPtr);
    std::complex<float> *dstAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(dstAlignPtr);
    srcAlignPtrFloat_complex += (offset1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat_complex += dstOffset;
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtrFloat_complex);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtrFloat_complex);
    cgemm_itcopy(size0, size1, srcAlignPtrFloat, srcStride0, dstAlignPtrFloat);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_oncopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    // printf("cgemm_nn_wrap is called\n");
    std::complex<float> *srcAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(srcAlignPtr);
    std::complex<float> *dstAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(dstAlignPtr);
    srcAlignPtrFloat_complex += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat_complex += dstOffset;
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtrFloat_complex);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtrFloat_complex);
    cgemm_oncopy(size0, size1, srcAlignPtrFloat, srcStride1, dstAlignPtrFloat);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_otcopy_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1)
{
    std::complex<float> *srcAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(srcAlignPtr);
    std::complex<float> *dstAlignPtrFloat_complex = reinterpret_cast<std::complex<float> *>(dstAlignPtr);
    srcAlignPtrFloat_complex += (offset1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat_complex += dstOffset;
    float *srcAlignPtrFloat = reinterpret_cast<float *>(srcAlignPtrFloat_complex);
    float *dstAlignPtrFloat = reinterpret_cast<float *>(dstAlignPtrFloat_complex);
    cgemm_otcopy(size0, size1, srcAlignPtrFloat, srcStride0, dstAlignPtrFloat);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_nn_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha0,
    float alpha1)
{
    // printf("cgemm_nn_wrap is called\n");
    std::complex<float> *alignPtrAFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrA);
    std::complex<float> *alignPtrBFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrB);
    std::complex<float> *alignPtrCFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrC);
    alignPtrAFloat_complex += offsetA;
    alignPtrBFloat_complex += offsetB;
    alignPtrCFloat_complex += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrAFloat_complex);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrBFloat_complex);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrCFloat_complex);
    cgemm_kernel_nn(
        m, n, k, alignPtrAFloat, lda, alpha0, alpha1, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_nt_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha0,
    float alpha1)
{
    std::complex<float> *alignPtrAFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrA);
    std::complex<float> *alignPtrBFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrB);
    std::complex<float> *alignPtrCFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrC);
    alignPtrAFloat_complex += offsetA;
    alignPtrBFloat_complex += offsetB;
    alignPtrCFloat_complex += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrAFloat_complex);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrBFloat_complex);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrCFloat_complex);
    // cgemm_kernel_nt(
    //   m, n, k, alignPtrAFloat, lda, alpha0, alpha1, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
}
/*
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void sdma_cgemm_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    float alpha0, float alpha1)
{
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrA);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrB);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += offsetC;

    cgemm_kernel_npa_npb(sizeC1,
        sizeC2,
        sizeA1,
        alignPtrAFloat,
        strideA1,
        alpha0, alpha1,
        alignPtrBFloat,
        strideB1,
        alignPtrCFloat,
        strideC2,
        nullptr,
        16);
}
*/
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_tn_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha0,
    float alpha1)
{
    std::complex<float> *alignPtrAFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrA);
    std::complex<float> *alignPtrBFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrB);
    std::complex<float> *alignPtrCFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrC);
    alignPtrAFloat_complex += offsetA;
    alignPtrBFloat_complex += offsetB;
    alignPtrCFloat_complex += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrAFloat_complex);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrBFloat_complex);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrCFloat_complex);
    cgemm_kernel_tn(
        m, n, k, alignPtrAFloat, lda, alpha0, alpha1, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void cgemm_tt_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha0,
    float alpha1)
{
    std::complex<float> *alignPtrAFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrA);
    std::complex<float> *alignPtrBFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrB);
    std::complex<float> *alignPtrCFloat_complex = reinterpret_cast<std::complex<float> *>(alignPtrC);
    alignPtrAFloat_complex += offsetA;
    alignPtrBFloat_complex += offsetB;
    alignPtrCFloat_complex += (offset0 * strideC1 + offset1 * strideC2 + offsetC);
    float *alignPtrAFloat = reinterpret_cast<float *>(alignPtrAFloat_complex);
    float *alignPtrBFloat = reinterpret_cast<float *>(alignPtrBFloat_complex);
    float *alignPtrCFloat = reinterpret_cast<float *>(alignPtrCFloat_complex);
    cgemm_kernel_tt(
        m, n, k, alignPtrAFloat, lda, alpha0, alpha1, alignPtrBFloat, ldb, alignPtrCFloat, strideC2, nullptr);
}

// SME常量
const uint64_t TILE_IDX_ZERO = 0;
const uint64_t TILE_IDX_ONE = 1;
const uint64_t TILE_IDX_TWO = 2;
const uint64_t TILE_IDX_THREE = 3;

#define KERNEL_M_STEP_SME 16
#define KERNEL_N_STEP_SME 16
#define KERNEL_SME_VL 64
#define KERNEL_K_STEP_SME 64
// 假设a和b都是列主序为N，行主序为T，packing沿k方向进行

template <typename T>
__arm_new("za") void pack_a_sme_n_v1(void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size,
    size_t lda, const T oa) __arm_streaming
{
    T *bufferA_typed = (T *)bufferA;
    T *curr_a_ptr_typed = (T *)curr_a_ptr;
    size_t k_block_size_up = (k_block_size + KERNEL_K_STEP_SME - 1) / KERNEL_K_STEP_SME * KERNEL_K_STEP_SME;
    size_t k_portions = k_block_size / KERNEL_K_STEP_SME;
    size_t k_resid = k_block_size - KERNEL_K_STEP_SME * k_portions;

    size_t m_portions = m_block_size / KERNEL_SME_VL;
    size_t m_resid = m_block_size - KERNEL_SME_VL * m_portions;

    svbool_t mask_p64_b8 = svptrue_pat_b8(SV_ALL);
    for (size_t im64 = 0; im64 < m_portions; ++im64) {
        // 沿着M方向读取，一次 KERNEL_M_SME_VL 个int8
        // M 按照 KERNEL_M_STEP_SME 一组 packing，所以packing分为 KERNEL_M_SME_VL/KERNEL_M_STEP_SME 个sub tile
        size_t sub_tile_inner_offset = 0;  // sub tile内是连续的，packed_a_offset用于控制地址偏移
        for (size_t ik64 = 0; ik64 < k_portions; ++ik64) {
            for (size_t ik = 0; ik < KERNEL_SME_VL; ++ik) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    ik,
                    mask_p64_b8,
                    curr_a_ptr_typed + (ik64 * KERNEL_K_STEP_SME + ik) * lda + (im64 * KERNEL_SME_VL));
            }

            for (size_t im = 0; im < KERNEL_SME_VL; im += KERNEL_M_STEP_SME) {
                for (size_t jm = 0; jm < KERNEL_M_STEP_SME; ++jm) {
                    svst1_hor_za8(TILE_IDX_ZERO,
                        im + jm,  // im + jm: za内的行index
                        mask_p64_b8,
                        bufferA_typed + (im + im64 * KERNEL_SME_VL) * k_block_size_up + sub_tile_inner_offset +
                            KERNEL_SME_VL * jm);
                }
            }
            sub_tile_inner_offset += KERNEL_SME_VL * KERNEL_M_STEP_SME;
        }
        if (k_resid) {
            svzero_za();
            for (size_t ik = 0; ik < k_resid; ++ik) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    ik,
                    mask_p64_b8,  // 沿m方向读取仍有完整的VL
                    curr_a_ptr_typed + (k_portions * KERNEL_K_STEP_SME + ik) * lda + (im64 * KERNEL_SME_VL));
            }

            for (size_t im = 0; im < KERNEL_SME_VL; im += KERNEL_M_STEP_SME) {
                for (size_t jm = 0; jm < KERNEL_M_STEP_SME; ++jm) {
                    svst1_hor_za8(TILE_IDX_ZERO,
                        im + jm,      // im + jm: za内的行index
                        mask_p64_b8,  // k方向上会进行padding
                        bufferA_typed + (im + im64 * KERNEL_SME_VL) * k_block_size_up + sub_tile_inner_offset +
                            KERNEL_SME_VL * jm);
                }
            }
        }
    }

    if (m_resid) {
        svbool_t pg = svwhilelt_b8((uint64_t)0, m_resid);  // m方向不足vl
        for (size_t ik64 = 0; ik64 < k_portions; ++ik64) {
            svzero_za();
            for (size_t ik = 0; ik < KERNEL_SME_VL; ++ik) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    ik,
                    pg,  // 沿m方向不足VL
                    curr_a_ptr_typed + (ik64 * KERNEL_K_STEP_SME + ik) * lda + (m_portions * KERNEL_SME_VL));
            }

            size_t sub_m_portions = m_resid / KERNEL_M_STEP_SME;
            size_t sub_m_resid = m_resid - sub_m_portions * KERNEL_M_STEP_SME;

            for (size_t im = 0; im < sub_m_portions; ++im) {
                for (size_t jm = 0; jm < KERNEL_M_STEP_SME; ++jm) {
                    svst1_hor_za8(TILE_IDX_ZERO,
                        im * KERNEL_M_STEP_SME + jm,  // im + jm: za内的行index
                        mask_p64_b8,                  // 沿k方向读取仍有完整的VL
                        bufferA_typed + (im * KERNEL_M_STEP_SME + m_portions * KERNEL_SME_VL) * k_block_size_up +
                            ik64 * KERNEL_SME_VL * KERNEL_M_STEP_SME + KERNEL_SME_VL * jm);
                }
            }
            if (sub_m_resid) {
                for (size_t jm = 0; jm < sub_m_resid; ++jm) {
                    svst1_hor_za8(TILE_IDX_ZERO,
                        sub_m_portions * KERNEL_M_STEP_SME + jm,  // im + jm: za内的行index
                        mask_p64_b8,                              // 沿k方向读取仍有完整的VL
                        bufferA_typed +
                            (sub_m_portions * KERNEL_M_STEP_SME + m_portions * KERNEL_SME_VL) * k_block_size_up +
                            ik64 * KERNEL_SME_VL * sub_m_resid + KERNEL_SME_VL * jm);
                }
            }
        }
        if (k_resid) {
            svzero_za();
            for (size_t ik = 0; ik < k_resid; ++ik) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    ik,
                    pg,
                    curr_a_ptr_typed + (k_portions * KERNEL_K_STEP_SME + ik) * lda + (m_portions * KERNEL_SME_VL));
            }

            size_t sub_m_portions = m_resid / KERNEL_M_STEP_SME;
            size_t sub_m_resid = m_resid - sub_m_portions * KERNEL_M_STEP_SME;

            for (size_t im = 0; im < sub_m_portions; ++im) {
                for (size_t jm = 0; jm < KERNEL_M_STEP_SME; ++jm) {
                    svst1_hor_za8(TILE_IDX_ZERO,
                        im * KERNEL_M_STEP_SME + jm,
                        mask_p64_b8,
                        bufferA_typed + (im * KERNEL_M_STEP_SME + m_portions * KERNEL_SME_VL) * k_block_size_up +
                            k_portions * KERNEL_SME_VL * KERNEL_M_STEP_SME + KERNEL_SME_VL * jm);
                }
            }
            if (sub_m_resid) {
                for (size_t jm = 0; jm < sub_m_resid; ++jm) {
                    svst1_hor_za8(TILE_IDX_ZERO,
                        sub_m_portions * KERNEL_M_STEP_SME + jm,
                        mask_p64_b8,
                        bufferA_typed +
                            (sub_m_portions * KERNEL_M_STEP_SME + m_portions * KERNEL_SME_VL) * k_block_size_up +
                            k_portions * KERNEL_SME_VL * sub_m_resid + KERNEL_SME_VL * jm);
                }
            }
        }
    }
}

template <typename T>
__arm_new("za") void pack_b_sme_n_v1(void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size,
    size_t ldb, const T ob) __arm_streaming
{
    T *bufferB_typed = (T *)bufferB;
    T *curr_b_ptr_typed = (T *)curr_b_ptr;
    size_t k_block_size_up = (k_block_size + KERNEL_K_STEP_SME - 1) / KERNEL_K_STEP_SME * KERNEL_K_STEP_SME;
    size_t k_portions = k_block_size / KERNEL_K_STEP_SME;
    size_t k_resid = k_block_size - KERNEL_K_STEP_SME * k_portions;

    size_t n_portions = n_block_size / KERNEL_N_STEP_SME;
    size_t n_resid = n_block_size - KERNEL_N_STEP_SME * n_portions;

    svbool_t mask_p64_b8 = svptrue_pat_b8(SV_ALL);
    size_t inner_offset = 0;
    for (size_t in16 = 0; in16 < n_portions; ++in16) {
        for (size_t ik64 = 0; ik64 < k_portions; ++ik64) {
            for (size_t in = 0; in < KERNEL_N_STEP_SME; ++in) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    in,
                    mask_p64_b8,
                    curr_b_ptr_typed + (in16 * KERNEL_N_STEP_SME + in) * ldb + (ik64 * KERNEL_SME_VL));
            }
            for (size_t in = 0; in < KERNEL_N_STEP_SME; ++in) {
                svst1_ver_za8(TILE_IDX_ZERO, in, mask_p64_b8, bufferB_typed + inner_offset);
                inner_offset += KERNEL_SME_VL;
            }
        }
        if (k_resid) {
            svzero_za();
            svbool_t pg = svwhilelt_b8((uint64_t)0, k_resid);
            for (size_t in = 0; in < KERNEL_N_STEP_SME; ++in) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    in,
                    pg,
                    curr_b_ptr_typed + (in16 * KERNEL_N_STEP_SME + in) * ldb + (k_portions * KERNEL_SME_VL));
            }
            for (size_t in = 0; in < KERNEL_N_STEP_SME; ++in) {
                svst1_ver_za8(TILE_IDX_ZERO, in, mask_p64_b8, bufferB_typed + inner_offset);
                inner_offset += KERNEL_SME_VL;
            }
        }
    }
    if (n_resid) {
        for (size_t ik64 = 0; ik64 < k_portions; ++ik64) {
            for (size_t in = 0; in < n_resid; ++in) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    in,
                    mask_p64_b8,
                    curr_b_ptr_typed + (n_portions * KERNEL_N_STEP_SME + in) * ldb + (ik64 * KERNEL_SME_VL));
            }
            for (size_t in = 0; in < n_resid; ++in) {
                svst1_ver_za8(TILE_IDX_ZERO, in, mask_p64_b8, bufferB_typed + inner_offset);
                inner_offset += KERNEL_SME_VL;
            }
        }
        if (k_resid) {
            svzero_za();
            svbool_t pg = svwhilelt_b8((uint64_t)0, k_resid);
            for (size_t in = 0; in < n_resid; ++in) {
                svld1_ver_za8(TILE_IDX_ZERO,
                    in,
                    pg,
                    curr_b_ptr_typed + (n_portions * KERNEL_N_STEP_SME + in) * ldb + (k_portions * KERNEL_SME_VL));
            }
            for (size_t in = 0; in < n_resid; ++in) {
                svst1_ver_za8(TILE_IDX_ZERO, in, mask_p64_b8, bufferB_typed + inner_offset);
                inner_offset += KERNEL_SME_VL;
            }
        }
    }
}

template <typename T>
__arm_new("za") void pack_a_sme_t_v1(void *bufferA, const void *curr_a_ptr, size_t m_block_size, size_t k_block_size,
    size_t lda, const T oa) __arm_streaming
{
    pack_b_sme_n_v1(bufferA, curr_a_ptr, m_block_size, k_block_size, lda, oa);
}

template <typename T>
__arm_new("za") void pack_b_sme_t_v1(void *bufferB, const void *curr_b_ptr, size_t n_block_size, size_t k_block_size,
    size_t ldb, const T ob) __arm_streaming
{
    pack_a_sme_n_v1(bufferB, curr_b_ptr, n_block_size, k_block_size, ldb, ob);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_a_u8_n_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, uint8_t oa)
{
    uint8_t *srcAlignPtrFloat = reinterpret_cast<uint8_t *>(srcAlignPtr);
    uint8_t *dstAlignPtrFloat = reinterpret_cast<uint8_t *>(dstAlignPtr);
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat += dstOffset;
    pack_a_sme_n_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride0, oa);
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_a_s8_n_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, int8_t oa)
{
    int8_t *srcAlignPtrFloat = reinterpret_cast<int8_t *>(srcAlignPtr);
    int8_t *dstAlignPtrFloat = reinterpret_cast<int8_t *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    pack_a_sme_n_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride0, oa);
    return;
}
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_a_u8_t_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, uint8_t oa)
{
    uint8_t *srcAlignPtrFloat = reinterpret_cast<uint8_t *>(srcAlignPtr);
    uint8_t *dstAlignPtrFloat = reinterpret_cast<uint8_t *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    pack_a_sme_t_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride1, oa);
    return;
}
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_a_s8_t_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, int8_t oa)
{
    int8_t *srcAlignPtrFloat = reinterpret_cast<int8_t *>(srcAlignPtr);
    int8_t *dstAlignPtrFloat = reinterpret_cast<int8_t *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    pack_a_sme_t_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride1, oa);
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_b_u8_n_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, uint8_t oa)
{
    uint8_t *srcAlignPtrFloat = reinterpret_cast<uint8_t *>(srcAlignPtr);
    uint8_t *dstAlignPtrFloat = reinterpret_cast<uint8_t *>(dstAlignPtr);
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat += dstOffset;
    pack_b_sme_n_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride1, oa);
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_b_s8_n_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, int8_t oa)
{
    int8_t *srcAlignPtrFloat = reinterpret_cast<int8_t *>(srcAlignPtr);
    int8_t *dstAlignPtrFloat = reinterpret_cast<int8_t *>(dstAlignPtr);
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    dstAlignPtrFloat += dstOffset;
    pack_b_sme_n_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride1, oa);
    return;
}
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_b_u8_t_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, int8_t oa)
{
    uint8_t *srcAlignPtrFloat = reinterpret_cast<uint8_t *>(srcAlignPtr);
    uint8_t *dstAlignPtrFloat = reinterpret_cast<uint8_t *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    pack_b_sme_t_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride0, oa);
    return;
}
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void pack_b_s8_t_wrap(void *srcAllocPtr, void *srcAlignPtr, int64_t srcOffset,
    int64_t srcSize0, int64_t srcSize1, int64_t srcStride0, int64_t srcStride1, void *dstAllocPtr, void *dstAlignPtr,
    int64_t dstOffset, int64_t dstSize0, int64_t dstSize1, int64_t dstStride0, int64_t dstStride1, int64_t offset0,
    int64_t offset1, int64_t size0, int64_t size1, int8_t oa)
{
    int8_t *srcAlignPtrFloat = reinterpret_cast<int8_t *>(srcAlignPtr);
    int8_t *dstAlignPtrFloat = reinterpret_cast<int8_t *>(dstAlignPtr);
    dstAlignPtrFloat += dstOffset;
    srcAlignPtrFloat += (offset1 * srcStride1 + offset0 * srcStride0 + srcOffset);
    pack_b_sme_t_v1(dstAlignPtrFloat, srcAlignPtrFloat, size1, size0, srcStride0, oa);
    return;
}

#define COMP_SV_LEN 512
#define KERNEL_M_STEP 4
#define KERNEL_N_STEP 4
#define KERNEL_K_STEP (COMP_SV_LEN / 8)

#define KERNEL_FUNC_NUM 16
#define SMALL_KERNEL_FUNC_NUM 12
#define BETA_FUNC_NUM 12
#define PACK_FUNC_NUM 2
#define POST_OP_FUNC_NUM 2
#define COMPUTE_IDX_NUM 2
#define MOVE_OC_NUM 3
#define KERNEL_SIZE_M_4 4
#define KERNEL_SIZE_M_3 3
#define KERNEL_SIZE_M_2 2
#define KERNEL_SIZE_N_4 4
#define KERNEL_SIZE_N_3 3
#define KERNEL_SIZE_N_2 2
#define ADD_M_N_SIZES(name, m_size, n_size) name##_##m_size##x##n_size
#define ADD_M_N_SIZES_MACRO(name, m_size, n_size) ADD_M_N_SIZES(name, m_size, n_size)
#define ADD_TYPES(name, lhs_type, rhs_type) name##_##lhs_type##8##rhs_type##8s32
#define ADD_TYPES_MACRO(name, lhs_type, rhs_type) ADD_TYPES(name, lhs_type, rhs_type)

#define ADD_KERNEL_SUFF(name, m_size, n_size) \
    ADD_M_N_SIZES_MACRO(ADD_TYPES_MACRO(name, LHS_TYPE, RHS_TYPE), m_size, n_size)

#define ADD_KERNEL_SUFF_REV(name, m_size, n_size) \
    ADD_M_N_SIZES_MACRO(ADD_TYPES_MACRO(name, RHS_TYPE, LHS_TYPE), m_size, n_size)

#define LHS_TYPE s
#define RHS_TYPE s
static void (*gemmKernels0[KERNEL_FUNC_NUM])(const void *, const void *, int32_t *, size_t, int64_t, int64_t) = {
    // We have 16 kernels for one precision combination. Kernels differ in block size of C matrix they process.
    // They process minimal block if size 1x1 to maximum block of size 4x4. All the kernels are listed below.
    ADD_KERNEL_SUFF(gemm_kernel, 1, 1),
    ADD_KERNEL_SUFF(gemm_kernel, 1, KERNEL_SIZE_N_2),
    ADD_KERNEL_SUFF(gemm_kernel, 1, KERNEL_SIZE_N_3),
    ADD_KERNEL_SUFF(gemm_kernel, 1, KERNEL_SIZE_N_4),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_2, 1),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_2, KERNEL_SIZE_N_2),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_2, KERNEL_SIZE_N_3),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_2, KERNEL_SIZE_N_4),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_3, 1),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_3, KERNEL_SIZE_N_2),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_3, KERNEL_SIZE_N_3),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_3, KERNEL_SIZE_N_4),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_4, 1),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_4, KERNEL_SIZE_N_2),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_4, KERNEL_SIZE_N_3),
    ADD_KERNEL_SUFF(gemm_kernel, KERNEL_SIZE_M_4, KERNEL_SIZE_N_4)};

#define SME_KERNEL_K_STEP_V4 64
#define SME_KERNEL_M_BLOCK_V4 16
#define SME_KERNEL_N_BLOCK_V4 16

template <typename T>
inline auto svreinterpret_x8_u32(svuint32_t vec) __arm_in("za") __arm_streaming
{
    if constexpr (std::is_same_v<T, int8_t>) {
        return svreinterpret_s8_u32(vec);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        return svreinterpret_u8_u32(vec);
    } else {
        static_assert(std::is_same_v<T, int8_t> && !std::is_same_v<T, uint8_t>,
            "Unsupported type for svreinterpret_x8_u32. Only int8_t and uint8_t are allowed.");
    }
}

template <typename SVTYPEA, typename SVTYPEB>
inline void svmopa_za32_x8_m_tile0(svbool_t mask_i, svbool_t mask_j, SVTYPEA va, SVTYPEB vb)
    __arm_out("za") __arm_streaming
{
    if constexpr (std::is_same_v<SVTYPEA, svint8_t>) {
        if constexpr (std::is_same_v<SVTYPEB, svint8_t>) {
            svmopa_za32_s8_m(TILE_IDX_ZERO, mask_i, mask_i, va, vb);
        } else if constexpr (std::is_same_v<SVTYPEB, svuint8_t>) {
            svsumopa_za32_s8_m(TILE_IDX_ZERO, mask_i, mask_i, va, vb);
        } else {
            static_assert(std::is_same_v<SVTYPEB, svint8_t> && !std::is_same_v<SVTYPEB, svuint8_t>,
                "Unsupported type for svmopa_za32_x8_m. Only int8_t and uint8_t are allowed.");
        }
    } else if constexpr (std::is_same_v<SVTYPEA, svuint8_t>) {
        if constexpr (std::is_same_v<SVTYPEB, svint8_t>) {
            svusmopa_za32_u8_m(TILE_IDX_ZERO, mask_i, mask_i, va, vb);
        } else if constexpr (std::is_same_v<SVTYPEB, svuint8_t>) {
            svmopa_za32_u8_m(TILE_IDX_ZERO, mask_i, mask_i, va, vb);
        } else {
            static_assert(std::is_same_v<SVTYPEB, svint8_t> && !std::is_same_v<SVTYPEB, svuint8_t>,
                "Unsupported type for svmopa_za32_x8_m. Only int8_t and uint8_t are allowed.");
        }
    } else {
        static_assert(std::is_same_v<SVTYPEA, svint8_t> && !std::is_same_v<SVTYPEA, svuint8_t>,
            "Unsupported type for svmopa_za32_x8_m. Only int8_t and uint8_t are allowed.");
    }
}

template <typename TYPEA, typename TYPEB>
__arm_new("za") void gemm_kernel_sme_kernel_v4(
    TYPEA *a, TYPEB *b, int32_t *c, uint32_t ldc, uint32_t k_depth) __arm_streaming
{
    svbool_t p16_b32 = svptrue_pat_b32(SV_ALL);
    for (int i = 0; i < SME_KERNEL_M_BLOCK_V4; ++i) {
        svld1_ver_za32(TILE_IDX_ZERO, i, p16_b32, c + ldc * i);
    }

    svbool_t mask_i = svptrue_pat_b8(SV_VL64);
    TYPEA *current_a_ptr = a;
    TYPEB *current_b_ptr = b;
    for (size_t i = 0; i < k_depth; i += SME_KERNEL_K_STEP_V4) {
        for (int za_idx = 0; za_idx < SME_KERNEL_M_BLOCK_V4; ++za_idx) {
            svld1_hor_za32(TILE_IDX_ONE, za_idx, p16_b32, current_a_ptr);
            current_a_ptr += SME_KERNEL_K_STEP_V4;
        }
        for (int za_idx = 0; za_idx < SME_KERNEL_N_BLOCK_V4; ++za_idx) {
            svld1_hor_za32(TILE_IDX_TWO, za_idx, p16_b32, current_b_ptr);
            current_b_ptr += SME_KERNEL_K_STEP_V4;
        }
        svuint32_t va000 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_ONE, 0);
        svuint32_t va001 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_ONE, 1);
        svuint32_t va002 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_ONE, 2);
        svuint32_t va003 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_ONE, 3);
        svuint32_t va004 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_ONE, 4);
        svuint32_t va005 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_ONE, 5);
        svuint32_t va006 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_ONE, 6);
        svuint32_t va007 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_ONE, 7);
        auto va0 = svreinterpret_x8_u32<TYPEA>(va000);
        auto va1 = svreinterpret_x8_u32<TYPEA>(va001);
        auto va2 = svreinterpret_x8_u32<TYPEA>(va002);
        auto va3 = svreinterpret_x8_u32<TYPEA>(va003);
        auto va4 = svreinterpret_x8_u32<TYPEA>(va004);
        auto va5 = svreinterpret_x8_u32<TYPEA>(va005);
        auto va6 = svreinterpret_x8_u32<TYPEA>(va006);
        auto va7 = svreinterpret_x8_u32<TYPEA>(va007);

        svuint32_t va008 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_TWO, 0);
        svuint32_t va009 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_TWO, 1);
        svuint32_t va010 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_TWO, 2);
        svuint32_t va011 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_TWO, 3);
        svuint32_t va012 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_TWO, 4);
        svuint32_t va013 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_TWO, 5);
        svuint32_t va014 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_TWO, 6);
        svuint32_t va015 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_TWO, 7);
        auto vb0 = svreinterpret_x8_u32<TYPEB>(va008);
        auto vb1 = svreinterpret_x8_u32<TYPEB>(va009);
        auto vb2 = svreinterpret_x8_u32<TYPEB>(va010);
        auto vb3 = svreinterpret_x8_u32<TYPEB>(va011);
        auto vb4 = svreinterpret_x8_u32<TYPEB>(va012);
        auto vb5 = svreinterpret_x8_u32<TYPEB>(va013);
        auto vb6 = svreinterpret_x8_u32<TYPEB>(va014);
        auto vb7 = svreinterpret_x8_u32<TYPEB>(va015);

        svmopa_za32_x8_m_tile0(mask_i, mask_i, va0, vb0);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va1, vb1);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va2, vb2);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va3, vb3);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va4, vb4);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va5, vb5);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va6, vb6);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va7, vb7);

        va000 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_ONE, 8);
        va001 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_ONE, 9);
        va002 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_ONE, 10);
        va003 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_ONE, 11);
        va004 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_ONE, 12);
        va005 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_ONE, 13);
        va006 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_ONE, 14);
        va007 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_ONE, 15);
        va0 = svreinterpret_x8_u32<TYPEA>(va000);
        va1 = svreinterpret_x8_u32<TYPEA>(va001);
        va2 = svreinterpret_x8_u32<TYPEA>(va002);
        va3 = svreinterpret_x8_u32<TYPEA>(va003);
        va4 = svreinterpret_x8_u32<TYPEA>(va004);
        va5 = svreinterpret_x8_u32<TYPEA>(va005);
        va6 = svreinterpret_x8_u32<TYPEA>(va006);
        va7 = svreinterpret_x8_u32<TYPEA>(va007);

        va008 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_TWO, 8);
        va009 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_TWO, 9);
        va010 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_TWO, 10);
        va011 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_TWO, 11);
        va012 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_TWO, 12);
        va013 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_TWO, 13);
        va014 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_TWO, 14);
        va015 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_TWO, 15);
        vb0 = svreinterpret_x8_u32<TYPEB>(va008);
        vb1 = svreinterpret_x8_u32<TYPEB>(va009);
        vb2 = svreinterpret_x8_u32<TYPEB>(va010);
        vb3 = svreinterpret_x8_u32<TYPEB>(va011);
        vb4 = svreinterpret_x8_u32<TYPEB>(va012);
        vb5 = svreinterpret_x8_u32<TYPEB>(va013);
        vb6 = svreinterpret_x8_u32<TYPEB>(va014);
        vb7 = svreinterpret_x8_u32<TYPEB>(va015);

        svmopa_za32_x8_m_tile0(mask_i, mask_i, va0, vb0);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va1, vb1);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va2, vb2);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va3, vb3);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va4, vb4);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va5, vb5);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va6, vb6);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va7, vb7);
    }

    // write back
    for (int i = 0; i < SME_KERNEL_M_BLOCK_V4; ++i) {
        svst1_ver_za32(TILE_IDX_ZERO, i, p16_b32, c + ldc * i);
    }
}

template <typename TYPEA, typename TYPEB>
__arm_new("za") void gemm_kernel_sme_kernel_corner_v4(
    TYPEA *a, TYPEB *b, int32_t *c, uint32_t res_m, uint32_t res_n, uint32_t ldc, uint32_t k_depth) __arm_streaming
{
    svbool_t p16_b32 = svptrue_pat_b32(SV_ALL);
    svzero_za();
    for (int i = 0; i < SME_KERNEL_M_BLOCK_V4; ++i) {
        svld1_ver_za32(TILE_IDX_ZERO, i, p16_b32, c + ldc * i);
    }

    svbool_t mask_i = svptrue_pat_b8(SV_VL64);
    TYPEA *current_a_ptr = a;
    TYPEB *current_b_ptr = b;
    for (size_t i = 0; i < k_depth; i += SME_KERNEL_K_STEP_V4) {
        for (int za_idx = 0; za_idx < res_m; ++za_idx) {
            svld1_hor_za32(TILE_IDX_ONE, za_idx, p16_b32, current_a_ptr);
            current_a_ptr += SME_KERNEL_K_STEP_V4;
        }
        for (int za_idx = 0; za_idx < res_n; ++za_idx) {
            svld1_hor_za32(TILE_IDX_TWO, za_idx, p16_b32, current_b_ptr);
            current_b_ptr += SME_KERNEL_K_STEP_V4;
        }
        svuint32_t va000 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_ONE, 0);
        svuint32_t va001 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_ONE, 1);
        svuint32_t va002 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_ONE, 2);
        svuint32_t va003 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_ONE, 3);
        svuint32_t va004 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_ONE, 4);
        svuint32_t va005 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_ONE, 5);
        svuint32_t va006 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_ONE, 6);
        svuint32_t va007 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_ONE, 7);
        auto va0 = svreinterpret_x8_u32<TYPEA>(va000);
        auto va1 = svreinterpret_x8_u32<TYPEA>(va001);
        auto va2 = svreinterpret_x8_u32<TYPEA>(va002);
        auto va3 = svreinterpret_x8_u32<TYPEA>(va003);
        auto va4 = svreinterpret_x8_u32<TYPEA>(va004);
        auto va5 = svreinterpret_x8_u32<TYPEA>(va005);
        auto va6 = svreinterpret_x8_u32<TYPEA>(va006);
        auto va7 = svreinterpret_x8_u32<TYPEA>(va007);

        svuint32_t va008 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_TWO, 0);
        svuint32_t va009 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_TWO, 1);
        svuint32_t va010 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_TWO, 2);
        svuint32_t va011 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_TWO, 3);
        svuint32_t va012 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_TWO, 4);
        svuint32_t va013 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_TWO, 5);
        svuint32_t va014 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_TWO, 6);
        svuint32_t va015 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_TWO, 7);
        auto vb0 = svreinterpret_x8_u32<TYPEB>(va008);
        auto vb1 = svreinterpret_x8_u32<TYPEB>(va009);
        auto vb2 = svreinterpret_x8_u32<TYPEB>(va010);
        auto vb3 = svreinterpret_x8_u32<TYPEB>(va011);
        auto vb4 = svreinterpret_x8_u32<TYPEB>(va012);
        auto vb5 = svreinterpret_x8_u32<TYPEB>(va013);
        auto vb6 = svreinterpret_x8_u32<TYPEB>(va014);
        auto vb7 = svreinterpret_x8_u32<TYPEB>(va015);

        svmopa_za32_x8_m_tile0(mask_i, mask_i, va0, vb0);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va1, vb1);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va2, vb2);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va3, vb3);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va4, vb4);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va5, vb5);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va6, vb6);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va7, vb7);

        va000 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_ONE, 8);
        va001 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_ONE, 9);
        va002 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_ONE, 10);
        va003 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_ONE, 11);
        va004 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_ONE, 12);
        va005 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_ONE, 13);
        va006 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_ONE, 14);
        va007 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_ONE, 15);
        va0 = svreinterpret_x8_u32<TYPEA>(va000);
        va1 = svreinterpret_x8_u32<TYPEA>(va001);
        va2 = svreinterpret_x8_u32<TYPEA>(va002);
        va3 = svreinterpret_x8_u32<TYPEA>(va003);
        va4 = svreinterpret_x8_u32<TYPEA>(va004);
        va5 = svreinterpret_x8_u32<TYPEA>(va005);
        va6 = svreinterpret_x8_u32<TYPEA>(va006);
        va7 = svreinterpret_x8_u32<TYPEA>(va007);

        va008 = svread_ver_za32_u32_m(va000, p16_b32, TILE_IDX_TWO, 8);
        va009 = svread_ver_za32_u32_m(va001, p16_b32, TILE_IDX_TWO, 9);
        va010 = svread_ver_za32_u32_m(va002, p16_b32, TILE_IDX_TWO, 10);
        va011 = svread_ver_za32_u32_m(va003, p16_b32, TILE_IDX_TWO, 11);
        va012 = svread_ver_za32_u32_m(va004, p16_b32, TILE_IDX_TWO, 12);
        va013 = svread_ver_za32_u32_m(va005, p16_b32, TILE_IDX_TWO, 13);
        va014 = svread_ver_za32_u32_m(va006, p16_b32, TILE_IDX_TWO, 14);
        va015 = svread_ver_za32_u32_m(va007, p16_b32, TILE_IDX_TWO, 15);
        vb0 = svreinterpret_x8_u32<TYPEB>(va008);
        vb1 = svreinterpret_x8_u32<TYPEB>(va009);
        vb2 = svreinterpret_x8_u32<TYPEB>(va010);
        vb3 = svreinterpret_x8_u32<TYPEB>(va011);
        vb4 = svreinterpret_x8_u32<TYPEB>(va012);
        vb5 = svreinterpret_x8_u32<TYPEB>(va013);
        vb6 = svreinterpret_x8_u32<TYPEB>(va014);
        vb7 = svreinterpret_x8_u32<TYPEB>(va015);

        svmopa_za32_x8_m_tile0(mask_i, mask_i, va0, vb0);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va1, vb1);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va2, vb2);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va3, vb3);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va4, vb4);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va5, vb5);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va6, vb6);
        svmopa_za32_x8_m_tile0(mask_i, mask_i, va7, vb7);
    }

    // write back
    svbool_t pn_b32 = svwhilelt_b32((uint32_t)0, res_m);
    for (int i = 0; i < res_n; ++i) {
        svst1_ver_za32(TILE_IDX_ZERO, i, pn_b32, c + ldc * i);
    }
}

template <typename TYPEA, typename TYPEB>
void gemm_kernel_x8x8s32(void *allocPtrA, void *alignPtrA, int64_t offsetA, int64_t sizeA1, int64_t sizeA2,
    int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB, int64_t offsetB, int64_t sizeB1,
    int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC, void *alignPtrC, int64_t offsetC,
    int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2, int64_t offset0, int64_t offset1, int64_t m,
    int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    TYPEA *alignPtrAFloat = reinterpret_cast<TYPEA *>(alignPtrA);
    TYPEB *alignPtrBFloat = reinterpret_cast<TYPEB *>(alignPtrB);
    int32_t *alignPtrCFloat = reinterpret_cast<int32_t *>(alignPtrC);
    alignPtrAFloat += offsetA;
    alignPtrBFloat += offsetB;
    alignPtrCFloat += (offset0 * strideC1 + offset1 * strideC2 + offsetC);

    size_t k_block_size_up = (k + SME_KERNEL_K_STEP_V4 - 1) / SME_KERNEL_K_STEP_V4 * SME_KERNEL_K_STEP_V4;
    for (size_t n_sub_block = 0; n_sub_block < n; n_sub_block += SME_KERNEL_N_BLOCK_V4) {
        size_t n_sub_block_size = n - n_sub_block;
        if (n_sub_block_size > SME_KERNEL_N_BLOCK_V4) {
            n_sub_block_size = SME_KERNEL_N_BLOCK_V4;
        }
        auto *current_bufferB_ptr = alignPtrBFloat + n_sub_block * k_block_size_up;
        for (size_t m_sub_block = 0; m_sub_block < m; m_sub_block += SME_KERNEL_M_BLOCK_V4) {
            size_t m_sub_block_size = m - m_sub_block;
            if (m_sub_block_size > SME_KERNEL_M_BLOCK_V4) {
                m_sub_block_size = SME_KERNEL_M_BLOCK_V4;
            }
            auto *current_bufferA_ptr = alignPtrAFloat + m_sub_block * k_block_size_up;
            int32_t *current_bufferC_ptr = alignPtrCFloat + n_sub_block * strideC2 + m_sub_block * strideC1;
            if (n_sub_block_size != SME_KERNEL_N_BLOCK_V4 || m_sub_block_size != SME_KERNEL_M_BLOCK_V4) {
                gemm_kernel_sme_kernel_corner_v4(current_bufferA_ptr,
                    current_bufferB_ptr,
                    current_bufferC_ptr,
                    m_sub_block_size,
                    n_sub_block_size,
                    strideC2,
                    k_block_size_up);
            } else {
                gemm_kernel_sme_kernel_v4(
                    current_bufferA_ptr, current_bufferB_ptr, current_bufferC_ptr, strideC2, k_block_size_up);
            }
        }
    }
    return;
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void gemm_kernel_s8s8s32_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    gemm_kernel_x8x8s32<int8_t, int8_t>(allocPtrA,
        alignPtrA,
        offsetA,
        sizeA1,
        sizeA2,
        strideA1,
        strideA2,
        allocPtrB,
        alignPtrB,
        offsetB,
        sizeB1,
        sizeB2,
        strideB1,
        strideB2,
        allocPtrC,
        alignPtrC,
        offsetC,
        sizeC1,
        sizeC2,
        strideC1,
        strideC2,
        offset0,
        offset1,
        m,
        n,
        k,
        lda,
        ldb,
        alpha);
}
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void gemm_kernel_u8s8s32_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    gemm_kernel_x8x8s32<uint8_t, int8_t>(allocPtrA,
        alignPtrA,
        offsetA,
        sizeA1,
        sizeA2,
        strideA1,
        strideA2,
        allocPtrB,
        alignPtrB,
        offsetB,
        sizeB1,
        sizeB2,
        strideB1,
        strideB2,
        allocPtrC,
        alignPtrC,
        offsetC,
        sizeC1,
        sizeC2,
        strideC1,
        strideC2,
        offset0,
        offset1,
        m,
        n,
        k,
        lda,
        ldb,
        alpha);
}

extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void gemm_kernel_s8u8s32_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    gemm_kernel_x8x8s32<int8_t, uint8_t>(allocPtrA,
        alignPtrA,
        offsetA,
        sizeA1,
        sizeA2,
        strideA1,
        strideA2,
        allocPtrB,
        alignPtrB,
        offsetB,
        sizeB1,
        sizeB2,
        strideB1,
        strideB2,
        allocPtrC,
        alignPtrC,
        offsetC,
        sizeC1,
        sizeC2,
        strideC1,
        strideC2,
        offset0,
        offset1,
        m,
        n,
        k,
        lda,
        ldb,
        alpha);
}
extern "C" MLIR_KUNPENG_WRAPPERS_EXPORT void gemm_kernel_u8u8s32_wrap(void *allocPtrA, void *alignPtrA, int64_t offsetA,
    int64_t sizeA1, int64_t sizeA2, int64_t strideA1, int64_t strideA2, void *allocPtrB, void *alignPtrB,
    int64_t offsetB, int64_t sizeB1, int64_t sizeB2, int64_t strideB1, int64_t strideB2, void *allocPtrC,
    void *alignPtrC, int64_t offsetC, int64_t sizeC1, int64_t sizeC2, int64_t strideC1, int64_t strideC2,
    int64_t offset0, int64_t offset1, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, float alpha)
{
    gemm_kernel_x8x8s32<uint8_t, uint8_t>(allocPtrA,
        alignPtrA,
        offsetA,
        sizeA1,
        sizeA2,
        strideA1,
        strideA2,
        allocPtrB,
        alignPtrB,
        offsetB,
        sizeB1,
        sizeB2,
        strideB1,
        strideB2,
        allocPtrC,
        alignPtrC,
        offsetC,
        sizeC1,
        sizeC2,
        strideC1,
        strideC2,
        offset0,
        offset1,
        m,
        n,
        k,
        lda,
        ldb,
        alpha);
}
