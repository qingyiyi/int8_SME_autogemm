#include "int8_gemm.hpp"

#include <cstring>

// This file intentionally contains a separate phase-1 driver.  The existing
// int8_gemm.cpp path is left byte-for-byte unchanged so it remains the
// reference implementation while the fused symbol is brought up.

extern int nthreadsM;
extern int nthreadsN;

namespace {

constexpr BLASLONG kGemmR = 8192;
constexpr BLASLONG kGemmQ = 2048;
constexpr BLASLONG kGemmP = 128;
constexpr int kJBlock = 32;
constexpr int kModuli[19] = {
    255, 253, 251, 247, 241, 239, 233, 229, 227, 223,
    217, 211, 199, 197, 193, 191, 181, 179, 173
};

static void get_m_simple_threads_regions(const BLASLONG n, BLASLONG *range,
                                         const BLASLONG threads,
                                         const BLASLONG aligned_size) {
    size_t i;
    if (threads == 1 || n < aligned_size) {
        i = 2;
        range[0] = 0;
        range[1] = n;
    } else {
        BLASLONG average = n / threads;
        if (average < aligned_size) average = aligned_size;
        average = ((average + aligned_size - 1) / aligned_size) * aligned_size;
        BLASLONG remainder = 0;
        const BLASLONG temporary_threads = MIN(threads, n / average);
        if (temporary_threads * average < n) {
            remainder = n - temporary_threads * average;
        }
        for (i = 0; i < static_cast<size_t>(temporary_threads); ++i) {
            range[i] = average * static_cast<BLASLONG>(i);
        }
        const BLASLONG average_add_range =
            (threads - temporary_threads) > 0
                ? remainder / (threads - temporary_threads)
                : 0;
        for (; i < static_cast<size_t>(threads); ++i) {
            range[i] = range[i - 1] + average_add_range;
        }
        if (threads < MAX_CPU_NUMBER) {
            range[threads] = n;
            ++i;
        }
    }
    for (; i < MAX_CPU_NUMBER; ++i) range[i] = range[i - 1];
}

static void fused_gemm_driver(const BlasArgs *args, FLOAT *sa, FLOAT *sb,
                              BLASULONG mask, const BLASLONG *range_m,
                              const BLASLONG *range_n, int8_t *c8,
                              size_t ldc8, int32_t modulus) {
    const int thread_id = omp_get_thread_num();
    const int my_pos = thread_id % nthreadsM;
    const int ny_pos = thread_id / nthreadsM;

    FLOAT *a = static_cast<FLOAT *>(const_cast<void *>(args->a));
    FLOAT *b = static_cast<FLOAT *>(args->b);
    FLOATC *c = static_cast<FLOATC *>(args->c);
    const BLASLONG lda = args->lda;
    const BLASLONG ldb = args->ldb;
    const BLASLONG ldc = args->ldc;
    const BLASLONG k = args->k;
    float *alpha = static_cast<float *>(const_cast<void *>(args->alpha));

    const BLASLONG m_from = range_m[my_pos];
    const BLASLONG m_to = range_m[my_pos + 1];
    const BLASLONG n_from = range_n[ny_pos];
    const BLASLONG n_to = range_n[ny_pos + 1];

    FLOAT *buf_aa = sa;
    FLOAT *buf_bb = sb;
    const int threads = nthreadsM;
    FLOAT **buffer_b = reinterpret_cast<FLOAT **>(args->common);

    for (BLASLONG js = n_from; js < n_to;) {
        BLASLONG min_j = n_to - js;
        if (min_j > kGemmR) min_j = kGemmR;

        // Phase 1 deliberately supports the fixed K=2048 contract only.  The
        // single K panel is important: inverse scaling must happen after the
        // complete C32 result, not after an intermediate K partial sum.
        for (BLASLONG ls = 0; ls < k; ) {
            BLASLONG min_l = k - ls;
            if (min_l >= kGemmQ) min_l = kGemmQ;

#pragma omp barrier
            if (mask & GEMM_PB_MASK) {
                const int my_j = static_cast<int>((min_j + threads - 1) / threads);
                int packed_j = my_j;
                if (my_j * my_pos > n_to) packed_j = static_cast<int>(n_to - my_j * my_pos);
                PACK_B(min_l, packed_j, b, ldb, ls,
                       js + my_j * my_pos,
                       buf_bb + ny_pos * min_l * min_j + min_l * (my_j * my_pos),
                       args->ob);
                buffer_b[my_pos] = buf_bb;
            }

#pragma omp barrier
            for (BLASLONG is = m_from; is < m_to;) {
                BLASLONG min_i = m_to - is;
                if (min_i >= kGemmP) min_i = kGemmP;

                PACK_A(min_l, min_i, a, lda, ls, is, buf_aa, args->oa);

                for (BLASLONG jj = js; jj < js + min_j && jj < n_to;
                     jj += kJBlock) {
                    BLASLONG min_jj = js + min_j - jj;
                    if (min_jj > kJBlock) min_jj = kJBlock;

                    Int8FusedStoreParams store_params{};
                    store_params.c32 = c + is + jj * ldc;
                    store_params.c8 = c8 + is + jj * static_cast<BLASLONG>(ldc8);
                    store_params.ldc32 = ldc;
                    store_params.ldc8 = static_cast<int64_t>(ldc8);
                    store_params.rows = min_i;
                    store_params.cols = min_jj;
                    store_params.modulus = modulus;

                    int8_sme_gemm_kernel_nn_fused(
                        min_i, min_jj, min_l, buf_aa, lda, alpha[0],
                        buf_bb + ny_pos * min_l * min_j + (jj - js) * min_l,
                        ldb, store_params.c32, ldc, &store_params);
                }
                is += min_i;
            }
            ls += min_l;
        }
        js += min_j;
    }
}

static void fallback_to_original(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const CBLAS_OFFSET offsetc,
    const BLASINT m, const BLASINT n, const BLASINT k, const float alpha,
    void *a, const BLASINT lda, const BLASINT8 oa, void *b,
    const BLASINT ldb, const BLASINT8 ob, const float beta, int32_t *c,
    const BLASINT ldc, const int32_t *oc, int8_t *sa, int8_t *sb,
    int8_t *c8, size_t ldc8, unsigned num_moduli) {
    cblas_gemm_s8s8s32(layout, transa, transb, offsetc, m, n, k, alpha,
                        a, lda, oa, b, ldb, ob, beta, c, ldc, oc, sa, sb,
                        c8, ldc8, num_moduli);
}

}  // namespace

extern "C" void cblas_gemm_s8s8s32_fused(
    const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const CBLAS_OFFSET offsetc,
    const BLASINT m, const BLASINT n, const BLASINT k, const float alpha,
    void *a_, const BLASINT lda, const BLASINT8 oa, void *b_,
    const BLASINT ldb, const BLASINT8 ob, const float beta, int32_t *c,
    const BLASINT ldc, const int32_t *oc, int8_t *sa, int8_t *sb,
    int8_t *c8, size_t ldc8, unsigned num_moduli) {
    // A K panel split would expose an intermediate C32 value to the fused
    // store.  Keep every non-business M/N shape (for example the unrelated
    // 512x64 case) on the untouched original path too.  Phase 1 accepts only
    // positive 2048-multiple M/N and one K=2048 panel.
    const bool phase1_shape = m > 0 && n > 0 &&
                              m % kGemmQ == 0 && n % kGemmQ == 0;
    if (!phase1_shape || k != kGemmQ ||
        num_moduli > sizeof(kModuli) / sizeof(kModuli[0])) {
        fallback_to_original(layout, transa, transb, offsetc, m, n, k, alpha,
                             a_, lda, oa, b_, ldb, ob, beta, c, ldc, oc, sa,
                             sb, c8, ldc8, num_moduli);
        return;
    }

    const int nthreads = nthreadsM * nthreadsN;
    BlasArgs args{};
    static const float alpha_one = 1.0f;
    args.alpha = &alpha_one;
    args.nthreads = nthreads;
    args.a = a_;
    args.b = b_;
    args.c = c;
    args.m = m;
    args.n = n;
    args.k = k;
    args.lda = lda;
    args.ldb = ldb;
    args.ldc = ldc;
    args.oa = oa;
    args.ob = ob;
    args.oc = oc;

    BLASLONG range_m[MAX_CPU_NUMBER + 2];
    BLASLONG range_n[MAX_CPU_NUMBER + 2];
    get_m_simple_threads_regions(m, range_m, nthreadsM, REGION_ALIGN_SIZE);
    get_m_simple_threads_regions(n, range_n, nthreadsN, REGION_ALIGN_SIZE);

    const BLASULONG mask = GEMM_PB_MASK | GEMM_PA_MASK;
    volatile int *job_t[nthreads * nthreads];
    std::memset(job_t, 0, sizeof(job_t));
    args.common = const_cast<void *>(reinterpret_cast<const void *>(job_t));

    const int32_t modulus = num_moduli == 0 ? 0 : kModuli[num_moduli - 1];
#pragma omp parallel for num_threads(nthreads) schedule(static) shared(sa, sb)
    for (int i = 0; i < nthreads; ++i) {
        const int thread_id = omp_get_thread_num();
        fused_gemm_driver(
            &args,
            sa + kGemmQ * kGemmP * thread_id,
            sb, mask, range_m, range_n, c8, ldc8, modulus);
    }
}
