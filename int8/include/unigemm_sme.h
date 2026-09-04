#ifndef UNIGEMM_SME_H
#define UNIGEMM_SME_H

#include <cstdint>
#include <cstddef>
#include <cstdio>

inline void sme_enter() {
#ifdef __aarch64__
    __asm__ volatile("msr SVCRSMZA, #1\nisb\n":::"z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15");
#endif
}

inline void sme_exit() {
#ifdef __aarch64__
    __asm__ volatile("msr SVCRSMZA, #0\nisb\n":::"z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15");
#endif
}

void init_hbm();
bool has_hbm();
double get_cpu_freq_ghz();

void unigemm_sme_int8(
    const int8_t* a, long lda,
    const int8_t* b, long ldb,
    int32_t* c, long ldc,
    long m, long n, long k);

void unigemm_sme_int8_mt(
    const int8_t* a, long lda,
    const int8_t* b, long ldb,
    int32_t* c, long ldc,
    long m, long n, long k);

void unigemm_sme_f16(
    const uint16_t* a, long lda,
    const uint16_t* b, long ldb,
    float* c, long ldc,
    long m, long n, long k);

void unigemm_sme_bf16(
    const uint16_t* a, long lda,
    const uint16_t* b, long ldb,
    float* c, long ldc,
    long m, long n, long k);

void unigemm_sme_f32(
    const float* a, long lda,
    const float* b, long ldb,
    float* c, long ldc,
    long m, long n, long k);

int unigemm_sme_get_num_threads();

#ifdef GRID_DEBUG_TIMING
void grid_init(size_t rows, size_t cols, long k);
void grid_record(size_t row, size_t col, uint64_t ns);
void grid_write(const char* filename, long m, long n, double peak_gflops, double flops_per_call);
#endif

#ifdef SME_DEBUG_TIMING
void sme_timing_reset();
void sme_timing_print();
#endif

#endif