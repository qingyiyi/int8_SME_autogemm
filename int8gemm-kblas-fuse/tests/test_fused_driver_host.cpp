// Portable phase-1 driver test.  It replaces the target-only packers and SME
// kernel with strict mocks, so it validates C++ tiling/ABI-parameter wiring on
// any host.  It does NOT execute the AArch64 assembly; test_fused_stage1 does
// that on the target machine.
#include "tests/reference.hpp"
#include "int8_gemm.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int nthreadsM = 1;
int nthreadsN = 1;

namespace {
constexpr int kM = 2048;
constexpr int kN = 2048;
constexpr int kK = fusion_test::kFixedK;
constexpr int kLda = kM + 7;
constexpr int kLdb = kK + 11;
constexpr int kLdc32 = kM + 16;
constexpr int kLdc8 = kM + 23;
constexpr int32_t kC32Padding = 0x2468ace0;
constexpr int8_t kC8Padding = static_cast<int8_t>(-91);

struct Invocation {
    int32_t* c32 = nullptr;
    int8_t* c8 = nullptr;
    unsigned mode = 0;
    std::atomic<size_t> pack_a_calls{0};
    std::atomic<size_t> pack_b_calls{0};
    std::atomic<size_t> kernel_calls{0};
    std::atomic<bool> invalid{false};
};

Invocation* g_invocation = nullptr;
std::atomic<size_t> g_fallback_calls{0};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

int32_t expected_c32(int row, int col) {
    // Keep the values far from the padding sentinel and exercise signs.
    return static_cast<int32_t>(((row * 131 + col * 17 + 7) % 100003) - 50001);
}

bool tile_origin(const Int8FusedStoreParams* params, BLASLONG* row, BLASLONG* col) {
    if (g_invocation == nullptr || params == nullptr) return false;
    const uintptr_t c32_begin = reinterpret_cast<uintptr_t>(g_invocation->c32);
    const uintptr_t c32_end = c32_begin + static_cast<uintptr_t>(kLdc32) * kN * sizeof(int32_t);
    const uintptr_t c8_begin = reinterpret_cast<uintptr_t>(g_invocation->c8);
    const uintptr_t c8_end = c8_begin + static_cast<uintptr_t>(kLdc8) * kN * sizeof(int8_t);
    const uintptr_t c32_address = reinterpret_cast<uintptr_t>(params->c32);
    const uintptr_t c8_address = reinterpret_cast<uintptr_t>(params->c8);
    if (c32_address < c32_begin || c32_address >= c32_end ||
        c8_address < c8_begin || c8_address >= c8_end ||
        ((c32_address - c32_begin) % sizeof(int32_t)) != 0) {
        return false;
    }
    const size_t c32_offset = (c32_address - c32_begin) / sizeof(int32_t);
    const size_t c8_offset = c8_address - c8_begin;
    *row = static_cast<BLASLONG>(c32_offset % kLdc32);
    *col = static_cast<BLASLONG>(c32_offset / kLdc32);
    return c8_offset == static_cast<size_t>(*row) + static_cast<size_t>(*col) * kLdc8;
}

void check_full_team(int threads) {
    int observed = 0;
#pragma omp parallel num_threads(threads) reduction(max:observed)
    { observed = omp_get_num_threads(); }
    require(observed == threads,
            "portable driver test could not create the requested OpenMP team");
}

void run_one(unsigned mode, int threads) {
    check_full_team(threads);
    nthreadsM = threads;
    nthreadsN = 1;

    std::vector<int8_t> a(static_cast<size_t>(kLda) * kK, 3);
    std::vector<int8_t> b(static_cast<size_t>(kLdb) * kN, -5);
    std::vector<int8_t> sa(static_cast<size_t>(kK) * 128 * 32 + 65536, 0);
    std::vector<int8_t> sb(static_cast<size_t>(kK) * 8192 + 65536, 0);
    std::vector<int32_t> c32(static_cast<size_t>(kLdc32) * kN, kC32Padding);
    std::vector<int8_t> c8(static_cast<size_t>(kLdc8) * kN, kC8Padding);

    Invocation invocation;
    invocation.c32 = c32.data();
    invocation.c8 = c8.data();
    invocation.mode = mode;
    g_invocation = &invocation;
    const int32_t offset = 0;

    cblas_gemm_s8s8s32_fused(
        CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
        kM, kN, kK, 1.0f, a.data(), kLda, 0, b.data(), kLdb, 0, 0.0f,
        c32.data(), kLdc32, &offset, sa.data(), sb.data(), c8.data(), kLdc8, mode);
    g_invocation = nullptr;

    require(!invocation.invalid.load(), "fused driver passed an invalid tile parameter block");
    const size_t expected_m_tiles = threads == 1 ? 16 : 32;
    const size_t expected_kernel_calls = expected_m_tiles * (kN / 32);
    require(invocation.pack_a_calls.load() == expected_m_tiles,
            "unexpected A-pack call count");
    require(invocation.pack_b_calls.load() == static_cast<size_t>(threads),
            "unexpected B-pack call count");
    require(invocation.kernel_calls.load() == expected_kernel_calls,
            "unexpected fused-kernel call count");

    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row < kM; ++row) {
            const int32_t expected32 = expected_c32(row, col);
            const int expected8 = fusion_test::inverse_integer(expected32, mode);
            const size_t c32_index = static_cast<size_t>(col) * kLdc32 + row;
            const size_t c8_index = static_cast<size_t>(col) * kLdc8 + row;
            if (c32[c32_index] != expected32 || c8[c8_index] != expected8) {
                fail("mocked fused output mismatch mode=" + std::to_string(mode) +
                     " threads=" + std::to_string(threads) + " row=" +
                     std::to_string(row) + " col=" + std::to_string(col));
            }
        }
        for (int row = kM; row < kLdc32; ++row) {
            if (c32[static_cast<size_t>(col) * kLdc32 + row] != kC32Padding) {
                fail("fused driver mock touched C32 padding");
            }
        }
        for (int row = kM; row < kLdc8; ++row) {
            if (c8[static_cast<size_t>(col) * kLdc8 + row] != kC8Padding) {
                fail("fused driver mock touched C8 padding");
            }
        }
    }
}

void invoke_fallback_candidate(BLASINT m, BLASINT n, BLASINT k) {
    cblas_gemm_s8s8s32_fused(
        CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
        m, n, k, 1.0f, nullptr, 1, 0, nullptr, 1, 0, 0.0f,
        nullptr, 1, nullptr, nullptr, nullptr, nullptr, 1, 0);
}

void test_fallback() {
    size_t before = g_fallback_calls.load();
    invoke_fallback_candidate(2048, 2048, kK - 1);
    require(g_fallback_calls.load() == before + 1,
            "non-2048 K did not dispatch to the original API");

    before = g_fallback_calls.load();
    invoke_fallback_candidate(2047, 2048, kK);
    require(g_fallback_calls.load() == before + 1,
            "non-2048-multiple M did not dispatch to the original API");

    before = g_fallback_calls.load();
    invoke_fallback_candidate(2048, 64, kK);
    require(g_fallback_calls.load() == before + 1,
            "non-2048-multiple N did not dispatch to the original API");
}
}  // namespace

extern "C" void int8_sme_gemm_itcopy(
    const BLASLONG, const BLASLONG, const BLASINT8*, const BLASLONG,
    BLASINT8*, BLASINT8) {
    if (g_invocation != nullptr) ++g_invocation->pack_a_calls;
}

extern "C" void int8_sme_gemm_oncopy(
    const BLASLONG, const BLASLONG, const BLASINT8*, const BLASLONG,
    BLASINT8*, BLASINT8) {
    if (g_invocation != nullptr) ++g_invocation->pack_b_calls;
}

extern "C" void int8_sme_gemm_kernel_nn_fused(
    BLASLONG rows, BLASLONG cols, BLASLONG k, void*, BLASLONG, float alpha,
    void*, BLASLONG, int32_t* c32, BLASLONG ldc32, void* buf) {
    Invocation* invocation = g_invocation;
    const auto* params = static_cast<const Int8FusedStoreParams*>(buf);
    BLASLONG base_row = 0;
    BLASLONG base_col = 0;
    const int32_t expected_modulus = invocation != nullptr && invocation->mode != 0
        ? fusion_test::kModuli[invocation->mode - 1]
        : 0;
    if (invocation == nullptr || params == nullptr || k != kK || alpha != 1.0f ||
        params->c32 != c32 || params->ldc32 != ldc32 || params->ldc32 != kLdc32 ||
        params->ldc8 != kLdc8 || params->rows != rows || params->cols != cols ||
        params->modulus != expected_modulus || rows <= 0 || cols <= 0 ||
        !tile_origin(params, &base_row, &base_col) || base_row + rows > kM ||
        base_col + cols > kN) {
        if (invocation != nullptr) invocation->invalid.store(true);
        return;
    }

    for (BLASLONG col = 0; col < cols; ++col) {
        for (BLASLONG row = 0; row < rows; ++row) {
            const int32_t value = expected_c32(static_cast<int>(base_row + row),
                                               static_cast<int>(base_col + col));
            params->c32[col * params->ldc32 + row] = value;
            params->c8[col * params->ldc8 + row] =
                fusion_test::inverse_integer(value, invocation->mode);
        }
    }
    ++invocation->kernel_calls;
}

extern "C" void cblas_gemm_s8s8s32(
    const CBLAS_LAYOUT, const CBLAS_TRANSPOSE, const CBLAS_TRANSPOSE,
    const CBLAS_OFFSET, const BLASINT, const BLASINT, const BLASINT,
    const float, void*, const BLASINT, const BLASINT8, void*, const BLASINT,
    const BLASINT8, const float, int32_t*, const BLASINT, const int32_t*,
    int8_t*, int8_t*, int8_t*, size_t, unsigned) {
    ++g_fallback_calls;
}

int main() {
    try {
        for (int threads : {1, 32}) {
            for (unsigned mode : {0u, 1u, 19u}) run_one(mode, threads);
        }
        test_fallback();
        std::cout << "PASS host fused-driver mock: fixed-K tiling, business-shape "
                     "fallback, C32/C8 tile addresses, strides, modes, padding and "
                     "1/32-thread paths verified.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL host fused-driver mock: " << error.what() << '\n';
        return 1;
    }
}
