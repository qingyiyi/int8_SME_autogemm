// Portable production-driver test.  It replaces the target-only packers and
// SME kernel with strict mocks, so it verifies that the original
// cblas_gemm_s8s8s32() driver passes the fused-store ABI to the original NN
// kernel symbol.  It does NOT execute AArch64 SME assembly.
#include "tests/reference.hpp"
#include "int8_gemm.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Production definitions live in int8_gemm.cpp, which this test links.
extern int nthreadsM;
extern int nthreadsN;

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
constexpr int8_t kKernelMarker = static_cast<int8_t>(91);

enum class C8Behavior { Inverse, Marker };

struct Invocation {
    int32_t* c32 = nullptr;
    int8_t* c8 = nullptr;
    unsigned mode = 0;
    C8Behavior c8_behavior = C8Behavior::Inverse;
    std::atomic<size_t> pack_a_calls{0};
    std::atomic<size_t> pack_b_calls{0};
    std::atomic<size_t> kernel_calls{0};
    std::atomic<bool> invalid{false};
};

Invocation* g_invocation = nullptr;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

int32_t expected_c32(int row, int col) {
    // Keep values far from padding and exercise positive/negative remainders.
    return static_cast<int32_t>(((row * 131 + col * 17 + 7) % 100003) - 50001);
}

bool tile_origin(const Int8FusedStoreParams* params, BLASLONG* row, BLASLONG* col) {
    if (g_invocation == nullptr || params == nullptr) return false;
    const uintptr_t c32_begin = reinterpret_cast<uintptr_t>(g_invocation->c32);
    const uintptr_t c32_end = c32_begin +
        static_cast<uintptr_t>(kLdc32) * kN * sizeof(int32_t);
    const uintptr_t c8_begin = reinterpret_cast<uintptr_t>(g_invocation->c8);
    const uintptr_t c8_end = c8_begin + static_cast<uintptr_t>(kLdc8) * kN;
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
            "portable production-driver test could not create the requested OpenMP team");
}

void run_one(unsigned mode, int threads, C8Behavior c8_behavior) {
    check_full_team(threads);
    nthreadsM = threads;
    nthreadsN = 1;

    std::vector<int8_t> a(static_cast<size_t>(kLda) * kK, 3);
    std::vector<int8_t> b(static_cast<size_t>(kLdb) * kN, -5);
    const auto original_a = a;
    const auto original_b = b;
    std::vector<int8_t> sa(static_cast<size_t>(kK) * 128 * 32 + 65536, 0);
    std::vector<int8_t> sb(static_cast<size_t>(kK) * 8192 + 65536, 0);
    std::vector<int32_t> c32(static_cast<size_t>(kLdc32) * kN, kC32Padding);
    std::vector<int8_t> c8(static_cast<size_t>(kLdc8) * kN, kC8Padding);

    Invocation invocation;
    invocation.c32 = c32.data();
    invocation.c8 = c8.data();
    invocation.mode = mode;
    invocation.c8_behavior = c8_behavior;
    g_invocation = &invocation;
    const int32_t offset = 0;

    cblas_gemm_s8s8s32(
        CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
        kM, kN, kK, 1.0f, a.data(), kLda, 0, b.data(), kLdb, 0, 0.0f,
        c32.data(), kLdc32, &offset, sa.data(), sb.data(), c8.data(), kLdc8, mode);
    g_invocation = nullptr;

    require(!invocation.invalid.load(),
            "production driver passed an invalid fused-store parameter block");
    const size_t expected_m_tiles = threads == 1 ? 16 : 32;
    const size_t expected_kernel_calls = expected_m_tiles * (kN / 32);
    require(invocation.pack_a_calls.load() == expected_m_tiles,
            "unexpected A-pack call count");
    require(invocation.pack_b_calls.load() == static_cast<size_t>(threads),
            "unexpected B-pack call count");
    require(invocation.kernel_calls.load() == expected_kernel_calls,
            "unexpected production-kernel call count");

    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row < kM; ++row) {
            const int32_t expected32 = expected_c32(row, col);
            const int8_t expected8 = c8_behavior == C8Behavior::Inverse
                ? fusion_test::inverse_integer(expected32, mode)
                : kKernelMarker;
            const size_t c32_index = static_cast<size_t>(col) * kLdc32 + row;
            const size_t c8_index = static_cast<size_t>(col) * kLdc8 + row;
            if (c32[c32_index] != expected32 || c8[c8_index] != expected8) {
                fail("mocked production output mismatch mode=" + std::to_string(mode) +
                     " threads=" + std::to_string(threads) + " row=" +
                     std::to_string(row) + " col=" + std::to_string(col));
            }
        }
        for (int row = kM; row < kLdc32; ++row) {
            if (c32[static_cast<size_t>(col) * kLdc32 + row] != kC32Padding) {
                fail("production driver mock touched C32 padding");
            }
        }
        for (int row = kM; row < kLdc8; ++row) {
            if (c8[static_cast<size_t>(col) * kLdc8 + row] != kC8Padding) {
                fail("production driver mock touched C8 padding");
            }
        }
    }
    require(a == original_a && b == original_b,
            "production driver modified an input matrix");
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

extern "C" void int8_sme_gemm_kernel_nn(
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
                invocation->c8_behavior == C8Behavior::Inverse
                    ? fusion_test::inverse_integer(value, invocation->mode)
                    : kKernelMarker;
        }
    }
    ++invocation->kernel_calls;
}

int main() {
    try {
        // All modes validate the C++ mode-to-modulus wiring on the serial path.
        for (unsigned mode = 0; mode <= fusion_test::kModuli.size(); ++mode) {
            run_one(mode, 1, C8Behavior::Inverse);
        }
        // Exercise production tiling and barriers with the target thread layout.
        for (unsigned mode : {0u, 1u, 19u}) {
            run_one(mode, 32, C8Behavior::Inverse);
        }
        // The mock imitates the assembly C8 store with a non-oracle marker.  If
        // a C++ inverse-scaling loop still runs after the kernel returns, this
        // marker would be overwritten and this call fails.
        run_one(1, 1, C8Behavior::Marker);
        std::cout << "PASS host production-driver mock: original API wiring, all "
                     "moduli, C32/C8 strides, padding, 1/32-thread tiling, and "
                     "absence of a post-kernel C++ inverse-scaling pass verified.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL host production-driver mock: " << error.what() << '\n';
        return 1;
    }
}
