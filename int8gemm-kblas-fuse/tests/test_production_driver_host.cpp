// Portable production-driver test.  It replaces the target-only packers and
// SME kernel with strict mocks, so it verifies that the original
// cblas_gemm_s8s8s8() driver passes the C8-only fused-store ABI to the
// original NN kernel symbol.  It does NOT execute AArch64 SME assembly.
#include "tests/reference.hpp"
#include "int8_gemm.hpp"

#include <algorithm>
#include <array>
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
constexpr int kLdc8 = kM + 23;
constexpr int8_t kC8Padding = static_cast<int8_t>(-91);
constexpr int8_t kKernelMarker = static_cast<int8_t>(91);

enum class C8Behavior { Inverse, Marker };

struct Invocation {
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

int32_t expected_accumulator(int row, int col) {
    // Keep values far from padding and exercise positive/negative remainders.
    return static_cast<int32_t>(((row * 131 + col * 17 + 7) % 100003) - 50001);
}

Int8FusedStoreParams expected_scaling(unsigned mode) {
    Int8FusedStoreParams expected{};
    if (mode != 0) {
        const double p = static_cast<double>(fusion_test::kModuli[mode - 1]);
        expected.modulus = static_cast<int32_t>(p);
        expected.inv_p = 1.0 / p;
        expected.neg_p = -p;
    }
    return expected;
}

bool tile_origin(const Int8FusedStoreParams* params, BLASLONG* row, BLASLONG* col) {
    if (g_invocation == nullptr || params == nullptr || params->c8 == nullptr) {
        return false;
    }
    const uintptr_t c8_begin = reinterpret_cast<uintptr_t>(g_invocation->c8);
    const uintptr_t c8_end = c8_begin + static_cast<uintptr_t>(kLdc8) * kN;
    const uintptr_t c8_address = reinterpret_cast<uintptr_t>(params->c8);
    if (c8_address < c8_begin || c8_address >= c8_end) return false;
    const size_t c8_offset = c8_address - c8_begin;
    *row = static_cast<BLASLONG>(c8_offset % kLdc8);
    *col = static_cast<BLASLONG>(c8_offset / kLdc8);
    return c8_offset == static_cast<size_t>(*row) +
                         static_cast<size_t>(*col) * kLdc8;
}

void check_full_team(int threads) {
    int observed = 0;
#pragma omp parallel num_threads(threads) reduction(max:observed)
    { observed = omp_get_num_threads(); }
    require(observed == threads,
            "portable production-driver test could not create the requested OpenMP team");
}

// P3 uses the kernel C cursor directly as a C8 byte cursor.  Under the fixed
// production SVL=64 contract, one .s vector contains 16 INT32 lanes and packs
// to 16 C8 bytes after UZP1 x 2.  Thus SAVE_ZACOL offsets 0..3 map to
// {0, 16, 32, 48} C8 bytes, not {0, 64, 128, 192} bytes.
//
// The original save traversal has a second, algorithmic stride: after a
// SAVE_ZACOL_1/2/3/4VL group, pc advances by 4 * LDC.  That is four output
// columns, not a C32 element-width conversion, so it intentionally remains in
// P3.  This host-only arithmetic test verifies every part of the direct C8
// cursor algebra and proves equivalence to the retired virtual-C32 mapping at
// the one supported SVL=64.
void verify_direct_c8_cursor_mapping() {
    constexpr uint64_t kSyntheticC8Base = UINT64_C(0x100000003);
    constexpr BLASLONG kSmeVectorBytes = 64;
    constexpr BLASLONG kPackedC8Bytes =
        kSmeVectorBytes / static_cast<BLASLONG>(sizeof(int32_t));
    constexpr std::array<BLASLONG, 2> kMatrixSizes = {2048, 8192};
    constexpr std::array<BLASLONG, 4> kRelativeRows = {0, 16, 64, 127};
    constexpr std::array<BLASLONG, 4> kVectorGroups = {0, 1, 7, 15};

    static_assert(kPackedC8Bytes == 16, "P3 direct-store contract requires SVL=64");
    for (const BLASLONG matrix_m : kMatrixSizes) {
        const BLASLONG ldc8 = matrix_m + 23;  // Deliberately not a 4-byte stride.
        const std::array<BLASLONG, 3> tile_rows = {0, 128, matrix_m - 256};
        for (const BLASLONG matrix_n : kMatrixSizes) {
            const std::array<BLASLONG, 3> tile_cols = {0, 7, matrix_n - 128};
            for (const BLASLONG tile_row : tile_rows) {
                for (const BLASLONG tile_col : tile_cols) {
                    const uint64_t c8_tile = kSyntheticC8Base +
                        static_cast<uint64_t>(tile_row) +
                        static_cast<uint64_t>(tile_col) * ldc8;
                    for (const BLASLONG relative_row : kRelativeRows) {
                        for (const BLASLONG pC_slot : {BLASLONG{0}, BLASLONG{1},
                                                       BLASLONG{2}, BLASLONG{3}}) {
                            // pC0..pC3 differ by one direct C8 ldc8 stride.
                            const uint64_t direct_pc0 = c8_tile +
                                static_cast<uint64_t>(relative_row) +
                                static_cast<uint64_t>(pC_slot) * ldc8;
                            const uint64_t retired_virtual_pc0 = c8_tile + 4 *
                                (static_cast<uint64_t>(relative_row) +
                                 static_cast<uint64_t>(pC_slot) * ldc8);
                            for (const BLASLONG group : kVectorGroups) {
                                // SAVE_ZACOL_*VL retains this 4*LDC group stride.
                                const BLASLONG col = tile_col + pC_slot + 4 * group;
                                if (col >= matrix_n) continue;
                                const uint64_t direct_pc = direct_pc0 +
                                    static_cast<uint64_t>(4 * group) * ldc8;
                                const uint64_t retired_virtual_pc = retired_virtual_pc0 +
                                    static_cast<uint64_t>(16 * group) * ldc8;
                                for (BLASLONG off = 0; off < 4; ++off) {
                                    const BLASLONG row = tile_row + relative_row +
                                        off * kPackedC8Bytes;
                                    if (row + kPackedC8Bytes > matrix_m) continue;

                                    const uint64_t direct_dst = direct_pc +
                                        static_cast<uint64_t>(off * kPackedC8Bytes);
                                    const uint64_t expected_dst = kSyntheticC8Base +
                                        static_cast<uint64_t>(row) +
                                        static_cast<uint64_t>(col) * ldc8;
                                    require(direct_dst == expected_dst,
                                            "direct C8 cursor base-address mapping mismatch");

                                    const uint64_t retired_dst = c8_tile +
                                        ((retired_virtual_pc +
                                          static_cast<uint64_t>(off * kSmeVectorBytes) -
                                          c8_tile) >> 2);
                                    require(direct_dst == retired_dst,
                                            "direct C8 cursor does not match fixed-SVL legacy mapping");

                                    for (BLASLONG lane = 0; lane < kPackedC8Bytes; ++lane) {
                                        require(direct_dst + static_cast<uint64_t>(lane) ==
                                                    expected_dst + static_cast<uint64_t>(lane),
                                                "direct C8 cursor vector-lane mapping mismatch");
                                    }
                                    const uint64_t last_offset =
                                        direct_dst - kSyntheticC8Base + kPackedC8Bytes - 1;
                                    require(last_offset / ldc8 == static_cast<uint64_t>(col) &&
                                                last_offset % ldc8 <
                                                static_cast<uint64_t>(matrix_m),
                                            "direct C8 cursor mapping touched C8 padding");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
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
    std::vector<int8_t> c8(static_cast<size_t>(kLdc8) * kN, kC8Padding);
    Invocation invocation;
    invocation.c8 = c8.data();
    invocation.mode = mode;
    invocation.c8_behavior = c8_behavior;
    g_invocation = &invocation;
    cblas_gemm_s8s8s8(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        kM, kN, kK, 1.0f, a.data(), kLda, 0, b.data(), kLdb, 0, 0.0f,
        sa.data(), sb.data(), c8.data(), kLdc8, mode);
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
            const int32_t accumulator = expected_accumulator(row, col);
            const int8_t expected8 = c8_behavior == C8Behavior::Inverse
                ? fusion_test::inverse_integer(accumulator, mode)
                : kKernelMarker;
            const size_t c8_index = static_cast<size_t>(col) * kLdc8 + row;
            if (c8[c8_index] != expected8) {
                fail("mocked C8-only production output mismatch mode=" +
                     std::to_string(mode) + " threads=" + std::to_string(threads) +
                     " row=" + std::to_string(row) + " col=" + std::to_string(col));
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
    void*, BLASLONG, int8_t* c8, BLASLONG ldc8, void* buf) {
    Invocation* invocation = g_invocation;
    const auto* params = static_cast<const Int8FusedStoreParams*>(buf);
    BLASLONG base_row = 0;
    BLASLONG base_col = 0;
    const Int8FusedStoreParams expected = invocation != nullptr
        ? expected_scaling(invocation->mode)
        : Int8FusedStoreParams{};
    if (invocation == nullptr || params == nullptr || k != kK || alpha != 1.0f ||
        params->c8 != c8 || ldc8 != kLdc8 ||
        params->modulus != expected.modulus || params->reserved != 0 ||
        params->inv_p != expected.inv_p || params->neg_p != expected.neg_p ||
        rows <= 0 || cols <= 0 ||
        !tile_origin(params, &base_row, &base_col) || base_row + rows > kM ||
        base_col + cols > kN) {
        if (invocation != nullptr) invocation->invalid.store(true);
        return;
    }

    for (BLASLONG col = 0; col < cols; ++col) {
        for (BLASLONG row = 0; row < rows; ++row) {
            const int32_t value = expected_accumulator(static_cast<int>(base_row + row),
                                               static_cast<int>(base_col + col));
            params->c8[col * ldc8 + row] =
                invocation->c8_behavior == C8Behavior::Inverse
                    ? fusion_test::inverse_integer(value, invocation->mode)
                    : kKernelMarker;
        }
    }
    ++invocation->kernel_calls;
}

int main() {
    try {
        verify_direct_c8_cursor_mapping();
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
        std::cout << "PASS host production-driver mock: direct C8 cursor address "
                     "mapping, C8-only API wiring, all moduli, C8 stride, padding, "
                     "1/32-thread tiling, and absence of a post-kernel C++ "
                     "inverse-scaling pass verified.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL host production-driver mock: " << error.what() << '\n';
        return 1;
    }
}
