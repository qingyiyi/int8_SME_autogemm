#include "guarded_matrix.hpp"
#include "reference.hpp"

#include <cfenv>
#include <iostream>
#include <limits>
#include <random>
#include <string>

namespace {
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void test_guards() {
    fusion_test::GuardedMatrix<int32_t> matrix(3, 2, 7, 0x5a5a5a5a);
    matrix.reset(0);
    matrix.check_guards("clean");
    // All three corrupted positions stay within the backing vector allocation.
    for (int position : {-1, 3, 14}) {
        matrix.reset(0);
        matrix.data()[position] = 123;
        bool caught = false;
        try { matrix.check_guards("injected"); }
        catch (const std::runtime_error&) { caught = true; }
        require(caught, "guard checker missed injected corruption");
    }
    matrix.reset(0);
    matrix.check_guards("reset");
}

void test_gemm_oracle() {
    constexpr int m = 3, n = 2, lda = 5, ldb = fusion_test::kFixedK + 7;
    std::vector<int8_t> a(lda * fusion_test::kFixedK, 99), b(ldb * n, 99);
    for (int l = 0; l < fusion_test::kFixedK; ++l) {
        for (int i = 0; i < m; ++i) a[l * lda + i] = static_cast<int8_t>(i - 1);
        for (int j = 0; j < n; ++j) b[j * ldb + l] = static_cast<int8_t>(j + 2);
    }
    const auto result = fusion_test::gemm_reference(m, n, a.data(), lda, b.data(), ldb);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            require(result[j * m + i] == (i - 1) * (j + 2) * fusion_test::kFixedK,
                    "GEMM oracle layout/stride mismatch");
        }
    }
}
}  // namespace

int main() {
    try {
        require(std::fesetround(FE_TONEAREST) == 0 && std::fegetround() == FE_TONEAREST,
                "FE_TONEAREST unavailable");
        size_t checked = 0;
        const auto check = [&](int32_t value, unsigned mode) {
            const int actual = fusion_test::inverse_fp64(value, mode);
            const int expected = fusion_test::inverse_integer(value, mode);
            const int fused_scalar = fusion_test::inverse_fused_scalar_model(value, mode);
            require(actual == expected && fused_scalar == expected,
                    "inverse mismatch: C=" + std::to_string(value) +
                    " mode=" + std::to_string(mode) + " fp64=" + std::to_string(actual) +
                    " integer=" + std::to_string(expected) +
                    " fused_scalar=" + std::to_string(fused_scalar));
            ++checked;
        };
        std::mt19937 rng(20260910);
        for (unsigned mode = 0; mode <= fusion_test::kModuli.size(); ++mode) {
            for (int value = -32768; value <= 32767; ++value) check(value, mode);
            for (int32_t value : {
                     std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min() + 1,
                     -16777217, -16777216, 16777215, 16777216, 16777217,
                     std::numeric_limits<int32_t>::max() - 1, std::numeric_limits<int32_t>::max()}) {
                check(value, mode);
            }
            for (int sample = 0; sample < 20000; ++sample) {
                const int64_t value = static_cast<int64_t>(rng()) - 2147483648LL;
                check(static_cast<int32_t>(value), mode);
            }
            if (mode != 0) {
                const int64_t p = fusion_test::kModuli[mode - 1];
                for (int64_t q : {-8000000, -10000, -1, 0, 1, 10000, 8000000}) {
                    for (int64_t delta : {-p / 2 - 1, -p / 2, int64_t{-1}, int64_t{0},
                                         int64_t{1}, p / 2, p / 2 + 1}) {
                        const int64_t value = q * p + delta;
                        if (value >= std::numeric_limits<int32_t>::min() &&
                            value <= std::numeric_limits<int32_t>::max()) {
                            check(static_cast<int32_t>(value), mode);
                        }
                    }
                }
            }
        }
        require(fusion_test::low_byte(128) == -128 && fusion_test::low_byte(255) == -1 &&
                fusion_test::low_byte(-129) == 127, "low-byte wrap semantics changed");
        require(fusion_test::inverse_fp64(16777217, 1) == 2, "FP64 precision regression");
        const float rounded = static_cast<float>(16777217);
        require(fusion_test::inverse_fp64(static_cast<int32_t>(rounded), 1) == 1,
                "FP32 counterexample no longer exercises the intended loss");
        for (auto reference : {fusion_test::inverse_integer, fusion_test::inverse_fp64}) {
            bool caught = false;
            try { (void)reference(1, 20); }
            catch (const std::invalid_argument&) { caught = true; }
            require(caught, "invalid mode was not rejected by reference");
        }
        test_guards();
        test_gemm_oracle();
        std::cout << "PASS host reference: " << checked
                  << " inverse cases; guard fault injection; fixed-K GEMM oracle.\n"
                  << "This is NOT an SME kernel acceptance test.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL host reference: " << e.what() << '\n';
        return 1;
    }
}
