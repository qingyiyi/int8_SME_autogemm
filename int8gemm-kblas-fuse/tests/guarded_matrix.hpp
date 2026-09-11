#ifndef INT8_FUSION_TEST_GUARDED_MATRIX_HPP
#define INT8_FUSION_TEST_GUARDED_MATRIX_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace fusion_test {

// Canary-check writes before/after the allocation and into each column's padding.
// This does not detect out-of-bounds reads by assembly.
template <typename T>
class GuardedMatrix {
public:
    GuardedMatrix(int rows, int cols, int ld, T canary)
        : rows_(rows), cols_(cols), ld_(ld), canary_(canary),
          storage_(checked_size(rows, cols, ld), canary) {}

    T* data() { return storage_.data() + kGuard; }
    const T* data() const { return storage_.data() + kGuard; }

    void reset(T active_value) {
        std::fill(storage_.begin(), storage_.end(), canary_);
        for (int j = 0; j < cols_; ++j) {
            std::fill_n(data() + static_cast<size_t>(j) * ld_, rows_, active_value);
        }
    }

    void check_guards(const std::string& name) const {
        for (size_t i = 0; i < kGuard; ++i) {
            if (storage_[i] != canary_ ||
                storage_[storage_.size() - kGuard + i] != canary_) {
                throw std::runtime_error(name + ": outer output guard modified");
            }
        }
        for (int j = 0; j < cols_; ++j) {
            for (int i = rows_; i < ld_; ++i) {
                if (data()[static_cast<size_t>(j) * ld_ + i] != canary_) {
                    throw std::runtime_error(name + ": output padding modified at row=" +
                                             std::to_string(i) + " col=" + std::to_string(j));
                }
            }
        }
    }

private:
    static size_t checked_size(int rows, int cols, int ld) {
        if (rows <= 0 || cols <= 0 || ld < rows) {
            throw std::invalid_argument("invalid guarded matrix dimensions");
        }
        return 2 * kGuard + static_cast<size_t>(ld) * cols;
    }
    static constexpr size_t kGuard = 64;
    int rows_, cols_, ld_;
    T canary_;
    std::vector<T> storage_;
};

}  // namespace fusion_test
#endif
