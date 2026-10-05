// Output selection only: never alter dense-ALL decode/head computation.
#pragma once
#include "nlohmann/json.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace block_teacher {
using json = nlohmann::json;
inline std::vector<size_t> block_indices(size_t prompt_length, size_t token_count) {
    std::vector<size_t> result;
    // Exactly Python range(prompt_length - 1, token_count - 7, 7).
    for (size_t anchor = prompt_length - 1; anchor + 7 < token_count; anchor += 7)
        for (size_t row = anchor; row < anchor + 7; ++row) result.push_back(row);
    return result;
}
inline std::vector<size_t> explicit_indices(const json & indices, size_t token_count) {
    if (!indices.is_array() || indices.empty()) throw std::runtime_error("nonempty indexed logit row map required");
    std::vector<size_t> result;
    for (const auto & item : indices) {
        if (!item.is_number_integer() || item.get<int64_t>() < 0)
            throw std::runtime_error("indexed logit positions must be integers");
        const auto row = item.get<size_t>();
        if (row >= token_count || (!result.empty() && row <= result.back()))
            throw std::runtime_error("indexed logit positions must be sorted unique in-range");
        result.push_back(row);
    }
    return result;
}
inline json block_selection() {
    return {{"kind", "block_anchors"}, {"stride", 7}, {"horizon", 7}};
}
} // namespace block_teacher
