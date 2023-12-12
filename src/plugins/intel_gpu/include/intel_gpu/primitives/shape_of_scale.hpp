// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

struct shape_of_scale : public primitive_base<shape_of_scale> {
    CLDNN_DECLARE_PRIMITIVE(shape_of_scale);

    shape_of_scale() : primitive_base("", {}) {}

    /// @brief Constructs shape_of_scale primitive
    /// @param id This primitive id
    /// @param input Input primitive id
    /// @param epsilon Epsilon for not dividing by zero while normalizing
    shape_of_scale(const primitive_id& id,
        const input_info& input,
        const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}) {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<shape_of_scale>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<shape_of_scale>::load(ib);
    }
};
}  // namespace cldnn
