// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive.hpp"

namespace cldnn {
struct range: public primitive_base<range> {
    CLDNN_DECLARE_PRIMITIVE(range)

    range(const primitive_id &id, const std::vector<input_info> &inputs, const layout &output_layout) :
        primitive_base {id, inputs, {output_layout.data_padding}}, output_layout{output_layout} {
    }
    layout output_layout;
};
}  // namespace cldnn
