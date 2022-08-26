// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include <vector>

#include "openvino/core/shape.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief StridedSlice extracts a strided slice of a tensor.
struct strided_slice : public primitive_base<strided_slice> {
    CLDNN_DECLARE_PRIMITIVE(strided_slice)

    /// @brief Constructs strided_slice primitive.
    /// @param id This primitive id.
    /// @param input Input data primitive id.
    /// @param begin_id Begin position primitive id.
    /// @param end_id End position primitive id.
    /// @param strides_id Step of slicing primitive id.
    /// @param begin_mask Array of bits, that provide replace begin[i] to max possible range in that dimension.
    /// @param end_mask Array of bits, that provide replace end[i] to max possible range in that dimension.
    /// @param new_axis_mask Array of bits, that provide adding a new length 1 dimension at ith position in the output tensor.
    /// @param shrink_axis_mask Array of bits, that provide shrinks the dimensionality by 1, taking on the value at index begin[i].
    /// @param ellipsis_mask Array of bits, that provide inserts missing dimensions on a position of a non-zero bit.
    /// @param out_size Size of output tensor
    strided_slice(const primitive_id& id,
                  const primitive_id& input,
                  const primitive_id& begin_id,
                  const primitive_id& end_id,
                  const primitive_id& strides_id,
                  const std::vector<int64_t>& begin_mask,
                  const std::vector<int64_t>& end_mask,
                  const std::vector<int64_t>& new_axis_mask,
                  const std::vector<int64_t>& shrink_axis_mask,
                  const std::vector<int64_t>& ellipsis_mask,
                  const ov::PartialShape out_size,
                  const primitive_id& ext_prim_id = "",
                  const padding& output_padding = padding())
        : primitive_base(id, {input, begin_id, end_id, strides_id}, ext_prim_id, output_padding),
          begin_mask(begin_mask),
          end_mask(end_mask),
          new_axis_mask(new_axis_mask),
          shrink_axis_mask(shrink_axis_mask),
          ellipsis_mask(ellipsis_mask),
          out_size(out_size) {}

    /// @brief Array of bits, that provide replace begin[i] to max possible range in that dimension.
    std::vector<int64_t> begin_mask;
    /// @brief Array of bits, that provide replace end[i] to max possible range in that dimension.
    std::vector<int64_t> end_mask;
    /// @brief Array of bits, that provide adding a new length 1 dimension at ith position in the output tensor.
    std::vector<int64_t> new_axis_mask;
    /// @brief Array of bits, that provide shrinks the dimensionality by 1, taking on the value at index begin[i].
    std::vector<int64_t> shrink_axis_mask;
    /// @brief Array of bits, that provide inserts missing dimensions on a position of a non-zero bit.
    std::vector<int64_t> ellipsis_mask;
    /// @brief Size of output tensor
    ov::PartialShape out_size;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
