// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief
/// @details
struct gather_elements : public primitive_base<gather_elements> {
    CLDNN_DECLARE_PRIMITIVE(gather_elements)

    /// @brief Constructs gather_elements primitive.
    ///
    /// @param id                   This primitive id.
    /// @param data                 Input data primitive id.
    /// @param indices              Input indexes primitive id.
    /// @param axis                 Target axis.
    gather_elements(const primitive_id& id,
              const primitive_id& data,
              const primitive_id& indices,
              const uint8_t axis,
              const primitive_id& ext_prim_id = "",
              const padding& output_padding = padding())
              :primitive_base(id, {data, indices}, ext_prim_id, output_padding), axis(axis) {}

    /// @brief GatherElements input_rank
    uint8_t axis;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
