// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/gather_elements.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/gather_elements.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateGatherElementsOp(Program& p, const std::shared_ptr<ngraph::op::v6::GatherElements>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    int32_t input_rank = static_cast<int32_t>(op->get_input_shape(0).size());
    int32_t indices_rank = static_cast<int32_t>(op->get_input_shape(1).size());

    auto batch_dims = 0;//op->get_batch_dims();

    auto primitive = cldnn::gather_elements(layerName,
                                      inputPrimitives[0],
                                      inputPrimitives[1],
                                      input_rank,
                                      indices_rank,
                                      batch_dims,
                                      true,
                                      op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v6, GatherElements);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
