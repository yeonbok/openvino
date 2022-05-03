// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/non_zero.hpp"

#include "intel_gpu/primitives/non_zero.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

static void CreateNonZeroOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    p.ValidateInputs(op, {1});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    cldnn::primitive_id countID = layerName + "_count";
    auto primitive1 = cldnn::count_nonzero(countID,
                                           inputPrimitives[0],
                                           op->get_friendly_name());

    auto primitive2 = cldnn::gather_nonzero(layerName,
                                            inputPrimitives[0],
                                            countID,
                                            op->get_friendly_name());

    p.AddPrimitive(primitive1);
    p.AddPrimitive(primitive2);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v3, NonZero);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
