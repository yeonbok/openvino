// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/space_to_batch.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/space_to_batch.hpp"

namespace ov {
namespace intel_gpu {

static void CreateSpaceToBatchOp(Program& p, const std::shared_ptr<ngraph::op::v1::SpaceToBatch>& op) {
    p.ValidateInputs(op, {4});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto rank = op->get_input_shape(0).size();
    auto format = cldnn::format::get_default_format(rank);

    std::vector<cldnn::tensor> inputs;
    inputs.reserve(3);

    for (size_t i = 1; i < 4; ++i) {
        auto inConst = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(i));
        if (!inConst)
            IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

        std::vector<int32_t> sizes = inConst->cast_vector<int32_t>();
        int32_t default_size = i == 1 ? 1 : 0;
        for (size_t s = sizes.size(); s < rank; s++) {
            sizes.push_back(default_size);
        }
        inputs.emplace_back(format, sizes, default_size);
    }
    auto out_size = tensor_from_dims(op->get_output_shape(0));

    auto batchToSpacePrim = cldnn::space_to_batch(layerName,
                                                  inputPrimitives[0], // input
                                                  inputs[0],          // block_shape
                                                  inputs[1],          // crops_begin
                                                  inputs[2],          // crops_end
                                                  out_size,
                                                  op->get_friendly_name());

    p.AddPrimitive(batchToSpacePrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, SpaceToBatch);

}  // namespace intel_gpu
}  // namespace ov
