// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/select.hpp"

#include "intel_gpu/primitives/select.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace intel_gpu {

static void CreateSelectOp(Program& p, const std::shared_ptr<ngraph::op::v1::Select>& op) {
    p.ValidateInputs(op, {3});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto output_pshape = op->get_output_partial_shape(0);
    auto output_rank = output_pshape.size();

    auto broadcast_type = op->get_auto_broadcast();

    if (broadcast_type.m_type != ngraph::op::AutoBroadcastType::NONE &&
        broadcast_type.m_type != ngraph::op::AutoBroadcastType::NUMPY) {
        IE_THROW() << "Unsupported broadcast type (" << broadcast_type.m_type << ") in layer " + op->get_friendly_name();
    }

    if (broadcast_type.m_type == ngraph::op::AutoBroadcastType::NUMPY) {
        // Preprocess inputs
        for (size_t i = 0; i < inputPrimitives.size(); ++i) {
            auto input_pshape = op->get_input_partial_shape(i);
            OPENVINO_ASSERT(input_pshape.is_static(), "Dynamic shapes are not supported for v1::Select with NUMPY mode yet");
            auto input_shape = input_pshape.to_shape();
            auto input_rank = input_shape.size();

            // Add reorder if changing number of dimensions requires changing format
            auto targetFormat = cldnn::format::get_default_format(output_rank);

            if (targetFormat.value != cldnn::format::get_default_format(input_rank).value) {
                auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
                auto targetDatatype = DataTypeFromPrecision(op->get_input_element_type(i));
                auto reorderPrim = cldnn::reorder(reorderName,
                                                  inputPrimitives[i],
                                                  targetFormat,
                                                  targetDatatype,
                                                  std::vector<float>(),
                                                  cldnn::reorder_mean_mode::subtract,
                                                  op->get_friendly_name());

                p.AddPrimitive(reorderPrim);
                p.AddInnerPrimitiveToProfiler(reorderName, layerName, op);

                inputPrimitives[i] = reorderName;
            }

            // Reshape input if they differ or select specific shape matches default one
            if (input_rank != output_rank || input_rank < 4) {
                auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

                // Extend input dimensions to the same size as output dimensions by prepending ones
                input_shape.insert(input_shape.begin(), output_rank - input_rank, 1ul);

                auto reshapePrim = cldnn::reshape(reshapeName, inputPrimitives[i], input_pshape, op->get_friendly_name());

                p.AddPrimitive(reshapePrim);
                p.AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

                inputPrimitives[i] = reshapeName;
            }
        }
    }

    std::string bc_string = broadcast_type.m_type == ngraph::op::AutoBroadcastType::NUMPY ? "numpy" : "none";

    auto selectPrim = cldnn::select(layerName,
                                    inputPrimitives[0],
                                    inputPrimitives[1],
                                    inputPrimitives[2],
                                    op->get_friendly_name(),
                                    cldnn::padding(),
                                    bc_string);

    p.AddPrimitive(selectPrim);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v1, Select);

}  // namespace intel_gpu
}  // namespace ov
