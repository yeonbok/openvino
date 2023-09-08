// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"

namespace ov {
namespace op {
namespace internal {
using FullyConnected = ov::intel_gpu::op::FullyConnected;
using FullyConnectedCompressed = ov::intel_gpu::op::FullyConnectedCompressed;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateFullyConnectedCompressedOp(ProgramBuilder& p, const std::shared_ptr<op::FullyConnectedCompressed>& op) {
    validate_inputs_count(op, {3, 4});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto input_name = inputs[0].pid;
    auto weights_name = inputs[1].pid;
    auto scale_name = inputs[2].pid;
    auto zp_name = inputs.size() == 4 ? inputs[3].pid : "";

    auto fc = cldnn::fully_connected(primitive_name,
                                     cldnn::input_info(input_name),
                                     weights_name,
                                     "",
                                     scale_name,
                                     zp_name,
                                     cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                     cldnn::padding(),
                                     op->get_input_partial_shape(0).size(),
                                     op->get_input_partial_shape(1).size());

    p.add_primitive(*op, fc);
}

static void CreateFullyConnectedOp(ProgramBuilder& p, const std::shared_ptr<op::FullyConnected>& op) {
    validate_inputs_count(op, {2});
    auto inputs = p.GetInputInfo(op);
    std::string layerName = layer_type_name_ID(op);

    auto input_name = inputs[0].pid;
    auto weights_name = inputs[1].pid;

    auto shape_a = op->get_input_partial_shape(0);
    auto shape_b = op->get_input_partial_shape(1);

    auto rank_a = shape_a.rank().get_length();
    auto rank_b = shape_b.rank().get_length();

    auto fcPrim = cldnn::fully_connected(layerName,
                                         cldnn::input_info(input_name),
                                         weights_name,
                                         "",
                                         cldnn::element_type_to_data_type(op->get_output_element_type(0)),
                                         cldnn::padding(),
                                         rank_a,
                                         rank_b);

    p.add_primitive(*op, fcPrim);

    if (shape_a.size() > 3 && !p.use_new_shape_infer()) {
        auto lastLayerName = layerName;
        auto outReshapeName = layerName + "_cldnn_out_reshape";

        // add reorder
        auto outDims = op->get_output_shape(0);
        auto outTensor = tensor_from_dims(outDims);

        if (outDims.size() > 4) {
            cldnn::format outputFormat = cldnn::format::bfyx;
            switch (outDims.size()) {
                case 5: outputFormat = cldnn::format::bfzyx; break;
                case 6: outputFormat = cldnn::format::bfwzyx; break;
                default: break;
            }

            cldnn::primitive_id reorderId = "reorder:" + outReshapeName + "_reorder";
            cldnn::layout outputLayout(cldnn::element_type_to_data_type(op->get_output_element_type(0)), outputFormat, outTensor);
            auto reorder_prim = cldnn::reorder(reorderId, cldnn::input_info(layerName), outputLayout);
            p.add_primitive(*op, reorder_prim);
            lastLayerName = reorderId;
        }

        // add reshape
        auto outReshapePrim = cldnn::reshape(outReshapeName, cldnn::input_info(lastLayerName), outTensor);
        p.add_primitive(*op, outReshapePrim);
    }
}

REGISTER_FACTORY_IMPL(internal, FullyConnected);
REGISTER_FACTORY_IMPL(internal, FullyConnectedCompressed);

}  // namespace intel_gpu
}  // namespace ov
