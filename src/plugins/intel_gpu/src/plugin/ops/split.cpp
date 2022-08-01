// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/split.hpp"
#include "ngraph/op/variadic_split.hpp"

#include "intel_gpu/primitives/crop.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonSplitOp(Program& p, const std::shared_ptr<ngraph::Node>& op) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto inputDims = op->get_input_partial_shape(0);
    InferenceEngine::SizeVector startOffset(inputDims.size());

    bool is_single_out_split = op->get_output_size() == 1;

    for (size_t i = 0; i < op->get_output_size(); i++) {
        std::string outLayerName = layerName + (is_single_out_split ? "" : ".out" + std::to_string(i));
        const auto outLayerDims = op->get_output_partial_shape(i);
        NGRAPH_SUPPRESS_DEPRECATED_START
        if (outLayerDims.size() != startOffset.size()) {
            IE_THROW() << "Invalid dimesions in split layer: " << op->get_friendly_name()
                               << " output: " <<  op->get_output_tensor_name(i);
        }
        #if 0
        for (size_t i = 0; i < inputDims.size(); i++) {
            if ((outLayerDims[i] + startOffset[i]) > inputDims[i]) {
                IE_THROW() << "Invalid dimesions in split layer: " << op->get_friendly_name()
                                   << " output: " <<  op->get_output_tensor_name(i);
            }
        }
        NGRAPH_SUPPRESS_DEPRECATED_END
        #endif
        auto offsetTensor = tensor_from_dims(startOffset, 0);

        auto cropPrim = cldnn::crop(outLayerName, inputPrimitives, offsetTensor, {}, i, op->get_output_size(), {}, op);
        p.add_primitive(*op, cropPrim);

        if (outLayerDims.is_static()) {
            for (size_t i = 0; i < inputDims.size(); i++) {
                if (outLayerDims[i] != inputDims[i]) {
                    startOffset[i] += outLayerDims.to_shape()[i];
                }
            }
        }
    }
}

static void CreateSplitOp(Program& p, const std::shared_ptr<ngraph::op::v1::Split>& op) {
    validate_inputs_count(op, {2});
    CreateCommonSplitOp(p, op);
}

static void CreateVariadicSplitOp(Program& p, const std::shared_ptr<ngraph::op::v1::VariadicSplit>& op) {
    validate_inputs_count(op, {3});
    CreateCommonSplitOp(p, op);
}

REGISTER_FACTORY_IMPL(v1, Split);
REGISTER_FACTORY_IMPL(v1, VariadicSplit);

}  // namespace intel_gpu
}  // namespace ov
