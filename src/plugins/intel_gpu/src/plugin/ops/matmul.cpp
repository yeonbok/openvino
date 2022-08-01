// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/matmul.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fake_quantize.hpp"

#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/permute.hpp"

namespace ov {
namespace intel_gpu {

/*
*  get_aligned_shapes function align two input shapes to have the same size and
*  the same batch dimensions (last two dimensions are not comparable).
*  It also checks that dimensions are compatible so in case with two shapes
*  for example: [2, 32, 64] [3, 64, 64] it will raise an exception.
*/

static std::pair<ngraph::Shape, ngraph::Shape> get_aligned_shapes(const ngraph::Shape& shape_a,
                                                                  const ngraph::Shape& shape_b,
                                                                  const std::shared_ptr<ngraph::op::v0::MatMul>& matmul) {
    ngraph::Shape shape_a_aligned(shape_a), shape_b_aligned(shape_b);
    size_t max_size = std::max(shape_a_aligned.size(), shape_b_aligned.size());
    for (size_t i = 0, cnt = max_size - shape_a_aligned.size(); i < cnt; ++i)
        shape_a_aligned.insert(shape_a_aligned.begin(), 1);
    for (size_t i = 0, cnt = max_size - shape_b_aligned.size(); i < cnt; ++i)
        shape_b_aligned.insert(shape_b_aligned.begin(), 1);

    if (matmul->get_transpose_a()) {
        std::swap(*(shape_a_aligned.end() - 1), *(shape_a_aligned.end() - 2));
    }
    if (matmul->get_transpose_b()) {
        std::swap(*(shape_b_aligned.end() - 1), *(shape_b_aligned.end() - 2));
    }

    for (size_t i = 0; i < max_size - 2; ++i) {
        if (shape_a_aligned[i] != shape_b_aligned[i] && shape_a_aligned[i] > 1 && shape_b_aligned[i] > 1) {
            IE_THROW() << "Shapes can't be aligned: " << shape_a_aligned << " " << shape_b_aligned;
        }
        size_t max_value = std::max(shape_a_aligned[i], shape_b_aligned[i]);
        shape_a_aligned[i] = shape_b_aligned[i] = max_value;
    }

    return {shape_a_aligned, shape_b_aligned};
}

static void CreateMatMulOp(Program& p, const std::shared_ptr<ngraph::op::v0::MatMul>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto shape_a = op->get_input_shape(0);
    auto shape_b = op->get_input_shape(1);

    bool is_fc = IsNodeOnConstPath(op->get_input_node_shared_ptr(1));
    is_fc &= std::count_if(shape_b.begin(), shape_b.end(), [](size_t x) { return x != 1; }) <= 2;
    // TODO: This conditions can be relaxed with proper handling in FC path
    is_fc &= shape_b.size() > 1 && shape_a.size() > 1;

    if (is_fc) {
        ngraph::Shape shape_a_aligned, shape_b_aligned;
        std::tie(shape_a_aligned, shape_b_aligned) = get_aligned_shapes(shape_a, shape_b, op);
        if (shape_a_aligned.size() < 2 || shape_b_aligned.size() < 2) {
            IE_THROW() << "MatMul " << op->get_friendly_name() << " shapes are inconsistent.";
        }
        size_t K = *(shape_a_aligned.end() - 1);

        auto inputName = inputPrimitives[0];
        auto weightsName = inputPrimitives[1];

        // Weights normalization
        if (!op->get_transpose_b()) {
            std::vector<uint16_t> transpose_order(shape_b.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            for (auto o = transpose_order.size(); o < 4; o++)
                transpose_order.push_back((uint16_t)o);

            auto permuteName = op->get_friendly_name() + "/transpose_b";
            auto permutePrim = cldnn::permute(permuteName,
                                              weightsName,
                                              transpose_order,
                                              op->get_friendly_name());
            p.AddPrimitive(permutePrim);
            p.AddInnerPrimitiveToProfiler(permuteName, layerName, op);
            weightsName = permuteName;
        }

        // Input normalization
        if (op->get_transpose_a()) {
            std::vector<uint16_t> transpose_order(shape_a.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            for (auto o = transpose_order.size(); o < 4; o++)
                transpose_order.push_back((uint16_t)o);

            auto permuteName = op->get_friendly_name() + "/transpose_a";
            auto permutePrim = cldnn::permute(permuteName,
                                              inputName,
                                              transpose_order,
                                              op->get_friendly_name());
            p.AddPrimitive(permutePrim);
            p.AddInnerPrimitiveToProfiler(permuteName, layerName, op);
            inputName = permuteName;
        }

        bool reshape_fc = shape_a_aligned.size() > 3;

        auto reshape_to_2d = [&](const ngraph::Shape& shape, std::string inputName, size_t features, std::string suffix) -> std::string {
            size_t total = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            std::vector<size_t> reshapeSize = { total / features, features };

            if (total != reshapeSize[0] * reshapeSize[1])
                IE_THROW() << "Inconsistent reshape in Matmul op: " << op->get_friendly_name();

            auto reshapeInName = op->get_friendly_name() + suffix;
            auto reshapeInPrim = cldnn::reshape(reshapeInName,
                                                inputName,
                                                tensor_from_dims(reshapeSize),
                                                op->get_friendly_name());
            p.AddPrimitive(reshapeInPrim);
            p.AddInnerPrimitiveToProfiler(reshapeInName, layerName, op);
            return reshapeInName;
        };

        if (reshape_fc) {
            inputName = reshape_to_2d(shape_a, inputName, shape_a.back(), "_cldnn_reshape_in");
        }

        if (shape_b.size() != 2) {
            weightsName = reshape_to_2d(shape_b, weightsName, K, "_cldnn_reshape_weights");
        }

        auto input_rank = reshape_fc ? 2 : shape_a.size();
        auto fcPrim = cldnn::fully_connected(layerName,
                                             inputName,
                                             weightsName,
                                             "",
                                             DataTypeFromPrecision(op->get_output_element_type(0)),
                                             op->get_friendly_name(),
                                             cldnn::padding(),
                                             input_rank);

        p.AddPrimitive(fcPrim);

        auto lastLayerName = layerName;
        if (reshape_fc) {
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
                cldnn::layout outputLayout(DataTypeFromPrecision(op->get_output_element_type(0)), outputFormat, outTensor);
                p.AddPrimitive(cldnn::reorder(reorderId,
                                            layerName,
                                            outputLayout,
                                            std::vector<float>(),
                                            cldnn::reorder_mean_mode::subtract,
                                            op->get_friendly_name()));
                p.InitProfileInfo(reorderId, "Reorder", false, InferenceEngine::InferenceEngineProfileInfo::EXECUTED, layerName);
                p.AddInnerPrimitiveToProfiler(reorderId, layerName, op);
                lastLayerName = reorderId;
            }

            // add reshape
            auto outReshapePrim = cldnn::reshape(outReshapeName, lastLayerName, outTensor, op->get_friendly_name());

            p.AddPrimitive(outReshapePrim);
            p.AddInnerPrimitiveToProfiler(outReshapeName, lastLayerName, op);

            lastLayerName = outReshapeName;
        }

        p.AddPrimitiveToProfiler(op, lastLayerName);
    } else {
        auto outDims = op->get_output_shape(0);
        auto outDimsN = outDims.size();

        auto gemmSpecificTensor = [](const InferenceEngine::SizeVector& dims) {
            switch (dims.size()) {
            case 2: return cldnn::tensor(cldnn::spatial(dims[1], dims[0]));
            case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::spatial(dims[2], dims[1]));
            case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
            case 5: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[4], dims[3], dims[2]));
            case 6: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[5], dims[4], dims[3], dims[2]));
            default: IE_THROW() << "Invalid dimensions size(" << dims.size() << ") for Gemm layer";
            }
        };

        // Preprocess inputs
        for (size_t i = 0; i < inputPrimitives.size(); ++i) {
            auto inputDims = op->get_input_shape(i);
            auto inputDimsN = inputDims.size();

            // Add reorder if changing number of dimensions requires changing format
            auto targetFormat = cldnn::format::get_default_format(outDimsN);

            if (targetFormat.value != cldnn::format::get_default_format(inputDimsN).value) {
                auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
                auto targetDatatype = DataTypeFromPrecision(op->get_output_element_type(0));
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

            // Reshape input if they differ or gemm specific shape matches default one
            if (inputDimsN != outDimsN || inputDimsN < 4) {
                auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

                // Extend input dimensions by prepending ones
                if (inputDimsN == 1) {
                    // One-dimensional tensors unsqueezing is applied for each input independently.
                    // The axes inserted in this step are not included in the output shape.
                    // * If rank of the **first** input is equal to 1, it is always unsqueezed to 2D tensor **row vector** (regardless of `transpose_a`)
                    // by adding axes with size 1 at ROW_INDEX_DIM, to the **left** of the shape. For example `[S]` will be reshaped to `[1, S]`.
                    // * If rank of the **second** input is equal to 1, it is always unsqueezed to 2D tensor **column vector** (regardless of `transpose_b`)
                    // by adding axes with size 1 at COL_INDEX_DIM, to the **right** of the shape. For example `[S]` will be reshaped to `[S, 1]`.
                    bool transpose = false;
                    if (i == 0) {
                        transpose = op->get_transpose_a();
                        inputDims.insert(inputDims.begin(), 1);
                    } else {
                        transpose = op->get_transpose_b();
                        inputDims.insert(inputDims.end(), 1);
                    }
                    // Specs says that shapes must be unsqueezed regardless of tranpose flag, but primitive implementation always respects transposes
                    // so we have to swap dimensions correspondingly to have consistent shapes.
                    if (transpose) {
                        std::swap(inputDims[0], inputDims[1]);
                    }
                }
                if (inputDimsN < outDimsN)
                    inputDims.insert(inputDims.begin(), outDimsN - inputDimsN, 1ul);

                auto targetShape = gemmSpecificTensor(inputDims);

                auto reshapePrim = cldnn::reshape(reshapeName, inputPrimitives[i], targetShape, op->get_friendly_name());

                p.AddPrimitive(reshapePrim);
                p.AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

                inputPrimitives[i] = reshapeName;
            }
        }

        // Add actual gemm
        auto alpha = 1.0f;
        auto beta = 0.0f;
        auto transA = op->get_transpose_a();
        auto transB = op->get_transpose_b();

        auto gemmPrim = cldnn::gemm(layerName,
                                    inputPrimitives,
                                    DataTypeFromPrecision(op->get_output_element_type(0)),
                                    transA,
                                    transB,
                                    alpha,
                                    beta,
                                    op->get_friendly_name());

        p.AddPrimitive(gemmPrim);

        auto lastLayerName = layerName;

        // Reshape output if gemm specific shape does not match default one
        if (outDimsN < 4) {
            auto outputShape = tensor_from_dims(outDims);
            auto outReshapeName = layerName + "_cldnn_out_reshape";
            auto outReshapePrim = cldnn::reshape(outReshapeName, layerName, outputShape, op->get_friendly_name());

            p.AddPrimitive(outReshapePrim);
            p.AddInnerPrimitiveToProfiler(outReshapeName, layerName, op);

            lastLayerName = outReshapeName;
        }

        p.AddPrimitiveToProfiler(op, lastLayerName);
    }
}

REGISTER_FACTORY_IMPL(v0, MatMul);

}  // namespace intel_gpu
}  // namespace ov
