// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {
/// \brief Operator performing Root Mean Square Normalization
///
/// \note Performs re-scaling invariance and regularizes the summed input according to RMS statistics
class ShapeOfScale : public ov::op::Op {
public:
    OPENVINO_OP("ShapeOfScale", "gpu_opset");

    ShapeOfScale() = default;
    /// \brief Constructs an RMS operation.
    ///
    /// \param data Input tensor with data
    /// \param output_type Output element type
    ShapeOfScale(const Output<Node>& data,
        const ov::element::Type output_type = ov::element::undefined);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

private:
    ov::element::Type m_output_type;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
