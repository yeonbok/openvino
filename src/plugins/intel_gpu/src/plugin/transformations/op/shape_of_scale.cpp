// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/shape_of_scale.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

ShapeOfScale::ShapeOfScale(const Output<Node>& data,
         const ov::element::Type output_type)
    : Op({data}), m_output_type(output_type) {
    validate_and_infer_types();
}

bool ShapeOfScale::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void ShapeOfScale::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
//    set_output_type(0, output_type, get_input_partial_shape(0));
    set_output_type(0, output_type, ov::PartialShape{});
}

std::shared_ptr<Node> ShapeOfScale::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<ShapeOfScale>(new_args.at(0), m_output_type);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
