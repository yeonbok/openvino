// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gemm.hpp"
#include "matmul_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/matmul.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

Gemm::Gemm(const ov::Output<Node>& A,
           const ov::Output<Node>& B,
           const std::vector<int64_t>& order_a,
           const std::vector<int64_t>& order_b,
           const std::vector<int64_t>& order_c,
           const ov::element::Type output_type)
    : ov::op::v0::MatMul(A, B, false, false)
    , m_order_a(order_a)
    , m_order_b(order_b)
    , m_order_c(order_c)
    , m_output_type(output_type) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> Gemm::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<Gemm>(new_args.at(0), new_args.at(1), m_order_a, m_order_b, m_order_c, m_output_type);
}

void Gemm::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 2,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 2.");

    auto out_shapes = shape_infer(this, std::vector<ov::PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)}, m_order_a, m_order_b, m_order_c);

    auto output_type = m_output_type == ov::element::undefined ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool Gemm::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("order_a", m_order_a);
    visitor.on_attribute("order_b", m_order_b);
    visitor.on_attribute("order_c", m_order_c);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::vector<ov::PartialShape> shape_infer(const Gemm* op,
                                          std::vector<ov::PartialShape> input_shapes,
                                          const std::vector<int64_t>& order_a,
                                          const std::vector<int64_t>& order_b,
                                          const std::vector<int64_t>& order_c) {
    auto transpose_shape = [](const ov::PartialShape shape, const std::vector<int64_t>& order) {
        auto shape_transposed = ov::PartialShape::dynamic(shape.rank());
        for (size_t i = 0; i < order.size(); i++) {
            shape_transposed[i] = shape[order[i]];
        }

        return shape_transposed;
    };
    auto shape_a = input_shapes[0];
    auto shape_b = input_shapes[1];

    auto shape_a_t = transpose_shape(shape_a, order_a);
    auto shape_b_t = transpose_shape(shape_b, order_b);
    auto out_shapes = ov::op::v0::shape_infer(dynamic_cast<const ov::op::v0::MatMul*>(op), std::vector<ov::PartialShape>{shape_a_t, shape_b_t});

    return { transpose_shape(out_shapes[0], order_c) };
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
