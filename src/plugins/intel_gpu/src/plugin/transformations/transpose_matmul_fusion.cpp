// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gemm.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "transpose_matmul_fusion.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov {
namespace intel_gpu {

namespace {
std::vector<int64_t> default_order(size_t rank) {
    std::vector<int64_t> order(rank);
    std::iota(order.begin(), order.end(), 0);
    return order;
}
}  // namespace

class TransposeMatMulMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeMatMulMatcher", "0");
    TransposeMatMulMatcher();
};

class TransposeMatMulTransposeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TransposeMatMulTransposeMatcher", "0");
    TransposeMatMulTransposeMatcher();
};

TransposeMatMulFusion::TransposeMatMulFusion() {
    add_matcher<TransposeMatMulTransposeMatcher>();
    add_matcher<TransposeMatMulMatcher>();
}


TransposeMatMulMatcher::TransposeMatMulMatcher() {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr;
    };
    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m});
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m});

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b });

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto users = matmul->get_output_target_inputs(0);
        if (users.size() == 1 && dynamic_cast<ov::op::v1::Transpose*>(users.begin()->get_node()) != nullptr) {
            return false;
        }

        auto order_a = default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = default_order(matmul->get_output_partial_shape(0).size());

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
        }

        auto input_a = pattern_map.at(input_a_m).get_node_shared_ptr();
        auto input_b = pattern_map.at(input_b_m).get_node_shared_ptr();

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(matmul->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(matmul, gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matmul_m, "TransposeMatMulMatcher");
    this->register_matcher(m, callback);
}

TransposeMatMulTransposeMatcher::TransposeMatMulTransposeMatcher() {
    auto not_transpose = [](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Transpose>(output.get_node_shared_ptr()) == nullptr;
    };
    auto input_a_m = any_input(not_transpose);
    auto input_b_m = any_input(not_transpose);
    auto transpose_a_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_b_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_a_m = wrap_type<ov::op::v1::Transpose>({input_a_m, transpose_a_order_m});
    auto transpose_b_m = wrap_type<ov::op::v1::Transpose>({input_b_m, transpose_b_order_m});

    auto matmul_in_a = std::make_shared<Or>(OutputVector{input_a_m, transpose_a_m});
    auto matmul_in_b = std::make_shared<Or>(OutputVector{input_b_m, transpose_b_m});

    auto matmul_m = wrap_type<ov::op::v0::MatMul>({ matmul_in_a, matmul_in_b });
    auto transpose_c_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_c_m = wrap_type<ov::op::v1::Transpose>({matmul_m, transpose_c_order_m});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(pattern_map.at(matmul_m).get_node_shared_ptr());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto tranpose_c_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_c_order_m).get_node_shared_ptr());
        auto order_a = default_order(matmul->get_input_partial_shape(0).size());
        auto order_b = default_order(matmul->get_input_partial_shape(1).size());
        auto order_c = tranpose_c_order->cast_vector<int64_t>();

        if (pattern_map.count(transpose_a_m) > 0) {
            auto tranpose_a_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_a_order_m).get_node_shared_ptr());
            order_a = tranpose_a_order->cast_vector<int64_t>();
        }
        if (pattern_map.count(transpose_b_m) > 0) {
            auto tranpose_b_order = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(transpose_b_order_m).get_node_shared_ptr());
            order_b = tranpose_b_order->cast_vector<int64_t>();
        }

        auto input_a = pattern_map.at(input_a_m).get_node_shared_ptr();
        auto input_b = pattern_map.at(input_b_m).get_node_shared_ptr();

        auto gemm = std::make_shared<op::Gemm>(input_a, input_b, order_a, order_b, order_c);
        gemm->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), gemm);
        ov::replace_node(m.get_match_root(), gemm);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(transpose_c_m, "TransposeMatMulTransposeMatcher");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
