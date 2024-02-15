// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/rope.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

RoPE::RoPE(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> RoPE::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<RoPE>(new_args, m_config);
}

void RoPE::validate_and_infer_types() {
    auto input_pshape = get_input_partial_shape(0);
    auto input_slice_size = m_config.slice_stop - m_config.slice_start;

    if (m_config.is_qwen) {
        // Qwen specific RoPE
        // input  [batch_size, cur_length, (hidden_states_q + hidden_states_k + hidden_states_v)]
        // output [batch_size, cur_length, head_cnt, head_size]
        set_output_type(
            0,
            get_input_element_type(0),
            {input_pshape[0], input_pshape[1], ov::Dimension(m_config.head_cnt), ov::Dimension(m_config.head_size)});
        return;
    }

    if (m_config.is_chatglm) {
        // chatGLM specific RoPE
        // input  [length, batch_size, (hidden_states_q + hidden_states_k + hidden_states_v)]
        // output [length, batch_size, head_cnt, hidden_states_k]
        set_output_type(
            0,
            get_input_element_type(0),
            {input_pshape[0], input_pshape[1], ov::Dimension(m_config.head_cnt), ov::Dimension(m_config.head_size)});
        return;
    }

    if (input_slice_size > 0) {
        input_pshape[3] = input_slice_size;
    }
    if (m_config.input_trans0213) {
        // transpose 0213 ([B,L,H,S]=>[B,H,L,S]) happens before RoPE
        std::swap(input_pshape[2], input_pshape[1]);
    } else if (m_config.is_interleaved) {
        // transpose 0213 ([B,L,H,S]=>[B,H,L,S]) happens after RoPE
        std::swap(input_pshape[2], input_pshape[1]);
    }

    set_output_type(0, get_input_element_type(0), input_pshape);
}

bool RoPE::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.start_structure("config");
    visitor.on_attribute("slice_start", m_config.slice_start);
    visitor.on_attribute("slice_stop", m_config.slice_stop);
    visitor.on_attribute("input_trans0213", m_config.input_trans0213);
    visitor.on_attribute("is_interleaved", m_config.is_interleaved);
    visitor.on_attribute("rotary_ndims", m_config.rotary_ndims);
    visitor.on_attribute("is_chatglm", m_config.is_chatglm);
    visitor.on_attribute("is_qwen", m_config.is_qwen);
    visitor.on_attribute("head_cnt", m_config.head_cnt);
    visitor.on_attribute("head_size", m_config.head_size);
    visitor.on_attribute("gather_position_arg_id", m_config.gather_position_arg_id);
    visitor.finish_structure();
    return true;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
