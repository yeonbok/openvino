// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "gemm_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include <utility>
#include <algorithm>

#include "matmul_shape_inference.hpp"

namespace cldnn {
primitive_type_id gemm::type_id() {
    static primitive_type_base<gemm> instance;
    return &instance;
}

layout gemm_inst::calc_output_layout(gemm_node const& node, kernel_impl_params const& impl_param) {
    auto prim = impl_param.typed_desc<gemm>();

    auto input0_layout = impl_param.input_layouts[0];
    auto input1_layout = impl_param.input_layouts[1];
    bool transpose_input0 = prim->transpose_input0;
    bool transpose_input1 = prim->transpose_input1;

    auto M = !transpose_input0 ? input0_layout.spatial(1) : input0_layout.spatial(0);
    auto N = !transpose_input1 ? input1_layout.spatial(0) : input1_layout.spatial(1);

    auto output_size = input0_layout.get_tensor();

    for (size_t i = 1; i < prim->input_size(); ++i) {
        auto input_layout = impl_param.input_layouts[i];
        output_size = tensor::max(output_size, input_layout.get_tensor());
    }

    output_size.spatial[0] = N;
    output_size.spatial[1] = M;
    auto output_type = input0_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && prim->output_data_type)
        output_type = *prim->output_data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    auto output_format = input0_layout.format;

    return layout(output_type, output_format, output_size, prim->output_padding);
}

std::vector<layout> gemm_inst::calc_output_layouts(gemm_node const& node, kernel_impl_params const& impl_param) {
    auto prim = impl_param.typed_desc<gemm>();

    auto input0_layout = impl_param.input_layouts[0];
    auto input1_layout = impl_param.input_layouts[1];

    ov::op::v0::MatMul op;
    op.set_transpose_a(prim->transpose_input0);
    op.set_transpose_b(prim->transpose_input1);

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes = {
        node.get_dependency(0).get_output_layout().get_partial_shape(),
        node.get_dependency(1).get_output_layout().get_partial_shape(),
    };

    shape_infer(&op, input_shapes, output_shapes);

    auto output_type = input0_layout.data_type;
    if ((output_type == data_types::u8 || output_type == data_types::i8) && prim->output_data_type)
        output_type = *prim->output_data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    auto output_format = input0_layout.format;

    return { layout(output_shapes[0], output_type, output_format, prim->output_padding) };
}

std::string gemm_inst::to_string(gemm_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto alpha = desc->alpha;
    auto beta = desc->beta;
    auto transpose_input0 = desc->transpose_input0 ? " true" : "false";
    auto transpose_input1 = desc->transpose_input1 ? " true" : "false";
    std::stringstream primitive_description;

    json_composite gemm_info;
    for (size_t i = 0; i < node.inputs_count(); i++) {
        gemm_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    gemm_info.add("alpha", alpha);
    gemm_info.add("beta", beta);
    gemm_info.add("trasnpose_input0", transpose_input0);
    gemm_info.add("transpose_input1", transpose_input1);
    node_info->add("gemm info", gemm_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

gemm_inst::typed_primitive_inst(network& network, gemm_node const& node) : parent(network, node) { }
}  // namespace cldnn
