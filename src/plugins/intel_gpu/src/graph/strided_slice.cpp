// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "json_object.h"
#include "data_inst.h"
#include <string>
#include <vector>

#include "strided_slice_shape_inference.hpp"

namespace cldnn {
primitive_type_id strided_slice::type_id() {
    static primitive_type_base<strided_slice> instance;
    return &instance;
}

layout strided_slice_inst::calc_output_layout(strided_slice_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<strided_slice>();
    auto input_layout = impl_param.input_layouts[0];
    auto output_format = format::get_default_format(desc->out_size.size());
    auto out_shape = desc->out_size.to_shape();
    std::vector<tensor::value_type> dims_converted(out_shape.begin(), out_shape.end());
    // extend shape to 4d
    for (size_t i = dims_converted.size(); i < 4; i++) {
        dims_converted.push_back(1);
    }
    auto out_size = cldnn::tensor(output_format, dims_converted);
    return layout{input_layout.data_type, output_format, out_size};
}

template<typename ShapeType>
std::vector<layout> strided_slice_inst::calc_output_layouts(strided_slice_node const& node, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<strided_slice>();
    auto input_layout = impl_param.input_layouts[0];
    auto output_format = format::get_default_format(desc->out_size.size());
    if (!node.const_mem.empty()) {
        ov::op::v1::StridedSlice op;
        std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
        std::vector<ov::PartialShape> input_shapes = {
            node.get_dependency(0).get_output_layout().get_partial_shape(),
            node.get_dependency(1).get_output_layout().get_partial_shape(),
            node.get_dependency(2).get_output_layout().get_partial_shape(),
            node.get_dependency(3).get_output_layout().get_partial_shape()
        };

        auto begin_mask = desc->begin_mask;
        auto end_mask = desc->end_mask;
        auto new_axis_mask = desc->new_axis_mask;
        auto shrink_axis_mask = desc->shrink_axis_mask;

        op.set_begin_mask(desc->begin_mask);
        op.set_end_mask(desc->end_mask);
        op.set_new_axis_mask(desc->new_axis_mask);
        op.set_shrink_axis_mask(desc->shrink_axis_mask);

        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(node.const_mem[0], node.get_program().get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock2(node.const_mem[1], node.get_program().get_stream());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> lock3(node.const_mem[2], node.get_program().get_stream());

        auto ptr1 = lock1.data();
        auto ptr2 = lock2.data();
        auto ptr3 = lock3.data();

        auto make_tensor = [](layout l, void* memory_pointer) {
            ov::element::Type et;

            switch (l.data_type) {
                case data_types::i64: et = ov::element::i64; break;
                case data_types::i32: et = ov::element::i32; break;
                default: IE_THROW() << "unsupported element type in strided slice primitive";
            }

            return std::make_shared<ngraph::runtime::HostTensor>(et, l.get_partial_shape().to_shape(), memory_pointer);
        };

        auto tensor1 = make_tensor(node.const_mem[0]->get_layout(), ptr1);
        auto tensor2 = make_tensor(node.const_mem[1]->get_layout(), ptr2);
        auto tensor3 = make_tensor(node.const_mem[2]->get_layout(), ptr3);

        std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
            {1, tensor1},
            {2, tensor2},
            {3, tensor3},
        };
        ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
        return {layout{output_shapes[0], input_layout.data_type, output_format}};
    }
    return layout{input_layout.data_type, output_format, output_shapes[0]};
}

std::vector<layout> strided_slice_inst::calc_output_layouts(strided_slice_node const& /*node*/, const kernel_impl_params& impl_param) {
    auto desc = impl_param.typed_desc<strided_slice>();
    auto input0_layout = impl_param.get_input_layout(0);

    auto& constant_mem = impl_param.memory_deps;

    if (constant_mem.empty()) {
        auto out_shape = ov::PartialShape::dynamic(input0_layout.get_rank());
        return { layout{out_shape, input0_layout.data_type, format::get_default_format(out_shape.rank().get_length())} };
    }

    ov::op::v1::StridedSlice op;
    std::vector<ShapeType> output_shapes = {ShapeType{}};
    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),
        impl_param.get_input_layout(1).get<ShapeType>(),
        impl_param.get_input_layout(2).get<ShapeType>(),
        impl_param.get_input_layout(3).get<ShapeType>()
    };

    op.set_begin_mask(desc->begin_mask);
    op.set_end_mask(desc->end_mask);
    op.set_new_axis_mask(desc->new_axis_mask);
    op.set_shrink_axis_mask(desc->shrink_axis_mask);
    op.set_ellipsis_mask_mask(desc->ellipsis_mask);

    auto mem1 = constant_mem.at(1);
    auto mem2 = constant_mem.at(2);
    auto mem3 = constant_mem.at(3);

    cldnn::mem_lock<uint8_t, mem_lock_type::read> lock1(mem1, impl_param.prog.get_stream());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> lock2(mem2, impl_param.prog.get_stream());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> lock3(mem3, impl_param.prog.get_stream());

    auto tensor1 = make_host_tensor(mem1->get_layout(), lock1.data());
    auto tensor2 = make_host_tensor(mem2->get_layout(), lock2.data());
    auto tensor3 = make_host_tensor(mem3->get_layout(), lock3.data());

    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> const_data = {
        {1, tensor1},
        {2, tensor2},
        {3, tensor3},
    };
    ov::op::v1::shape_infer(&op, input_shapes, output_shapes, const_data);
    auto output_format = format::get_default_format(output_shapes[0].size());

    return { layout{output_shapes[0], input0_layout.data_type, output_format} };
}

std::string strided_slice_inst::to_string(strided_slice_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite strided_slice_info;
    strided_slice_info.add("input id", input.id());
    strided_slice_info.add("begin_param id", node.get_dependency(1).id());
    strided_slice_info.add("end_param id", node.get_dependency(2).id());
    strided_slice_info.add("stride_param id", node.get_dependency(3).id());
    strided_slice_info.add("begin mask", node.get_primitive()->begin_mask);
    strided_slice_info.add("end mask", node.get_primitive()->end_mask);
    strided_slice_info.add("new axis mask", node.get_primitive()->new_axis_mask);
    strided_slice_info.add("shrink axis mask", node.get_primitive()->shrink_axis_mask);
    strided_slice_info.add("ellipsis mask", node.get_primitive()->ellipsis_mask);

    node_info->add("strided_slice info", strided_slice_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

strided_slice_inst::typed_primitive_inst(network& network, strided_slice_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
