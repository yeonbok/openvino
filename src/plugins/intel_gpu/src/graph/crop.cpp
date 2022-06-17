// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "crop_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>
#include "variadic_split_shape_inference.hpp"
#include "split_shape_inference.hpp"
#include <ngraph/pattern/op/label.hpp>

namespace cldnn {
primitive_type_id crop::type_id() {
    static primitive_type_base<crop> instance;
    return &instance;
}

layout crop_inst::calc_output_layout(crop_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for crop_node!");

    bool all_static = true;
    for (auto i : node.get_dependencies()) {
        if (i->get_output_layout().is_dynamic()) {
            all_static = false;
            break;
        }
    }

    const auto in_layout = node.input().get_output_layout();
    const auto desc = node.get_primitive();
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes = {
        node.input().get_output_layout().size,
    };
    for (int i = 1; i < node.get_dependencies().size(); ++i) {
        input_shapes.push_back(node.get_dependency(i).get_output_layout().size);
    }


    if (node.get_dependencies().size() == 3) {
        auto ov_op = ov::as_type_ptr<ov::op::v1::VariadicSplit>(node.get_ov_op()).get();
        #if 1
        //auto input_op = ov_op->get_input_node_shared_ptr(0).get();
        std::vector<int64_t> axis_values;
        std::vector<int64_t> split_lengths;
        get_data_as_int64<int64_t>(1, ov_op, axis_values, {});
        const auto axis_val = axis_values[0];
        std::cout << "before normalize : " << axis_val << std::endl;
        const int64_t axis = ov::normalize_axis(ov_op, axis_val, input_shapes[0].rank());
        if (axis == 3) {
            //input_op->set_output_type(0, input_op->get_output_element_type(0), ov::PartialShape{1, 32, 1, 2}); // just a trial
            // just a trial
            std::cout << "calc_output_layout) axis = " << axis << std::endl;
            std::cout << "                    input_shapes[1] = " << input_shapes[1].size() << std::endl;
            input_shapes[0][2] = 1;
            input_shapes[0][3] = 2;
        } else if (axis == 2) {
            //input_op->set_output_type(0, input_op->get_output_element_type(0), ov::PartialShape{1, 32, 2}); // just a trial
        }
        #endif
        shape_infer(ov_op, input_shapes, output_shapes);
    } else if (node.get_dependencies().size() == 2) {
        auto ov_op = ov::as_type_ptr<ov::op::v1::Split>(node.get_ov_op()).get();
        shape_infer(ov_op, input_shapes, output_shapes);
    } else if (node.get_dependencies().size() == 1) {
        const auto& ref_in_sizes = node.get_primitive()->reference_input;
        const auto& in_sizes = in_layout.get_tensor();
        const auto& offsets = node.get_primitive()->offsets;

        // Check for borders variant of crop.
        if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
            ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
            // Ignore not supported dimensions.
            const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
            const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});

            const auto out_sizes = in_sizes - (rb_sizes + lt_sizes);

            return layout({in_layout.data_type, in_layout.format, out_sizes});
        }
        return layout({in_layout.data_type, in_layout.format, ref_in_sizes});
    }
    return layout({in_layout.data_type, in_layout.format, output_shapes[desc->output_idx]});
}

std::string crop_inst::to_string(crop_node const& node) {
    const auto& desc = node.get_primitive();
    auto ref_in_sizes = desc->reference_input;
    const auto& offsets = desc->offsets;
    const auto in_layout = node.input().get_output_layout();
    const auto& in_sizes = in_layout.get_tensor();

    auto node_info = node.desc_to_json();

    // Check for borders variant of crop.
    if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
        ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
        // Ignore not supported dimensions.
        const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
        const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});

        ref_in_sizes = in_sizes - (rb_sizes + lt_sizes);
    }

    std::stringstream primitive_description;

    json_composite crop_info;
    crop_info.add("reference input size", ref_in_sizes.to_string());
    crop_info.add("offset", offsets.to_string());

    node_info->add("crop info", crop_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

crop_inst::typed_primitive_inst(network& network, crop_node const& node) : parent(network, node) {
#if 0 // TODO
    const auto& ref_in_sizes = argument.reference_input;
    const auto in_layout = node.input().get_output_layout();
    const auto& in_sizes = in_layout.get_tensor();
    const auto& offsets = argument.offsets;
    tensor null_tensor {};
    tensor value_tensor { 1, 1, 1, 1, 1 };
    // Check for borders variant of crop.
    if (ref_in_sizes.batch[0] < 0 || ref_in_sizes.feature[0] < 0 || ref_in_sizes.spatial[0] < 0 ||
        ref_in_sizes.spatial[1] < 0 || ref_in_sizes.spatial[2] < 0) {
        // Ignore not supported dimensions.
        const auto rb_sizes = ref_in_sizes.negate().sub({0, 0, 0, 0, 0});
        const auto lt_sizes = offsets.sub({0, 0, 0, 0, 0});

        const auto out_sizes = in_sizes - (rb_sizes + lt_sizes);

        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                           "Left/top/lower borders",
                                           lt_sizes,
                                           "0 value",
                                           null_tensor,
                                           "Invalid border size: negative");
        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                           "Right/bottom/upper borders",
                                           rb_sizes,
                                           "0 value",
                                           null_tensor,
                                           "Invalid border size: negative");

        CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                           "Input sizes - border sizes",
                                           out_sizes,
                                           "1 value",
                                           value_tensor,
                                           "Invalid border sizes: greater-equal input sizes");
    }

    if (ref_in_sizes != in_sizes) {
        std::cout << "error" << std::endl;
    }
    // check if output sizes matches reference input sizes
    CLDNN_ERROR_TENSOR_SIZES_GREATER_THAN(node.id(),
                                          "Reference input",
                                          ref_in_sizes,
                                          "input sizes",
                                          in_sizes,
                                          "Reference input tensor/ input tensor mismtach");

    // check if offsets do not extend input sizes and if match the output sizes
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                       "Batch offsets",
                                       offsets,
                                       "0 value",
                                       null_tensor,
                                       "Invalid Batch offset: negative value");
    auto input_size_sub_offsets = in_sizes - offsets;
    CLDNN_ERROR_TENSOR_SIZES_LESS_THAN(node.id(),
                                       "input sizes - offsets",
                                       input_size_sub_offsets,
                                       "reference input sizes",
                                       ref_in_sizes,
                                       "Invalid Batch offset: exceeds data for output!");
#endif
    if (node.can_be_optimized()) { // TODO : taylor
        build_deps();
        if (input_memory_ptr())
            reuse_input();
    }
}

void crop_inst::on_execute() {
    if (!node.can_be_optimized())
        return;

    if (_output && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    reuse_input();
}

void crop_inst::reuse_input() {
    _output = _network.get_engine().reinterpret_buffer(input_memory(), node.get_output_layout());
}
}  // namespace cldnn
