// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_inst.h"

#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include <string>
#include <vector>
#include <set>

namespace cldnn {
primitive_type_id broadcast::type_id() {
    static primitive_type_base<broadcast> instance;
    return &instance;
}

layout broadcast_inst::calc_output_layout(broadcast_node const& node) {
    assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
           "Output data type forcing is not supported for broadcast_node!");
    auto desc = node.get_primitive();
    if (desc->broadcast_sizes_partial.get_shape().size() != 0) {
        // static
        auto input_layout = node.input().get_output_layout();
        return {input_layout.data_type, input_layout.format, desc->broadcast_sizes_partial};
    } else if (desc->broadcast_sizes.sizes()[0] != 0) {
        // static
        auto input_layout = node.input().get_output_layout();
        return {input_layout.data_type, input_layout.format, desc->broadcast_sizes};
    } else {
        // dynamic
        return {node.input().get_output_layout().data_type, node.input().get_output_layout().format, ov::PartialShape()};
    }
}

static std::vector<int64_t> read_vector(cldnn::memory::ptr mem, cldnn::stream& stream) {
    switch (mem->get_layout().data_type) {
        case data_types::i32: {
            mem_lock<int32_t, mem_lock_type::read> lock{mem, stream};
            return std::vector<int64_t>(lock.begin(), lock.end());
        }
        case data_types::i64: {
            mem_lock<int64_t, mem_lock_type::read> lock{mem, stream};
            return std::vector<int64_t>(lock.begin(), lock.end());
        }
        default: IE_THROW() << "read_vector: unsupported data type";
    }
}

void broadcast_inst::update_shape() {
    if (!_network.shape_changed())
        return;

    auto& node = const_cast<broadcast_node&>(dynamic_cast<const broadcast_node&>(_node));

    auto shape_mem = _network.get_output_memory(_node.get_dependency(1).id());
    auto output_shape = ov::PartialShape(read_vector(shape_mem, _network.get_stream()));
    auto new_layout = layout{cldnn::data_types::i32, cldnn::format::bfyx, output_shape};
    auto out_layout = _node.is_valid_output_layout() ? _node.get_output_layout() : layout(data_types::i32, format::bfyx, tensor{});
    auto out_layout_str = _node.is_valid_output_layout() ? out_layout.to_string() : "invalid";
    if (!_node.is_valid_output_layout() || _node.get_output_layout() != new_layout)
        set_shape_change();

    node.set_output_layout(new_layout);
}

std::string broadcast_inst::to_string(broadcast_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    const auto& broadcast_sizes = desc->broadcast_sizes;
    const auto& broadcast_axes = desc->broadcast_axes;
    auto& input = node.input();

    std::stringstream primitive_description;
    std::stringstream ss_broadcast_axes;

    for (size_t i = 0; i < broadcast_axes.size(); ++i) {
        ss_broadcast_axes << broadcast_axes.at(i);
        i != (broadcast_axes.size() - 1) ? ss_broadcast_axes << ", " : ss_broadcast_axes << "";
    }

    json_composite broadcast_info;
    broadcast_info.add("input id", input.id());
    broadcast_info.add("broadcast_sizes", broadcast_sizes.to_string());
    broadcast_info.add("broadcast axes", ss_broadcast_axes.str());

    node_info->add("broadcast info", broadcast_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

broadcast_inst::typed_primitive_inst(network& network, broadcast_node const& node) : parent(network, node) {
    auto sizes = node.get_primitive()->broadcast_sizes.sizes();
    if (sizes[0] == 0 && sizes[1] == 0 && sizes[2] == 0 && sizes[3] == 0 && sizes[4] == 0 && sizes[5] == 0)
        return;

    auto input_layout = node.input().get_output_layout();

    const auto& output_sizes = argument.broadcast_sizes;
    const auto format = input_layout.format;

    std::vector<tensor::value_type> input_dims = input_layout.get_dims();
    size_t max_axes_num = input_layout.get_rank();

    std::vector<tensor::value_type> reordered_input_dims(max_axes_num, 0);
    std::set<uint16_t> existing;

    const auto& broadcast_axes = node.get_primitive()->broadcast_axes;
    size_t broadcast_axes_size = broadcast_axes.size();
    size_t index = 0;
    size_t input_index = broadcast_axes_size;

    if (format == format::bfzyx) {
        if (broadcast_axes_size > 5) {
            CLDNN_ERROR_MESSAGE(node.id(),
                                "Incorrect parameters configuration: broadcast_axes size should be less or equal 5.");
        }
    } else if (broadcast_axes_size > 4) {
        CLDNN_ERROR_MESSAGE(node.id(),
                            "Incorrect parameters configuration: broadcast_axes size should be less or equal 4.");
    }
    for (size_t i = 0; i < broadcast_axes_size; ++i) {
        if (broadcast_axes.at(i) >= max_axes_num) {
            CLDNN_ERROR_MESSAGE(
                node.id(),
                "Incorrect parameters configuration: broadcast_axes index should be within broadcast_sizes range.");
        }
        if (existing.find(broadcast_axes.at(i)) != existing.end()) {
            CLDNN_ERROR_MESSAGE(
                node.id(),
                "Incorrect parameters configuration: Duplicate axes numbers was found in broadcast_axes.");
        }
        existing.insert(broadcast_axes.at(i));
    }
    for (size_t i = 0; i < input_index; ++i) {
        CLDNN_ERROR_NOT_EQUAL(node.id(),
                              "Input size on dimension number " + std::to_string(i),
                              input_dims.at(i),
                              "",
                              1,
                              "Must be equal 1.");
    }
    // bfyx, bfzyx format
    for (size_t i = 0; i < max_axes_num; ++i) {
        if (std::find(broadcast_axes.begin(), broadcast_axes.end(), i) != broadcast_axes.end()) {
            reordered_input_dims.at(i) = input_dims.at(index);
            ++index;
        } else {
            reordered_input_dims.at(i) = input_dims.at(input_index);
            ++input_index;
        }
    }
    tensor input_sizes_to_compare = tensor(format::get_default_format(reordered_input_dims.size()), reordered_input_dims);

    CLDNN_ERROR_TENSOR_SIZES_NOT_DIVIDABLE(node.id(),
                                           "Broadcast sizes",
                                           output_sizes,
                                           "input sizes",
                                           input_sizes_to_compare,
                                           "Invalid broadcast size: not dividable by input size");
}
}  // namespace cldnn
