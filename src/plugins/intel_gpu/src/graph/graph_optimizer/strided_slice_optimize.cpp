// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "intel_gpu/runtime/error_handler.hpp"
#include "pass_manager.h"
#include "program_helpers.h"
#include "strided_slice_inst.h"
#include "reshape_inst.h"
#include "data_inst.h"
#include <vector>
#include <memory>
#include "intel_gpu/plugin/common_utils.hpp"

using namespace cldnn;

void strided_slice_optimize::run(program& p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = (*node_itr++);
        if (node->is_type<strided_slice>() && node->get_output_layout().is_static()) {
            auto& strided_slice_node = node->as<strided_slice>();
            auto& new_axis_mask = strided_slice_node.get_primitive()->new_axis_mask;

            if (std::find(new_axis_mask.begin(), new_axis_mask.end(), 1) == new_axis_mask.end())
                continue;

            auto& deps = node->get_dependencies();
            for (size_t i = deps.size(); i--;)
                if (deps[i]->is_type<data>())
                    node->remove_dependency(i);

            auto node_layout = strided_slice_node.get_output_layout();

            auto is_shift_possible = [&](const ov::PartialShape& dims) -> bool {
                if (dims.rank().get_length() == 0)
                    CLDNN_ERROR_MESSAGE(node->id(), "Error while adding new axis: node has incorrect dimensions");

                if (dims[dims.size() - 1] == 1)
                    return true;
                else
                    CLDNN_ERROR_MESSAGE(node->id(), "Not supported yet: too many axes for adding");
                return false;
            };

            auto output_dims_sizes = node_layout.get_shape();
            if (std::find(new_axis_mask.begin(), new_axis_mask.end(), 1) != new_axis_mask.end()) {
                for (size_t i = 0; i < new_axis_mask.size(); ++i) {
                    if (new_axis_mask[new_axis_mask.size() - i - 1] == 1) {
                        if (is_shift_possible(output_dims_sizes)) {
                            for (size_t j = output_dims_sizes.size() - 1; j > i; --j)
                                output_dims_sizes[j] = output_dims_sizes[j - 1];
                            output_dims_sizes[i] = 1;
                        }
                    }
                }
            }

            auto pattern = ov::PartialShape(output_dims_sizes);

            auto reshape_prim = std::make_shared<reshape>("reshape_" + node->id(), node->get_dependency(0).get_primitive()->id,
                                                          pattern);

            auto& reshape_prim_node = p.get_or_create(reshape_prim);

            layout output_layout = { reshape_prim->output_shape, node_layout.data_type, node_layout.format };
            reshape_prim_node.set_output_layout(output_layout);

            p.add_intermediate(reshape_prim_node, *node, 0, true);
            p.extract_and_remove(*node);
            reshape_prim_node.calc_output_layout();
        }
    }
}
