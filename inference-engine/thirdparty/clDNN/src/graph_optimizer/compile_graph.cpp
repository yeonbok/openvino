// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "internal_primitive.h"
#include "data_inst.h"
#include "mutable_data_inst.h"
#include "program_node.h"
#include "cldnn_itt.h"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace cldnn;

void compile_graph::run(program_impl& p) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNN, "CLDNN::pass::CompileGraph");
    size_t order_idx = 0;
    for (auto& node : p.get_processing_order()) {
        node->set_unique_id(std::to_string(order_idx++));
        if (!node->is_type<data>()) {
            node->get_output_layout();
        }
    }

    for (auto& node : p.get_processing_order()) {
        if (!node->is_type<data>() && !(node->is_type<mutable_data>() && node->get_dependencies().empty())) {
            node->selected_impl = node->type()->choose_impl(p.get_engine(), *node);
        }
    }
}
