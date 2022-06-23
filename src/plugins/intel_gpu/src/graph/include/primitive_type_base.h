// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "intel_gpu/runtime/engine.hpp"

#include "meta_utils.h"
#include "primitive_type.h"
#include "program_node.h"
#include "primitive_inst.h"
#include "arg_max_min_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "impls/implementation_map.hpp"

#include <memory>
#include <string>

namespace cldnn {
template <class PType>
struct primitive_type_base : primitive_type {
    std::shared_ptr<cldnn::program_node> create_node(program& program,
                                                     const std::shared_ptr<primitive> prim) const override {
        if (prim->type != this)
            throw std::invalid_argument("primitive_type_base::create_node: primitive type mismatch");

        return std::make_shared<typed_program_node<PType>>(std::static_pointer_cast<PType>(prim), program);
    }

    std::shared_ptr<cldnn::primitive_inst> create_instance(network& network,
                                                           const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::create_instance: primitive type mismatch");

        return std::make_shared<typed_primitive_inst<PType>>(network, node);
    }

    // TODO: Should we get rid of engine type in impl map? Or we must pass internal build engine to get real ocl type?
    std::unique_ptr<primitive_impl> choose_impl(const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::choose_impl: primitive type mismatch");

        auto factory = implementation_map<PType>::get(node);
        return std::move(std::unique_ptr<primitive_impl>(factory(node)));
    }

    bool does_an_implementation_exist(const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::does_an_implementation_exist: primitive type mismatch");

        return implementation_map<PType>::check(node);
    }

    bool does_possible_implementation_exist(const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::does_possible_implementation_exist: primitive type mismatch");
        return implementation_map<PType>::check_io_eq(node);
    }

    cldnn::layout calc_output_layout(const cldnn::program_node& node, int32_t idx = 0) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::calc_output_layout: primitive type mismatch");

        return typed_primitive_inst<PType>::calc_output_layout(node, idx);
    }

    std::vector<cldnn::layout> calc_output_layouts(const cldnn::program_node& node) const override {
        std::cout << "[" << __FILE__ << "] calc_output_layouts called for " << node.id() << std::endl;
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::calc_output_layout: primitive type mismatch");
        std::vector<cldnn::layout> layouts;
        for (size_t i = 0; i < node.get_outputs_count(); ++i) {
            layouts.push_back(calc_output_layout(node, i));
        }
        //return { typed_primitive_inst<PType>::calc_output_layout(node, 0) };
        return layouts;
    }

#if 0 // TODO(taylor)
    std::string to_string(const cldnn::program_node& node) const override {
        if (node.type() != this)
            throw std::invalid_argument("primitive_type_base::to_string: primitive type mismatch");

        return typed_primitive_inst<PType>::to_string(node);
    }
#endif
};

}  // namespace cldnn
