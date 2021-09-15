// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "cldnn/runtime/layout.hpp"

#include <memory>
#include <string>

namespace cldnn {
struct network;
class engine;
struct program_node;
struct primitive_impl;
class primitive_inst;
struct program;
struct primitive;

struct primitive_type {
    virtual ~primitive_type() = default;

    virtual std::shared_ptr<program_node> create_node(program& program,
                                                      const std::shared_ptr<primitive> prim) const = 0;
    virtual std::shared_ptr<primitive_inst> create_instance(network& network,
                                                            const program_node& node) const = 0;
    virtual void allocate_memories(primitive_inst& inst) const = 0;
    virtual std::unique_ptr<primitive_impl> choose_impl(const program_node& node) const = 0;
    virtual bool does_an_implementation_exist(const program_node& node) const = 0;
    virtual bool does_possible_implementation_exist(const program_node& node) const = 0;
    virtual layout calc_output_layout(const program_node& node) const = 0;
    virtual std::string to_string(const program_node& node) const = 0;
};
}  // namespace cldnn
