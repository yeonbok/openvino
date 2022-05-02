// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/non_zero.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<count_nonzero> : public typed_program_node_base<count_nonzero> {
    using parent = typed_program_node_base<count_nonzero>;
    typed_program_node(const std::shared_ptr<count_nonzero> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
public:
    using parent::parent;

    program_node& input() const {
        CLDNN_ERROR_LESS_THAN(id(), "the number of dependencies", dependencies.size(), "1", 1, "ERROR: the node has no input");
        return get_dependency(0);
    }
};

using count_nonzero_node = typed_program_node<count_nonzero>;

template <>
class typed_primitive_inst<count_nonzero> : public typed_primitive_inst_base<count_nonzero> {
    using parent = typed_primitive_inst_base<count_nonzero>;

public:
    static layout calc_output_layout(count_nonzero_node const& node);
    static std::string to_string(count_nonzero_node const& node);

    typed_primitive_inst(network& network, count_nonzero_node const& node);
};

using count_nonzero_inst = typed_primitive_inst<count_nonzero>;

template <>
struct typed_program_node<gather_nonzero> : public typed_program_node_base<gather_nonzero> {
    using parent = typed_program_node_base<gather_nonzero>;
    typed_program_node(const std::shared_ptr<gather_nonzero> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

    mutable bool _shape_ready = false;
public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        CLDNN_ERROR_LESS_THAN(id(), "the number of dependencies", dependencies.size(), "1", 1, "ERROR: the node has no input");
        return get_dependency(index);
    }

    void set_shape_ready() { _shape_ready = true; }
    void reset_shape_ready() const { _shape_ready = false; }
    bool get_shape_ready() const { return _shape_ready; }
};

using gather_nonzero_node = typed_program_node<gather_nonzero>;

template <>
class typed_primitive_inst<gather_nonzero> : public typed_primitive_inst_base<gather_nonzero> {
    using parent = typed_primitive_inst_base<gather_nonzero>;

public:
    static layout calc_output_layout(gather_nonzero_node const& node);
    static std::string to_string(gather_nonzero_node const& node);

    typed_primitive_inst(network& network, gather_nonzero_node const& node);

private:
    void update_shape() override;
};

using gather_nonzero_inst = typed_primitive_inst<gather_nonzero>;
}  // namespace cldnn
