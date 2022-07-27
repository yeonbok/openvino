// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_node.h"
#include "program_helpers.h"
#include "primitive_inst.h"

#ifdef ENABLE_ONEDNN_FOR_GPU
#if 0 // TODO(taylor)
#include "convolution_inst.h"
#include "quantize_inst.h"
#include "reorder_inst.h"
#include <impls/onednn/utils.hpp>
#endif
#endif // ENABLE_ONEDNN_FOR_GPU

#include "to_string_utils.h"
#include "json_object.h"
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <set>

using namespace cldnn;

thread_local size_t program_node::cur_id = 0;

program_node::program_node(std::shared_ptr<primitive> prim, program& prog)
    : desc(prim), myprog(prog), org_id(prim ? (prim->id) : 0) {
    if (prim) {
        num_outputs = prim->num_outputs;
        for (int i = 0 ; i < num_outputs; ++i) {
            layout output_layout = layout(data_types::f32, format::bfyx, tensor());
            // TODO(kelvin): Need to update for multiple output.
            // Apply each output padding by output index. ex) output_layout.data_padding = prim->output_paddings[i];
            output_layout.data_padding = prim->output_paddings[0];
            output_layouts.push_back(output_layout);
            valid_output_layouts.push_back(false);
        }
    }
}

void program_node::replace_dependency(size_t idx, program_node& new_dep) {
    if (idx >= dependencies.size())
        return;
    if (dependencies[idx].first == &new_dep)
        return;

    auto it = std::find(dependencies[idx].first->users.begin(), dependencies[idx].first->users.end(), this);
    if (it != dependencies[idx].first->users.end()) {
        dependencies[idx].first->users.erase(it);
    }
    myprog.remove_if_dangling(*dependencies[idx].first);

    dependencies[idx].first = &new_dep;
    new_dep.users.push_back(this);
}

void program_node::replace_dependency(program_node const& old_dep, program_node& new_dep) {
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i].first == &old_dep)
            return replace_dependency(i, new_dep);
}

std::vector<primitive_id> program_node::get_dependencies_ids() const {
    std::vector<primitive_id> dep_ids;
    for (auto& dependency : dependencies) dep_ids.push_back(dependency.first->get_primitive()->id);
    return dep_ids;
}

void program_node::remove_dependency(size_t idx) {
    if (idx >= dependencies.size())
        return;

    dependencies[idx].first->users.remove(this);
    myprog.remove_if_dangling(*dependencies[idx].first);
    dependencies.erase(dependencies.begin() + idx);
}

std::set<input_info, input_info::cmp> program_node::get_memory_dependencies() const { return memory_dependencies; }

void program_node::add_memory_dependency(input_info prim) { memory_dependencies.insert(prim); }

void program_node::add_memory_dependency(std::vector<input_info> prim_list) {
    memory_dependencies.insert(prim_list.begin(), prim_list.end());
}
#if 0 // TODO(taylor)
std::unique_ptr<json_composite> program_node::desc_to_json() const {
    std::unique_ptr<json_composite> node_info = std::unique_ptr<json_composite>(new json_composite());
    node_info->add("ptr", "node_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
    node_info->add("id", id());
    node_info->add("type", desc->type_string());
    node_info->add("valid output layout", bool_to_str(valid_output_layouts));
    std::stringstream s;
    s << get_preferred_impl_type();
    node_info->add("preferred impl", s.str());

    json_composite output_layout_info;
    output_layout_info.add("data type", dt_to_str(output_layout.data_type));
    output_layout_info.add("format", fmt_to_str(output_layout.format));
    output_layout_info.add("size", output_layout.size.to_string());

    json_composite padding_info;
    padding_info.add("lower size", output_layout.data_padding.lower_size().to_string());
    padding_info.add("upper size", output_layout.data_padding.upper_size().to_string());
    output_layout_info.add("padding info", padding_info);

    node_info->add("output layout", output_layout_info);

    node_info->add("in data flow", bool_to_str(data_flow));
    node_info->add("constant", bool_to_str(constant));
    node_info->add("in data flow", bool_to_str(data_flow));
    node_info->add("output", bool_to_str(output));

    json_composite fused_nodes_info;
    size_t index = 0;
    for (auto& fused_desc : get_fused_primitives()) {
        json_composite fused_node_info;
        fused_node_info.add("id", fused_desc.node->id());
        std::vector<primitive_id> dep_ids;
        for (auto dep : fused_desc.deps) {
            dep_ids.push_back(dep.first);
        }
        fused_node_info.add("dependencies", dep_ids);
        fused_node_info.add("dep start_idx", fused_desc.dep_start_idx);
        json_composite info;
        info.add("data type", dt_to_str(fused_desc.output_layout.data_type));
        info.add("format", fmt_to_str(output_layout.format));
        info.add("size", output_layout.size.to_string());
        fused_node_info.add("output layout", info);
        fused_nodes_info.add("fused primitive idx " + std::to_string(index++), fused_node_info);
    }
    node_info->add("fused primitives", fused_nodes_info);

    json_composite fused_activations;
    auto fused_activations_funcs = get_fused_activations_funcs();
    if (!fused_activations_funcs.empty()) {
        for (size_t i = 0; i < fused_activations_funcs.size(); i++) {
            json_composite fused_activation_info;
            auto activation_type = activation_type_to_str(fused_activations_funcs[i]);
            auto params = get_fused_activations_params()[i];
            fused_activation_info.add("params", "a=" + std::to_string(params.a) + ", b=" + std::to_string(params.b));
            fused_activation_info.add("activation", activation_type);
            fused_activations.add("fused activation idx " + std::to_string(i), fused_activation_info);
        }
        node_info->add("fused activations (legacy)", fused_activations);
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    auto& onednn_post_ops = get_fused_primitives_onednn();
    if (onednn_post_ops.size()) {
        size_t post_op_index = 0;
        json_composite post_ops_info;
        for (auto& fused_prim_desc : onednn_post_ops) {
            json_composite post_op_info;
            post_op_info.add("post op", onednn_post_op_type_to_str(fused_prim_desc.op_type));
            post_op_info.add("memory dependency", fused_prim_desc.mem_dep);
            post_op_info.add("memory offset", fused_prim_desc.mem_offset);
            post_ops_info.add("post ops idx " + std::to_string(post_op_index++), post_op_info);
        }
        node_info->add("onednn post ops", post_ops_info);
    }
#endif

    std::vector<std::string> deps_ptrs;
    {
        bool empty = true;
        auto itr = dependencies.begin();
        while (itr != dependencies.end()) {
            if (empty) {
                empty = false;
            }
            deps_ptrs.push_back(std::to_string(reinterpret_cast<uintptr_t>(*itr++)));
        }
        if (deps_ptrs.empty()) {
            deps_ptrs.push_back("null");
        }
    }
    node_info->add("dependencies", deps_ptrs);

    std::vector<std::string> users_ptrs;
    {
        bool empty = true;
        auto itr = users.begin();
        while (itr != users.end()) {
            if (empty) {
                empty = false;
            }
            users_ptrs.push_back(std::to_string(reinterpret_cast<uintptr_t>(*itr++)));
        }
        if (users_ptrs.empty()) {
            users_ptrs.push_back("null");
        }
    }
    node_info->add("users", users_ptrs);
    std::vector<std::string> impls;
    if (!selected_impl) {
        impls.push_back("null");
    } else {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpotentially-evaluated-expression"
#endif
        impls.push_back(selected_impl->get_kernel_name());
#ifdef __clang__
#pragma clang diagnostic pop
#endif
    }
    node_info->add("implementation", impls);
    return node_info;
}
void program_node::remove_dependency(program_node& node) {
    for (size_t i = 0; i < dependencies.size(); ++i)
        if (dependencies[i] == &node)
            remove_dependency(i);
}
bool program_node::is_detached(bool whole_branch) {
    if (!users.empty())
        return false;
    if (!whole_branch && !dependencies.empty())
        return false;
    return true;
}

#endif
layout program_node::calc_output_layout(int32_t idx) const {
    return type()->calc_output_layout(*this, idx);
}

std::vector<layout> program_node::calc_output_layouts() const {
    std::vector<layout> layouts;
    for (int32_t i = 0; i < get_primitive()->num_outputs; ++i) {
        layouts.push_back(calc_output_layout(i));
    }
    return layouts;
}

layout program_node::get_output_layout(int32_t idx) const {
    if (!valid_output_layouts[idx])
        throw std::runtime_error(std::to_string(idx) + "th Output layout not calculated for " + id() + " node");

    return output_layouts[idx];
}

layout program_node::get_output_layout(bool invalidate_users_if_changed, int32_t idx) {
    if (valid_output_layouts[idx])
        return output_layouts[idx];

    auto new_layout = calc_output_layout(idx);
    set_output_layout(new_layout, invalidate_users_if_changed, idx);
    return output_layouts[idx];
}

std::vector<layout> program_node::get_output_layouts(bool invalidate_users_if_changed) {
    if (is_all_valid_output_layout())
        return output_layouts;
    auto new_layouts = calc_output_layouts();
    set_output_layouts(new_layouts, invalidate_users_if_changed);
    return output_layouts;
}

std::vector<layout> program_node::get_output_layouts() const {
    if (!is_all_valid_output_layout()) {
        throw std::runtime_error("Output layouts not calculated");
    }

    return output_layouts;
}

layout program_node::get_non_padded_output_layout(bool invalidate_users_if_changed, int32_t idx) {
    auto out_layout = get_output_layout(invalidate_users_if_changed, idx);
    auto result = layout({out_layout.data_type, out_layout.format, out_layout.size});
    return result;
}

bool program_node::set_output_layouts(std::vector<layout>& new_layouts, bool invalidate_users_if_changed) {
    bool changed = false;
    for (auto i = 0; i < new_layouts.size(); ++i) {
        auto new_layout = new_layouts[i];
        changed |= set_output_layout(new_layout, invalidate_users_if_changed, i);
    }
    for (auto v : valid_output_layouts) {
        v = true;
    }
    return changed;
}

bool program_node::set_output_layout(layout& new_layout, bool invalidate_users_if_changed, int32_t idx) {
    merge_output_padding(new_layout.data_padding);
    new_layout.data_padding = output_layouts[idx].data_padding;
    bool changed = (new_layout != output_layouts[idx]);
    if (changed && invalidate_users_if_changed)  // output_layout has changed! invalidate users
        invalidate_users();

    output_layouts[idx] = new_layout;
    valid_output_layouts[idx] = true;
    return changed;
}

bool program_node::recalc_output_layouts(bool invalidate_users_if_changed) {
    auto new_layouts = calc_output_layouts();
    return set_output_layouts(new_layouts, invalidate_users_if_changed);
}

bool program_node::has_padded_dependency() {
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const std::pair<program_node*, int>& dep) {
        return dep.first->is_padded();
    });
}

bool program_node::has_padded_dependency() const {
    return std::any_of(get_dependencies().begin(), get_dependencies().end(), [](const std::pair<program_node*, int>& dep) {
        return dep.first->is_padded();
    });
}

void program_node::invalidate_users() const {
    for (auto& user : users) {
        for (size_t i = 0; i < user->valid_output_layouts.size(); ++i) {
            if (user->valid_output_layouts[i]) {
                user->valid_output_layouts[i] = false;
                user->invalidate_users();
            }
        }
    }
}

void program_node::support_padding_all(bool support) {
    std::fill(_support_padding_in_axis.begin(), _support_padding_in_axis.end(), support);
}

bool program_node::is_padding_supported(int axis, int padding) const {
    if (!support_padding(axis))
        return false;

    // TODO(andrew) - Need to check all format of output layouts?
    auto fmt = output_layouts.at(0).format;

    // WA for known cases of padding not supported in implementations
    if (fmt == format::b_fs_yx_fsv16) {
        if (axis == 0 || (axis == 1 && padding % 16 != 0))
            return false;
    }

    if (fmt == format::fs_b_yx_fsv32 && (axis == 0))
        return false;

    for (const auto& block : fmt.block_sizes()) {
        size_t block_axis = block.first;
        int block_size = block.second;

        if (axis != static_cast<int>(block_axis))
            continue;

        if (padding % block_size != 0)
            return false;
    }

    return true;
}

 void program_node::set_selected_impl(std::unique_ptr<primitive_impl> impl) {
    selected_impl = std::move(impl);
}

bool program_node::need_lockable_memory() const {
    bool need_lockable_mem = get_users().empty() || std::any_of(get_users().begin(), get_users().end(), [](const program_node* n) {
        return n->get_selected_impl()->is_cpu();
    });

    return need_lockable_mem;
}

    /* ----------------------------------------- */
    /* Onednn fused operations integration logic */
    /* ----------------------------------------- */

#ifdef ENABLE_ONEDNN_FOR_GPU

bool program_node::has_out_scales(const std::shared_ptr<dnnl::primitive_attr>& attr) {
    int mask;
    std::vector<float> scales;
    attr->get_output_scales(mask, scales);
    const auto drfv = reinterpret_cast<const int32_t&>(DNNL_RUNTIME_F32_VAL);
    return !scales.empty() && (reinterpret_cast<const int32_t&>(scales[0]) == drfv);
}

dnnl::post_ops program_node::try_optimize_post_ops(dnnl::post_ops& p_ops, const std::shared_ptr<dnnl::primitive_attr>& attr,
                                                   bool& optimization_is_completed) {
    // Create new dnnl::post_ops object which will be filled inside the optimization process
    dnnl::post_ops optimized_p_ops;

    // Add new post-op into optimized_p_ops structure
    auto add_post_op = [&](onednn_post_op_type type, const dnnl::post_ops& cur_p_ops, dnnl::post_ops& new_p_ops, int idx) {
        switch (type) {
            case onednn_post_op_type::eltwise_act:
            case onednn_post_op_type::eltwise_clip:
            case onednn_post_op_type::eltwise_linear:
            case onednn_post_op_type::eltwise_round:
            {
                dnnl::algorithm alg;
                float scale, alpha, beta;
                cur_p_ops.get_params_eltwise(idx, scale, alg, alpha, beta);
                new_p_ops.append_eltwise(scale, alg, alpha, beta);
                break;
            }

            case onednn_post_op_type::binary_add:
            case onednn_post_op_type::binary_mul:
            case onednn_post_op_type::binary_max:
            case onednn_post_op_type::binary_min:
            {
                dnnl::algorithm alg;
                dnnl::memory::desc desc;
                cur_p_ops.get_params_binary(idx, alg, desc);
                new_p_ops.append_binary(alg, desc);
                break;
            }

            case onednn_post_op_type::binary_relu:
            {
                int mask;
                cur_p_ops.get_params_prelu(idx, mask);
                new_p_ops.append_prelu(mask);
                break;
            }

            case onednn_post_op_type::scale:
            {
                break;
            }

            case onednn_post_op_type::sum:
            {
                float scale;
                dnnl::memory::data_type data_type;
                cur_p_ops.get_params_sum(idx, scale, data_type);
                if (is_type<convolution>()) {
                    new_p_ops.append_sum(scale, data_type);
                } else {
                    new_p_ops.append_sum(scale);
                }
                break;
            }

            case onednn_post_op_type::optimized:
            case onednn_post_op_type::optimized_sum:
            case onednn_post_op_type::optimized_eltwise_act:
            case onednn_post_op_type::optimized_eltwise_linear:
            case onednn_post_op_type::optimized_eltwise_clip:
            case onednn_post_op_type::optimized_eltwise_round:
            {
                // Current operation already has been optimized => don't need extra actions
                break;
            }

            default:
                throw std::runtime_error("Unsupported onednn post-operation type");
        }
    };

    // Check that post-op type is any optimized
    auto type_is_any_optimized = [](onednn_post_op_type type) -> bool {
        return type == onednn_post_op_type::optimized || type == onednn_post_op_type::optimized_sum ||
               type == onednn_post_op_type::optimized_eltwise_act ||
               type == onednn_post_op_type::optimized_eltwise_linear ||
               type == onednn_post_op_type::optimized_eltwise_clip ||
               type == onednn_post_op_type::optimized_eltwise_round;
    };

    // Check that post-op type is eltwise
    auto type_is_eltwise = [](onednn_post_op_type type) -> bool {
        return type == onednn_post_op_type::eltwise_round || type == onednn_post_op_type::eltwise_linear ||
               type == onednn_post_op_type::eltwise_clip  || type == onednn_post_op_type::eltwise_act;
    };

    // Check that post-op type is binary_add or binary_mul
    auto type_is_binary_add_or_mul = [](onednn_post_op_type type) -> bool {
        return type == onednn_post_op_type::binary_add || type == onednn_post_op_type::binary_mul;
    };

    // Simple post-op type checks
    auto type_is_optimized         = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::optimized; };
    auto type_is_eltwise_linear    = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::eltwise_linear; };
    auto type_is_optimized_eltwise = [](onednn_post_op_type type) -> bool {
         return type == onednn_post_op_type::optimized_eltwise_act || type == onednn_post_op_type::optimized_eltwise_linear ||
                type == onednn_post_op_type::optimized_eltwise_round || type == onednn_post_op_type::optimized_eltwise_clip;
    };
    auto type_is_binary_add        = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::binary_add; };
    auto type_is_binary_mul        = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::binary_mul; };
    auto type_is_sum               = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::sum; };
    auto type_is_optimized_sum     = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::optimized_sum; };
    auto type_is_scale             = [](onednn_post_op_type type) -> bool { return type == onednn_post_op_type::scale; };

    auto get_eltwise_type = [](onednn_post_op_type type) {
        switch (type) {
            case onednn_post_op_type::optimized_eltwise_act: return onednn_post_op_type::eltwise_act;
            case onednn_post_op_type::optimized_eltwise_clip: return onednn_post_op_type::eltwise_clip;
            case onednn_post_op_type::optimized_eltwise_linear: return onednn_post_op_type::eltwise_linear;
            case onednn_post_op_type::optimized_eltwise_round: return onednn_post_op_type::eltwise_round;
            default:
                throw std::runtime_error("Unsupported optimized eltwise post-operation type");
                break;
        }
    };

    auto& cur_post_ops = get_fused_primitives_onednn();

    size_t cur_post_op_idx = 1;
    size_t prev_post_op_idx = 0;
    bool optimization_done = false;

    // Check and update post-op map if we already optimized something
    for (size_t post_op_idx = 0; post_op_idx < cur_post_ops.size(); post_op_idx++) {
        if (type_is_optimized_sum(cur_post_ops[post_op_idx].op_type))
            cur_post_ops[post_op_idx].op_type = onednn_post_op_type::sum;
        else if (type_is_optimized_eltwise(cur_post_ops[post_op_idx].op_type))
            cur_post_ops[post_op_idx].op_type = get_eltwise_type(cur_post_ops[post_op_idx].op_type);
        else if (type_is_optimized(cur_post_ops[post_op_idx].op_type))
            cur_post_ops.erase(cur_post_ops.begin() + post_op_idx);
    }

    // Get post-ops size for current node
    auto post_ops_size = cur_post_ops.size();

    auto get_optimized_eltwise_type = [](onednn_post_op_type type) {
        switch (type) {
            case onednn_post_op_type::eltwise_linear: return onednn_post_op_type::optimized_eltwise_linear;
            case onednn_post_op_type::eltwise_act: return onednn_post_op_type::optimized_eltwise_act;
            case onednn_post_op_type::eltwise_round: return onednn_post_op_type::optimized_eltwise_round;
            case onednn_post_op_type::eltwise_clip: return onednn_post_op_type::optimized_eltwise_clip;
            default:
                throw std::runtime_error("Unsupported optimized eltwise post-operation type");
                break;
        }
    };

    // Try to combine pairs of arithmetic post-ops (adds and muls) into one operation inside this cycle
    while (!optimization_done) {
        auto cur_type = cur_post_ops[cur_post_op_idx].op_type;
        auto prev_type = cur_post_ops[prev_post_op_idx].op_type;

        // Ignore optimized operations for "previous" operation in our operation pair
        while (type_is_any_optimized(prev_type) && prev_post_op_idx < post_ops_size - 1) {
            prev_post_op_idx++;
            if (prev_post_op_idx == cur_post_op_idx && cur_post_op_idx < post_ops_size - 1)
                cur_post_op_idx++;
            prev_type = cur_post_ops[prev_post_op_idx].op_type;
            cur_type = cur_post_ops[cur_post_op_idx].op_type;
        }

        // Ignore optimized operations for "current" operation in our operation pair
        while (type_is_any_optimized(cur_type) && cur_post_op_idx < post_ops_size - 1) {
            cur_post_op_idx++;
            cur_type = cur_post_ops[cur_post_op_idx].op_type;
        }

        auto cur_idx = static_cast<int>(has_out_scales(attr) ? (cur_post_op_idx >= 1 ? cur_post_op_idx - 1 : 0) : cur_post_op_idx);
        auto prev_idx = static_cast<int>(has_out_scales(attr) ? (prev_post_op_idx >= 1 ? prev_post_op_idx - 1 : 0) : prev_post_op_idx);

        // If this is the last pair and it's optimized - add the last post-op and go out from the cycle
        if (cur_post_op_idx == post_ops_size - 1 && (type_is_any_optimized(cur_type) || type_is_any_optimized(prev_type))) {
            if (!type_is_any_optimized(prev_type)) {
                add_post_op(prev_type, p_ops, optimized_p_ops, prev_idx);
            }
            if (!type_is_any_optimized(cur_type)) {
                add_post_op(cur_type, p_ops, optimized_p_ops, cur_idx);
            }
            break;
        }

        // Post-ops combinations which can be simplified
        auto eltw_and_eltw  = type_is_eltwise(cur_type) && type_is_eltwise(prev_type);
        auto bin_and_eltw   = type_is_binary_add_or_mul(cur_type) && type_is_eltwise_linear(prev_type);
        auto eltw_and_bin   = type_is_eltwise_linear(cur_type) && type_is_binary_add_or_mul(prev_type);
        auto sum_and_eltw   = type_is_sum(cur_type) && type_is_eltwise(prev_type);
        auto eltw_and_scale = type_is_eltwise_linear(cur_type) && type_is_scale(prev_type);

        auto can_try_optimize = eltw_and_eltw ||
                                bin_and_eltw ||
                                eltw_and_bin ||
                                sum_and_eltw ||
                                eltw_and_scale;

        bool cur_ops_pair_is_optimized = false;

        if (can_try_optimize) {
            if (eltw_and_eltw) {
                dnnl::algorithm cur_alg, prev_alg;
                float cur_scale, prev_scale, cur_alpha, prev_alpha, cur_beta, prev_beta;

                p_ops.get_params_eltwise(prev_idx, prev_scale, prev_alg, prev_alpha, prev_beta);
                p_ops.get_params_eltwise(cur_idx, cur_scale, cur_alg, cur_alpha, cur_beta);

                auto eltw_linear_and_eltw_linear = type_is_eltwise_linear(cur_type) && type_is_eltwise_linear(prev_type);
                auto eltw_linear_and_eltw_non_linear = type_is_eltwise_linear(cur_type) && !type_is_eltwise_linear(prev_type) && cur_beta == 0;

                // eltwise_linear + eltwise_linear combination can be optimized always
                if (eltw_linear_and_eltw_linear) {
                    dnnl::post_ops eltw_p_op;
                    float optimized_alpha = cur_alpha * prev_alpha * prev_scale;
                    float optimized_beta = cur_alpha * prev_beta * prev_scale + cur_beta;
                    float optimized_scale = cur_scale;
                    eltw_p_op.append_eltwise(optimized_scale, cur_alg, optimized_alpha, optimized_beta);

                    // Combine 2 eltwises into one
                    add_post_op(cur_type, eltw_p_op, optimized_p_ops, 0);
                } else if (eltw_linear_and_eltw_non_linear) {
                    dnnl::post_ops eltw_p_op;
                    eltw_p_op.append_eltwise(cur_scale * prev_scale * cur_alpha, prev_alg, prev_alpha, prev_beta);

                    // Combine 2 eltwises into one
                    add_post_op(prev_type, eltw_p_op, optimized_p_ops, 0);
                }

                if (eltw_linear_and_eltw_linear || eltw_linear_and_eltw_non_linear) {
                    // Marked current and previous eltwise operations as 'optimized' (they will be ignored on the next iteration of cycle)
                    cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;
                    cur_post_ops[prev_post_op_idx].op_type = get_optimized_eltwise_type(prev_type);

                    // Set the flag if extra optimizations checking is needed
                    if (cur_post_op_idx < post_ops_size - 1) {
                        if (type_is_eltwise_linear(cur_post_ops[cur_post_op_idx + 1].op_type) ||
                            type_is_binary_add_or_mul(cur_post_ops[cur_post_op_idx + 1].op_type) ||
                            type_is_optimized_eltwise(cur_post_ops[cur_post_op_idx + 1].op_type)) {
                            optimization_is_completed = true;
                        }
                    }

                    cur_ops_pair_is_optimized = true;
                }
            } else if (bin_and_eltw) {
                dnnl::algorithm alg;
                dnnl::memory::desc desc;
                float scale, alpha, beta;

                cldnn::program_node& cur_node = get_dependency(cur_post_ops[cur_post_op_idx].mem_dep);

                p_ops.get_params_binary(cur_idx, alg, desc);
                p_ops.get_params_eltwise(prev_idx, scale, alg, alpha, beta);

                // Eltwise operations can use runtime non-constant data buffers, so check that memory buffers consist of constant data only
                auto bin_ops_can_be_optimized = cur_node.is_type<data>() && cur_node.is_constant() &&
                                                cur_node.get_users().size() == 1 && desc.data_type() == dnnl_f32;

                auto bin_add_and_eltw = alpha == 1.0f && scale == 1.0f && type_is_binary_add(cur_type) && bin_ops_can_be_optimized;
                auto bin_mul_and_eltw = beta == 0.f && type_is_binary_mul(cur_type) && bin_ops_can_be_optimized;

                if (bin_add_and_eltw || bin_mul_and_eltw) {
                    memory::ptr cur_bin_mem_ptr = cur_node.as<data>().get_attached_memory_ptr();
                    if (cur_bin_mem_ptr == nullptr)
                        throw std::runtime_error("OneDNN post-ops optimization error: nonexistent node for bin + eltw");
                    auto& stream = cur_bin_mem_ptr->get_engine()->get_program_stream();
                    mem_lock<float, mem_lock_type::read_write> bin_and_eltw_lock(cur_bin_mem_ptr, stream);

                    size_t cur_bin_mem_size = cur_node.get_output_layout().count();

                    // Update all binary coefficients
                    if (bin_add_and_eltw) {
                        for (size_t data_idx = 0; data_idx < cur_bin_mem_size; data_idx++) {
                            bin_and_eltw_lock[data_idx] += beta;
                        }
                    } else {
                        for (size_t data_idx = 0; data_idx < cur_bin_mem_size; data_idx++) {
                            bin_and_eltw_lock[data_idx] *= alpha * scale;
                        }
                    }

                    // Marked previous eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                    cur_post_ops[prev_post_op_idx].op_type = onednn_post_op_type::optimized;

                    cur_ops_pair_is_optimized = true;
                }
            } else if (eltw_and_bin) {
                dnnl::algorithm alg;
                dnnl::memory::desc desc;
                float scale, alpha, beta;

                cldnn::program_node& prev_node = get_dependency(cur_post_ops[prev_post_op_idx].mem_dep);

                p_ops.get_params_eltwise(cur_idx, scale, alg, alpha, beta);
                p_ops.get_params_binary(prev_idx, alg, desc);

                // Eltwise operations can use runtime non-constant data buffers, so check that memory buffers consist of constant data only
                auto bin_ops_can_be_optimized = prev_node.is_type<data>() && prev_node.is_constant() &&
                                                prev_node.get_users().size() == 1 && desc.data_type() == dnnl_f32;

                auto eltw_and_bin_add = alpha == 1.0f && scale == 1.0f && type_is_binary_add(prev_type) && bin_ops_can_be_optimized;
                auto eltw_and_bin_mul = beta == 0.f && type_is_binary_mul(prev_type) && bin_ops_can_be_optimized;

                if (eltw_and_bin_add || eltw_and_bin_mul) {
                    memory::ptr prev_bin_mem_ptr = prev_node.as<data>().get_attached_memory_ptr();
                    if (prev_bin_mem_ptr == nullptr)
                        throw std::runtime_error("OneDNN post-ops optimization error: nonexistent node for eltw + bin");
                    auto& stream = prev_bin_mem_ptr->get_engine()->get_program_stream();
                    mem_lock<float, mem_lock_type::read_write> eltw_and_bin_lock(prev_bin_mem_ptr, stream);

                    size_t prev_bin_mem_size = prev_node.get_output_layout().count();

                    // Update all binary coefficients
                    if (eltw_and_bin_add) {
                        for (size_t data_idx = 0; data_idx < prev_bin_mem_size; data_idx++) {
                            eltw_and_bin_lock[data_idx] += beta;
                        }
                    } else {
                        for (size_t data_idx = 0; data_idx < prev_bin_mem_size; data_idx++) {
                            eltw_and_bin_lock[data_idx] *= alpha * scale;
                        }
                    }

                    // Marked current eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                    cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;

                    cur_ops_pair_is_optimized = true;
                }
            } else if (sum_and_eltw) {
                dnnl::algorithm alg;
                float sum_scale, eltw_scale, alpha, beta;

                dnnl::algorithm next_alg;
                float next_scale, next_alpha, next_beta;
                size_t next_idx = cur_idx + 1;
                size_t next_post_op_idx = cur_post_op_idx + 1;

                bool can_optimize_eltw_and_sum = false;

                if (cur_post_op_idx < post_ops_size - 1) {
                    auto next_type = cur_post_ops[next_post_op_idx].op_type;
                    if (type_is_eltwise_linear(next_type)) {
                        p_ops.get_params_eltwise(next_idx, next_scale, next_alg, next_alpha, next_beta);

                        if (next_beta == 0)
                            can_optimize_eltw_and_sum = true;
                    }
                }

                // Try to optimize eltwise (any) + sum + eltwise_linear (with beta = 0) chain of operations
                if (can_optimize_eltw_and_sum) {
                    dnnl::memory::data_type data_type;
                    p_ops.get_params_sum(cur_idx, sum_scale, data_type);
                    p_ops.get_params_eltwise(prev_idx, eltw_scale, alg, alpha, beta);

                    dnnl::post_ops eltw_p_op_prev, sum_p_op;

                    eltw_p_op_prev.append_eltwise(eltw_scale * next_alpha * next_scale, alg, alpha, beta);
                    if (is_type<convolution>()) {
                        sum_p_op.append_sum(sum_scale * next_alpha, data_type);
                    } else {
                        sum_p_op.append_sum(sum_scale * next_alpha);
                    }
                    add_post_op(prev_type, eltw_p_op_prev, optimized_p_ops, 0);
                    add_post_op(cur_type, sum_p_op, optimized_p_ops, 0);

                    // Marked current, previous and next operations as 'optimized' (they will be ignored on the next iteration of cycle)
                    cur_post_ops[prev_post_op_idx].op_type = get_optimized_eltwise_type(prev_type);
                    cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized_sum;
                    cur_post_ops[next_post_op_idx].op_type = onednn_post_op_type::optimized;

                    // Set the flag if extra optimizations checking is needed
                    if (next_post_op_idx < post_ops_size - 1) {
                        if (type_is_eltwise_linear(cur_post_ops[next_post_op_idx + 1].op_type) ||
                            type_is_optimized_eltwise(cur_post_ops[next_post_op_idx + 1].op_type)) {
                            optimization_is_completed = true;
                        }
                    }

                    cur_ops_pair_is_optimized = true;
                }
            } else if (eltw_and_scale) {
                dnnl::algorithm alg;
                float eltw_scale, alpha, beta;

                cldnn::program_node& prev_node = get_dependency(cur_post_ops[prev_post_op_idx].mem_dep);

                p_ops.get_params_eltwise(cur_idx, eltw_scale, alg, alpha, beta);

                // Eltwise can be inserted into the output_scale if cur_beta is equal to 0.f
                if (beta == 0.f && prev_node.get_output_layout().data_type == data_types::f32) {
                    memory::ptr prev_scale_mem_ptr = prev_node.as<data>().get_attached_memory_ptr();
                    if (prev_scale_mem_ptr == nullptr)
                        throw std::runtime_error("OneDNN post-ops optimization error: nonexistent node for eltw + scale");
                    auto& stream = prev_scale_mem_ptr->get_engine()->get_program_stream();
                    mem_lock<float, mem_lock_type::read_write> eltw_and_scale_lock(prev_scale_mem_ptr, stream);

                    size_t prev_scale_mem_size = prev_node.get_output_layout().count();

                    // Update all scale coefficients
                    for (size_t data_idx = 0; data_idx < prev_scale_mem_size; data_idx++) {
                        eltw_and_scale_lock[data_idx] *= alpha * eltw_scale;
                    }

                    // Marked current eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                    cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;

                    cur_ops_pair_is_optimized = true;
                }
            }
        }

        // If no optimizations have been applied then copy post-op info into the new optimized_p_ops structure
        if (!(has_out_scales(attr) && prev_post_op_idx == 0) && !cur_ops_pair_is_optimized) {
            add_post_op(prev_type, p_ops, optimized_p_ops, prev_idx);
        }

        if (cur_post_op_idx == post_ops_size - 1 && !cur_ops_pair_is_optimized) {
            add_post_op(cur_type, p_ops, optimized_p_ops, cur_idx);
            optimization_done = true;
        } else if (cur_post_ops[cur_post_op_idx].op_type != onednn_post_op_type::optimized) {
            cur_post_op_idx++;
            prev_post_op_idx++;
        }
    }

    optimization_is_completed = !optimization_is_completed;

    add_onednn_fused_primitives(cur_post_ops);

    return optimized_p_ops;
}


void program_node::init_onednn_primitive_attributes() {
    const std::vector<fused_primitive_desc>& cldnn_post_ops = get_fused_primitives();
    auto attrs = std::make_shared<dnnl::primitive_attr>();
    dnnl::post_ops post_ops;
    size_t memory_offset = 0;

    // Create onednn post-ops list related to the current node
    std::vector<fused_primitive_desc_onednn> fused_ops;

    // Added this for debug purposes only
    size_t empty_mem = 0xff;

    // Add information about post-operation into the list, update indices
    auto update_onednn_post_op_list = [&](onednn_post_op_type type, size_t m_dep) {
        fused_primitive_desc_onednn cur_op_desc = { type, memory_offset, m_dep };
        fused_ops.push_back(cur_op_desc);

        auto has_memory_buffers = type == onednn_post_op_type::binary_add ||
                                  type == onednn_post_op_type::binary_mul ||
                                  type == onednn_post_op_type::binary_max ||
                                  type == onednn_post_op_type::binary_min ||
                                  type == onednn_post_op_type::binary_relu ||
                                  type == onednn_post_op_type::scale ||
                                  type == onednn_post_op_type::sum;
        if (has_memory_buffers)
            memory_offset++;
    };

    for (size_t idx = 0; idx < cldnn_post_ops.size(); idx++) {
        auto node = cldnn_post_ops[idx].node;

        if (node->is_type<activation>()) {
            auto fused_desc = node->as<activation>().get_primitive();;
            if (fused_desc->activation_function == cldnn::activation_func::relu_negative_slope
                && !fused_desc->additional_params_input.empty()) {
                auto dep_idx = cldnn_post_ops[idx].dep_start_idx;
                int oc_dim = node->get_output_layout().size.feature.size();
                post_ops.append_prelu(1 << oc_dim);
                update_onednn_post_op_list(onednn_post_op_type::binary_relu, dep_idx);
            } else if (fused_desc->activation_function == cldnn::activation_func::hard_sigmoid) {
                // Splits hard_sigmoid activation into eltwise_linear, min and max.
                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear,
                    fused_desc->additional_params.a, fused_desc->additional_params.b);
                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip, 0.0f, 1.0f);
                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                update_onednn_post_op_list(onednn_post_op_type::eltwise_clip, empty_mem);
            } else {
                dnnl::algorithm alg = onednn::convert_activation_func(fused_desc->activation_function);
                post_ops.append_eltwise(1.0f, alg, fused_desc->additional_params.a, fused_desc->additional_params.b);
                update_onednn_post_op_list(onednn_post_op_type::eltwise_act, empty_mem);
            }
        } else if (node->is_type<eltwise>()) {
            auto& e_node = node->as<eltwise>();
            auto dep_idx = cldnn_post_ops[idx].dep_start_idx;
            auto in = get_dependency(dep_idx).get_output_layout();

            if (e_node.get_primitive()->mode == eltwise_mode::sum) {
                if (program_helpers::needs_onednn_sum_post_op(e_node, in)) {
                    if (is_type<convolution>()) {
                        post_ops.append_sum(1.0f, onednn::convert_data_type(in.data_type));
                    } else {
                        post_ops.append_sum(1.0f);
                    }
                    update_onednn_post_op_list(onednn_post_op_type::sum, dep_idx);
                } else {
                    dnnl::memory::desc in_desc = onednn::layout_to_memory_desc(in);
                    post_ops.append_binary(dnnl::algorithm::binary_add, in_desc);
                    update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx);
                }
            } else {
                if (in.size.spatial[0] > 1 || in.size.spatial[1] > 1 || in.size.batch[0] > 1)
                    throw std::runtime_error("Unsupported eltwise mode for fused onednn op");
                if (idx == 0 && !has_out_scales(attrs)) {
                    int mask = in.count() > 1 ? 2 : 0;
                    attrs->set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
                    update_onednn_post_op_list(onednn_post_op_type::scale, dep_idx);
                } else {
                    dnnl::memory::desc in_desc = onednn::layout_to_memory_desc(in, dnnl::memory::format_tag::ab, true);
                    post_ops.append_binary(dnnl::algorithm::binary_mul, in_desc);
                    update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx);
                }
            }
        } else if (node->is_type<quantize>()) {
            auto& q_node = node->as<quantize>();
            auto dep_idx = cldnn_post_ops[idx].dep_start_idx;

            // ********************************* Common case with output range usage ********************************* //
            if (q_node.get_per_tensor_output_range() && q_node.get_output_lo_val() < q_node.get_output_hi_val()) {
                // 1. pre-scale & pre-shift
                {
                    if (q_node.get_per_tensor_input_scale() && q_node.get_per_tensor_input_shift()) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), q_node.get_input_shift_val());
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_node.get_per_tensor_input_scale()) {
                            post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), 0.0f);
                            update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                        } else {
                            auto in_scale = get_dependency(dep_idx++).get_output_layout();
                            if (idx == 0 && !has_out_scales(attrs) && in_scale.data_type == data_types::f32 &&
                                is_type<convolution>() &&
                                !data_type_traits::is_floating_point(get_dependency(0).get_output_layout().data_type)) {
                                int mask = in_scale.count() > 1 ? 2 : 0;
                                attrs->set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
                                update_onednn_post_op_list(onednn_post_op_type::scale, dep_idx - 1);
                            } else {
                                dnnl::memory::desc in_scale_desc = onednn::layout_to_memory_desc(in_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, in_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                            }
                        }

                        if (q_node.get_need_pre_shift()) {
                            if (q_node.get_per_tensor_input_shift()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_input_shift_val());
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto in_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc in_shift_desc = onednn::layout_to_memory_desc(in_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, in_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                            }
                        }
                    }
                }

                // 2. round
                auto out_dt = cldnn_post_ops[idx].output_layout.data_type;
                {
                    bool output_type_is_int8 = out_dt == data_types::u8 || out_dt == data_types::i8;
                    if (!output_type_is_int8) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_round, empty_mem);
                    }
                }

                // 3. post-scale & post-shift
                {
                    if (q_node.get_need_post_scale() && q_node.get_need_post_shift() &&
                        q_node.get_per_tensor_output_scale() && q_node.get_per_tensor_output_shift()) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), q_node.get_output_shift_val());
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_node.get_need_post_scale()) {
                            if (q_node.get_per_tensor_output_scale()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_scale = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_scale_desc = onednn::layout_to_memory_desc(out_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, out_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                            }
                        }

                        if (q_node.get_need_post_shift()) {
                            if (q_node.get_per_tensor_output_shift()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_output_shift_val());
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_shift_desc = onednn::layout_to_memory_desc(out_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, out_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                            }
                        }
                    }
                }

                // 4. clamp
                {
                    if (q_node.get_need_clamp()) {
                        float out_lo = q_node.get_need_min_clamp() ? q_node.get_output_lo_val() : data_type_traits::min<float>(out_dt);
                        float out_hi = q_node.get_need_max_clamp() ? q_node.get_output_hi_val() : data_type_traits::max<float>(out_dt);
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip, out_lo, out_hi);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_clip, empty_mem);
                    }
                }
            // ********************************* Rare case with input range usage ********************************* //
            } else {
                // 1. clamp
                {
                    if (q_node.get_need_clamp()) {
                        auto in_lo = get_dependency(dep_idx++).get_output_layout();
                        auto in_hi = get_dependency(dep_idx++).get_output_layout();
                        dnnl::algorithm clamp_max = dnnl::algorithm::binary_max;
                        dnnl::algorithm clamp_min = dnnl::algorithm::binary_min;
                        dnnl::memory::desc in_lo_desc = onednn::layout_to_memory_desc(in_lo, dnnl::memory::format_tag::ab, true);
                        dnnl::memory::desc in_hi_desc = onednn::layout_to_memory_desc(in_hi, dnnl::memory::format_tag::ab, true);

                        post_ops.append_binary(clamp_max, in_lo_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_max, dep_idx - 2);
                        post_ops.append_binary(clamp_min, in_hi_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_min, dep_idx - 1);
                    }
                }

                // 2. pre-scale & pre-shift
                {
                    if (q_node.get_per_tensor_input_scale() && q_node.get_per_tensor_input_shift()) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), q_node.get_input_shift_val());
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_node.get_per_tensor_input_scale()) {
                            post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), 0.0f);
                            update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                        } else {
                            auto in_scale = get_dependency(dep_idx++).get_output_layout();
                            if (idx == 0 && !q_node.get_need_clamp() && !has_out_scales(attrs) && in_scale.data_type == data_types::f32 &&
                                is_type<convolution>() &&
                                !data_type_traits::is_floating_point(get_dependency(0).get_output_layout().data_type)) {
                                int mask = in_scale.count() > 1 ? 2 : 0;
                                attrs->set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
                                update_onednn_post_op_list(onednn_post_op_type::scale, dep_idx - 1);
                            } else {
                                dnnl::memory::desc in_scale_desc = onednn::layout_to_memory_desc(in_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, in_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                            }
                        }

                        if (q_node.get_need_pre_shift()) {
                            if (q_node.get_per_tensor_input_shift()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_input_shift_val());
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto in_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc in_shift_desc = onednn::layout_to_memory_desc(in_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, in_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                            }
                        }
                    }
                }

                // 3. round
                {
                    post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
                    update_onednn_post_op_list(onednn_post_op_type::eltwise_round, empty_mem);
                }

                // 4. post-scale & post-shift
                {
                    if (q_node.get_need_post_scale() && q_node.get_need_post_shift() &&
                        q_node.get_per_tensor_output_scale() && q_node.get_per_tensor_output_shift()) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), q_node.get_output_shift_val());
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_node.get_need_post_scale()) {
                            if (q_node.get_per_tensor_output_scale()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_scale = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_scale_desc = onednn::layout_to_memory_desc(out_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, out_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                            }
                        }

                        if (q_node.get_need_post_shift()) {
                            if (q_node.get_per_tensor_output_shift()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_output_shift_val());
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_shift = get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_shift_desc = onednn::layout_to_memory_desc(out_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, out_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                            }
                        }
                    }
                }
            }
        } else if (node->is_type<reorder>()) {
            continue;
        } else {
            throw std::runtime_error("Unsupported fused op of " + node->get_primitive()->type_string() + " type for oneDNN primitive");
        }
    }

    if (cldnn_post_ops.size() && get_fused_activations_funcs().size())
        throw std::runtime_error("Unsupported mix of fused ops and activations");

    for (size_t i = 0; i < get_fused_activations_funcs().size(); i++) {
        auto activation_type = get_fused_activations_funcs()[i];
        auto params = get_fused_activations_params()[i];
        dnnl::algorithm alg = onednn::convert_activation_func(activation_type);
        post_ops.append_eltwise(1.0f, alg, params.a, params.b);
        update_onednn_post_op_list(onednn_post_op_type::eltwise_act, empty_mem);
    }

    // Trying to optimize more than 1 post-ops
    if (fused_ops.size() > 1) {
        dnnl::post_ops optimized_post_ops = post_ops;
        bool optimization_is_finished = false;

        add_onednn_fused_primitives(fused_ops);

        // Trying to combine multiplications and additions which are placed one after another.
        // We do it in the cycle because some optimization cases can be simplified again from time to time
        do {
            optimized_post_ops = try_optimize_post_ops(optimized_post_ops, attrs, optimization_is_finished);
        } while (!optimization_is_finished);

        attrs->set_post_ops(optimized_post_ops);
    } else {
        // Set post-ops without any optimizations
        add_onednn_fused_primitives(fused_ops);
        attrs->set_post_ops(post_ops);
    }

    add_onednn_attrs(attrs);
}

#endif // ENABLE_ONEDNN_FOR_GPU
