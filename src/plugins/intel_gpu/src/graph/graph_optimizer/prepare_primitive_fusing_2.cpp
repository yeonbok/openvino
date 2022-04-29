// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_helpers.h"
#include "pass_manager.h"

#include "pooling_inst.h"
#include "proposal_inst.h"
#include "roi_pooling_inst.h"
#include "quantize_inst.h"
#include "binary_convolution_inst.h"
#include "activation_inst.h"
#include "batch_to_space_inst.h"
#include "crop_inst.h"
#include "eltwise_inst.h"
#include "gemm_inst.h"
#include "lrn_inst.h"
#include "mutable_data_inst.h"
#include "mvn_inst.h"
#include "pooling_inst.h"
#include "normalize_inst.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "softmax_inst.h"
#include "scale_inst.h"
#include "resample_inst.h"
#include "depth_to_space_inst.h"
#include "space_to_depth_inst.h"
#include "gather_inst.h"
#include "gather_nd_inst.h"
#include "gather_elements_inst.h"
#include "scatter_update_inst.h"
#include "scatter_nd_update_inst.h"
#include "scatter_elements_update_inst.h"
#include "reverse_sequence_inst.h"
#include "shuffle_channels_inst.h"
#include "space_to_batch_inst.h"
#include "strided_slice_inst.h"
#include "cum_sum_inst.h"
#include "embedding_bag_inst.h"
#include "extract_image_patches_inst.h"
#include "reduce_inst.h"
#include <vector>
#include <map>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <deque>
#include "intel_gpu/runtime/error_handler.hpp"

void prepare_primitive_fusing_2::run(program& p) {
    fuse_simple_primitives(p);
}

void prepare_primitive_fusing_2::fuse_simple_primitives(program &p) {
    bool recalc_processing_order = false;
    std::map<primitive_id, std::vector<std::pair<primitive_id, size_t>>> fusing_history;
    std::cout << "fuse_simple_primitives2!!!!!!!!!!!!!!!" << std::endl;
    const uint8_t supports_immad = p.get_engine().get_device_info().supports_immad;
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto node_itr = itr++;
        auto& node = (*node_itr);
        if (node->is_output() || node->is_constant())
            continue;

        auto is_grouped_conv = [](convolution_node& node) -> bool {
            auto in_size = node.get_dependency(0).get_output_layout().size;
            return (node.get_split() > 1 && node.get_split() != in_size.feature[0]) ||
                   (node.get_groups() > 1 && node.get_groups() != static_cast<uint32_t>(in_size.feature[0]));
        };

        auto conv_supports_fusings = [&](convolution_node& node) -> bool {
            if (_lo.get_optimization_attributes().use_onednn_impls == 1)
                return true;

            // Since reorder inputs is called after this pass
            // we have to check that blocked formats can be used in the network and layer is optimized for it.
            if ((node.get_output_layout().format == format::b_fs_yx_fsv16 ||
                _lo.should_select_b_fs_yx_fsv16_layout(node, node.get_dependency(1).get_output_layout())) &&
                 !is_grouped_conv(node))
                return true;

            if ((node.get_output_layout().format == format::bfzyx &&
                (!_lo.get_optimization_attributes().b_fs_zyx_fsv16_network || !_lo.is_format_optimized(node, format::b_fs_zyx_fsv16))))
                return true;

            if ((node.get_output_layout().format == format::fs_b_yx_fsv32 ||
                (_lo.get_optimization_attributes().fs_b_yx_fsv32_network &&
                 _lo.is_format_optimized(node, format::fs_b_yx_fsv32) && node.get_primitive()->groups == 1)))
                    return true;

            const size_t in_feature = node.get_dependency(0).get_output_layout().feature();
            if ((node.get_output_layout().format == format::b_fs_zyx_fsv16 ||
                 (_lo.is_format_optimized(node, format::b_fs_zyx_fsv16) &&
                  _lo.get_optimization_attributes().b_fs_zyx_fsv16_network)) && in_feature != 3)
                return true;

            if ((node.get_output_layout().format == format::bs_fs_yx_bsv16_fsv16 ||
                 (_lo.is_format_optimized(node, format::bs_fs_yx_bsv16_fsv16) &&
                  _lo.get_optimization_attributes().bs_fs_yx_bsv16_fsv16_network)) && node.get_primitive()->groups == 1)
                return true;

            if (node.get_output_layout().format == format::bs_fs_yx_bsv32_fsv32 || _lo.is_format_optimized(node, format::bs_fs_yx_bsv32_fsv32))
                return true;

            if (node.get_output_layout().format == format::bs_fs_yx_bsv32_fsv16 || _lo.is_format_optimized(node, format::bs_fs_yx_bsv32_fsv16))
                return true;

            auto in_dt = node.get_dependency(0).get_output_layout().data_type;

            // TODO: check if that's enough for correct work
            return data_type_traits::is_i8_u8(in_dt);
        };

        auto bin_conv_supports_eltw_fusings = [](binary_convolution_node& conv_node) -> bool {
            auto& eltw_node = static_cast<const eltwise_node&>(*conv_node.get_users().front());
            auto& eltw_prim = *eltw_node.get_primitive();

            if (eltw_node.get_dependencies().size() < 2)
                return false;

            auto const_layout = eltw_node.get_dependency(1).get_output_layout();
            auto conv_layout = conv_node.get_output_layout();
            auto per_channel_eltwise = const_layout.feature() == conv_layout.feature();

            if (eltw_node.get_dependency(1).is_constant() && per_channel_eltwise &&
                (eltw_prim.mode == eltwise_mode::sum || eltw_prim.mode == eltwise_mode::prod) &&
                all_ones(conv_node.get_primitive()->dilation))
                return true;

            return false;
        };

        auto fc_supports_fusings = [](fully_connected_node& node) -> bool {
            auto in_dt = node.get_dependency(0).get_output_layout().data_type;

            return data_type_traits::is_i8_u8(in_dt);
        };

        auto gemm_supports_fusings = [](gemm_node& node) -> bool {
            bool does_support_fusings = false;
            auto in0_dt = node.get_dependency(0).get_output_layout().data_type;
            auto in1_dt = node.get_dependency(1).get_output_layout().data_type;
            auto in0_fmt = node.get_dependency(0).get_output_layout().format;
            auto in1_fmt = node.get_dependency(1).get_output_layout().format;

            if (data_type_traits::is_floating_point(in0_dt) &&
                data_type_traits::is_floating_point(in1_dt))
                does_support_fusings = true;

            if (data_type_traits::is_i8_u8(in0_dt) && in0_fmt == format::bfyx &&
                data_type_traits::is_i8_u8(in1_dt) && in1_fmt == format::bfyx) {
                if (node.inputs_count() == 3) {
                    auto in2_dt = node.get_dependency(2).get_output_layout().data_type;
                    auto in2_fmt = node.get_dependency(2).get_output_layout().format;
                    does_support_fusings = data_type_traits::is_i8_u8(in2_dt) && in2_fmt == format::bfyx ? true : false;
                } else {
                    does_support_fusings = true;
                }
            }

            return does_support_fusings;
        };

        auto mvn_supports_fusings = [](mvn_node& node) -> bool {
            auto in_dt = node.get_dependency(0).get_output_layout().data_type;
            return data_type_traits::is_i8_u8(in_dt);
        };

        auto pooling_supports_fusings = [](pooling_node& node) -> bool {
            auto pooling_mode = node.get_primitive()->mode;
            return pooling_mode != cldnn::pooling_mode::max_with_argmax;
        };

        auto dts_supports_fusings = [](depth_to_space_node& node) -> bool {
            bool input_conv = node.get_dependency(0).is_type<convolution>();
            bool out_eltw = node.get_users().front()->is_type<eltwise>();
            if (input_conv && out_eltw) {
                auto& eltw = static_cast<const eltwise&>(*node.get_users().front()->get_primitive());
                auto& conv = node.get_dependency(0).as<convolution>();
                auto eltw_mode = eltw.mode == eltwise_mode::sum;
                auto conv_size = conv.get_dependency(0).get_output_layout().spatial(0) % 128 == 0 &&
                                 conv.get_dependency(0).get_output_layout().spatial(1) % 2 == 0;
                auto format = conv.get_output_layout().format == format::bfyx;
                auto dt = conv.get_output_layout().data_type == data_types::f16;
                if (eltw_mode && conv_size && format && dt)
                    return false;
            }

            return true;
        };

        auto reduce_supports_fusings = [](reduce_node& node) -> bool {
            auto keep_dims = node.as<reduce>().get_primitive()->keep_dims;

            if (keep_dims)
                return true;

            return false;
        };

        auto eltwise_supports_fusings = [&](eltwise_node& node) -> bool {
            if (_lo.get_optimization_attributes().use_onednn_impls == 0) {
                auto out_layout = node.get_output_layout();
                if (out_layout.data_type == data_types::f16 && out_layout.batch() > 1 &&
                    (_lo.get_optimization_attributes().fs_b_yx_fsv32_network || out_layout.format == format::fs_b_yx_fsv32)) {
                    return false;
                }
            }
            return true;
        };

        auto get_users_from_fusing_history = [&](const primitive_id& id) {
            std::vector<primitive_id> users;
            for (auto fusing_info : fusing_history) {
                auto key = fusing_info.first;
                auto dep_info_vec = fusing_info.second;
                auto iter = std::find_if(dep_info_vec.begin(), dep_info_vec.end(), [&](std::pair<primitive_id, size_t>& dep_info) {
                    return (id == dep_info.first);
                });
                if (iter != dep_info_vec.end()) {
                    users.push_back(key);
                }
            }
            return users;
        };

        auto input_data_supports_fusings = [&](cldnn::program_node& input_data, primitive_id current_node_id) -> bool {
            if (input_data.get_users().size() != 1) {
                // If input_data has fused primitives,
                // find original dependency of current_node using fusing_history
                // and check the number of users of it.
                // If the node has multiple users it's not fusible.
                if (!supports_immad && input_data.has_fused_primitives()) {
                    size_t num_original_dependencies = 0;
                    auto iter = fusing_history.find(current_node_id);
                    if (iter != fusing_history.end()) {
                        // Find current_node's original dependency list
                        for (auto& prim_id : iter->second) {
                            // find input_data's fused_prims in the prim_deps_ids
                            auto& fused_descs = input_data.get_fused_primitives();
                            auto origin_input_iter = std::find_if(fused_descs.begin(), fused_descs.end(),
                                                                    [&](cldnn::fused_primitive_desc& desc) {
                                return (desc.node->id() == prim_id.first);
                            });
                            if (origin_input_iter != fused_descs.end()) {
                                auto users = get_users_from_fusing_history(origin_input_iter->node->id());
                                if (users.size() != 1) {
                                    return false;
                                }
                                num_original_dependencies++;
                            }
                        }
                    }
                    // If num_original_dependencies is zero, input_data is original parent
                    if (num_original_dependencies == 0) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        };

        auto fuse_scale_f = [&](scale_node& scale_node) {
            if (scale_node.get_dependencies().empty())
                CLDNN_ERROR_MESSAGE(scale_node.id(), "scale has invalid count of dependencies");

            auto& input_data = scale_node.get_dependency(0);
            if (input_data.get_users().size() != 1 || input_data.get_dependencies().empty())
                return;

            bool should_fuse = input_data.is_type<binary_convolution>() &&
                               all_ones(input_data.as<binary_convolution>().get_primitive()->dilation);

            should_fuse |= input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>());

            should_fuse |= input_data.is_type<fully_connected>() && fc_supports_fusings(input_data.as<fully_connected>());

            should_fuse |= input_data.is_type<gemm>() && gemm_supports_fusings(input_data.as<gemm>());

            should_fuse |= input_data.is_type<pooling>() && pooling_supports_fusings(input_data.as<pooling>());

            should_fuse |= input_data.is_type<resample>();

            should_fuse |= input_data.is_type<mvn>() && mvn_supports_fusings(input_data.as<mvn>());

            should_fuse |= input_data.is_type<normalize>() && data_type_traits::is_i8_u8(input_data.get_dependency(0).get_output_layout().data_type);

            should_fuse |= input_data.is_type<deconvolution>();

            should_fuse |= input_data.is_type<permute>();

            should_fuse |= input_data.is_type<activation>();

            should_fuse |= input_data.is_type<lrn>();

            should_fuse |= input_data.is_type<gather>();

            should_fuse |= input_data.is_type<gather_nd>();

            should_fuse |= input_data.is_type<gather_elements>();

            should_fuse |= input_data.is_type<scatter_update>();

            should_fuse |= input_data.is_type<scatter_nd_update>();

            should_fuse |= input_data.is_type<scatter_elements_update>();

            should_fuse |= input_data.is_type<depth_to_space>();

            should_fuse |= input_data.is_type<space_to_depth>();

            should_fuse |= input_data.is_type<batch_to_space>();

            should_fuse |= input_data.is_type<space_to_batch>();

            should_fuse |= input_data.is_type<reduce>() && reduce_supports_fusings(input_data.as<reduce>());

            should_fuse |= input_data.is_type<scale>();

            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>());

            if (!should_fuse)
                return;

            p.fuse_nodes(input_data, scale_node, &fusing_history);
        };

        auto fuse_quantize_f = [&](quantize_node& quantize_node) {
            auto& prog = quantize_node.get_program();
            auto input_data_opt = prog.get_node_ptr(quantize_node.get_dependency(0).id());
            if (input_data_opt->get_users().size() != 1 || input_data_opt->get_dependencies().empty())
                return;
            if (!input_data_opt->is_type<reshape>() && !input_data_opt->is_type<reorder>()) return;
            if (!input_data_opt->can_be_optimized()) return;

            auto& input_data = input_data_opt->get_dependency(0);

            auto& input_lo = quantize_node.get_dependency(1);
            auto& input_hi = quantize_node.get_dependency(2);

            auto out_layout = quantize_node.get_output_layout();
            auto in_layout = input_data.get_output_layout();
            auto out_dt = out_layout.data_type;
            auto in_dt = input_data.get_dependency(0).get_output_layout().data_type;
            auto out_dt_is_i8_u8 = data_type_traits::is_i8_u8(out_dt);
            auto in_dt_is_i8_u8 = data_type_traits::is_i8_u8(in_dt);

            bool per_tensor_values = quantize_node.get_scale_shift_opt() &&
                                     quantize_node.get_per_tensor_input_scale() &&
                                     quantize_node.get_per_tensor_input_shift() &&
                                     quantize_node.get_per_tensor_input_range() &&
                                     quantize_node.get_per_tensor_output_scale() &&
                                     quantize_node.get_per_tensor_output_shift() &&
                                     quantize_node.get_per_tensor_output_range();

            bool should_fuse = input_data.is_type<binary_convolution>() &&
                               ((out_dt == data_types::bin &&
                               quantize_node.get_dependencies().size() == 5 &&
                               ((in_layout.feature() == input_lo.get_output_layout().feature() &&
                                 in_layout.feature() == input_hi.get_output_layout().feature()) ||
                                (input_lo.get_output_layout().feature() == 1 &&
                                 input_hi.get_output_layout().feature() == 1)))) &&
                                 all_ones(input_data.as<binary_convolution>().get_primitive()->dilation);

            auto expected_format = _lo.get_preferred_format(input_data);

            should_fuse |= input_data.is_type<convolution>() && conv_supports_fusings(input_data.as<convolution>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           ((out_dt == data_types::f32 || out_dt == data_types::f16)  ||
                            in_layout.format == format::b_fs_yx_fsv16 ||
                            in_layout.format == format::bs_fs_yx_bsv32_fsv16 ||
                            (_lo.should_select_b_fs_yx_fsv16_layout(input_data.as<convolution>(), input_data.get_dependency(1).get_output_layout()) &&
                             !is_grouped_conv(input_data.as<convolution>())) ||
                           // Avoid fusing to b_fs_yx_fsv16 (and similar) kernels
                           expected_format == cldnn::format::bs_fs_yx_bsv32_fsv16 /* Allow quantization fusing for onednn */ ||
                           (in_dt_is_i8_u8 && out_dt_is_i8_u8));

            should_fuse |= input_data.is_type<pooling>() && quantize_node.get_scale_shift_opt() &&
                           pooling_supports_fusings(input_data.as<pooling>());

            should_fuse |= input_data.is_type<fully_connected>() && fc_supports_fusings(input_data.as<fully_connected>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           out_dt_is_i8_u8;

            should_fuse |= input_data.is_type<lrn>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gemm>() && gemm_supports_fusings(input_data.as<gemm>()) &&
                           quantize_node.get_scale_shift_opt() &&
                           out_dt_is_i8_u8;

            should_fuse |= input_data.is_type<resample>() &&
                           quantize_node.get_scale_shift_opt() &&
                           out_dt_is_i8_u8;

            should_fuse |= input_data.is_type<mvn>() && mvn_supports_fusings(input_data.as<mvn>()) &&
                           quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<activation>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<normalize>() && quantize_node.get_scale_shift_opt() &&
                           in_dt_is_i8_u8;

            should_fuse |= input_data.is_type<deconvolution>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather_nd>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<gather_elements>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<scatter_update>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<scatter_nd_update>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<scatter_elements_update>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<permute>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<depth_to_space>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<space_to_depth>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<batch_to_space>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<space_to_batch>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<reduce>() &&
                           reduce_supports_fusings(input_data.as<reduce>())
                           && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<eltwise>() && eltwise_supports_fusings(input_data.as<eltwise>()) && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<scale>() && quantize_node.get_scale_shift_opt();

            should_fuse |= input_data.is_type<softmax>() &&
                           input_data.as<softmax>().get_primitive()->dimension == softmax::dimension_t::normalize_f &&
                           per_tensor_values;


            if (!should_fuse)
                return;

            p.fuse_nodes_through_optimized_reshape(input_data, quantize_node, *input_data_opt, &fusing_history);
        };

       program_helpers::do_for_types<scale, quantize>(*node,
                fuse_scale_f,
                fuse_quantize_f);
    }

    // Need to update processing order to handle cases when peer node processing number is greater
    // than fused node one
    if (recalc_processing_order)
        p.get_processing_order().calc_processing_order(p);
}