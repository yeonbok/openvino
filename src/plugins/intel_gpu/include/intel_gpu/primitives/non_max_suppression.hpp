// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"

#include <vector>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs non max supression of input boxes and returns indices of selected boxes.
/// @detail Filters out boxes that have high intersection-over-union (IOU) with previously
/// selected boxes with higher score. Boxes with score higher than score_threshold are
/// filtered out. This filtering happens per class.
struct non_max_suppression : public primitive_base<non_max_suppression> {
    CLDNN_DECLARE_PRIMITIVE(non_max_suppression)

    /// @brief Creates non max supression primitive.
    /// @param id This primitive id.
    /// @param boxes_positions Id of primitive with bounding boxes.
    /// @param boxes_score Id of primitive with boxes scores per class.
    /// @param selected_indices_num Number of selected indices.
    /// @param center_point_box If true boxes are represented as [center x, center y, width, height].
    /// @param sort_result_descending Specifies whether it is necessary to sort selected boxes across batches or not.
    /// @param num_select_per_class Id of primitive producing number of boxes to select per class.
    /// @param iou_threshold Id of primitive producing threshold value for IOU.
    /// @param score_threshold Id of primitive producing threshold value for scores.
    /// @param soft_nms_sigma Id of primitive specifying the sigma parameter for Soft-NMS.
    non_max_suppression(const primitive_id& id,
                        const input_info& boxes_positions,
                        const input_info& boxes_score,
                        int selected_indices_num,
                        bool center_point_box = false,
                        bool sort_result_descending = true,
                        const primitive_id& num_select_per_class = primitive_id(),
                        const primitive_id& iou_threshold = primitive_id(),
                        const primitive_id& score_threshold = primitive_id(),
                        const primitive_id& soft_nms_sigma = primitive_id(),
                        const primitive_id& ext_prim_id = "")
        : primitive_base(id, {boxes_positions, boxes_score}, ext_prim_id, {padding()}, {optional_data_type()}, 3/*num_outputs*/)
        , selected_indices_num(selected_indices_num)
        , center_point_box(center_point_box)
        , sort_result_descending(sort_result_descending)
        , num_select_per_class(num_select_per_class)
        , iou_threshold(iou_threshold)
        , score_threshold(score_threshold)
        , soft_nms_sigma(soft_nms_sigma) {}

    int selected_indices_num;
    bool center_point_box;
    bool sort_result_descending;
    primitive_id num_select_per_class;
    primitive_id iou_threshold;
    primitive_id score_threshold;
    primitive_id soft_nms_sigma;

    std::vector<std::pair<std::reference_wrapper<const primitive_id>, int>> get_dependencies() const override {
        std::vector<std::pair<std::reference_wrapper<const primitive_id>, int>> ret;
        if (!num_select_per_class.empty())
            ret.push_back({std::ref(num_select_per_class), 0});
        if (!iou_threshold.empty())
            ret.push_back({std::ref(iou_threshold), 0});
        if (!score_threshold.empty())
            ret.push_back({std::ref(score_threshold), 0});
        if (!soft_nms_sigma.empty())
            ret.push_back({std::ref(soft_nms_sigma), 0});

        return ret;
    }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
