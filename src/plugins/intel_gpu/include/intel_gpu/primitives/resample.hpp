// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "primitive.hpp"
#include "openvino/op/interpolate.hpp"

#include <map>

namespace cldnn {
/// @addtogroup cpp_api C++ API
/// @{
/// @addtogroup cpp_topology Network Topology
/// @{
/// @addtogroup cpp_primitives Primitives
/// @{

/// @brief Performs nearest neighbor/bilinear resample
/// Also supports built-in Relu @ref activation available by setting it in arguments.
struct resample : public primitive_base<resample> {
    CLDNN_DECLARE_PRIMITIVE(resample)

    using InterpolateOp = ov::op::v4::Interpolate;

    /// @brief Constructs Resample primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param scale Resample scale.
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    /// @param sample_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    resample(const primitive_id& id,
             const primitive_id& input,
             tensor output_size,
             uint32_t num_filter,
             InterpolateOp::InterpolateMode operation_type = InterpolateOp::InterpolateMode::NEAREST,
             const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          output_size(output_size),
          num_filter(num_filter),
          output_pattern({}),
          scales({}),
          axes({}),
          pads_begin({}),
          pads_end({}),
          operation_type(operation_type),
          shape_calc_mode(InterpolateOp::ShapeCalcMode::SIZES),
          antialias(0),
          cube_coeff(0.0f),
          coord_trans_mode(InterpolateOp::CoordinateTransformMode::ASYMMETRIC),
          round_mode(InterpolateOp::NearestMode::FLOOR) {
//        if (scales.size() != axes.size())
//            throw std::runtime_error("Resample's scales/axes count does not match");
        if (operation_type == InterpolateOp::InterpolateMode::LINEAR) {
            coord_trans_mode = InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
        }
    }

    /// @brief resample with dynamic pattern
    resample(const primitive_id& id,
             const primitive_id& input,
             const primitive_id& pattern_id,
             const std::vector<float>& scales,
             const std::vector<int64_t>& axes,
             const std::vector<size_t>& pads_begin = {},
             const std::vector<size_t>& pads_end = {},
             int32_t antialias = 0,
             float cube_coeff = -0.75f,
             InterpolateOp::InterpolateMode operation_type = InterpolateOp::InterpolateMode::LINEAR,
             InterpolateOp::ShapeCalcMode shape_calc_mode = InterpolateOp::ShapeCalcMode::SIZES,
             InterpolateOp::CoordinateTransformMode ctm = InterpolateOp::CoordinateTransformMode::HALF_PIXEL,
             InterpolateOp::NearestMode nm = InterpolateOp::NearestMode::ROUND_PREFER_FLOOR,
             const padding& output_padding = padding())
        : primitive_base(id, {input, pattern_id}, output_padding),
          output_size(tensor()),
          num_filter(0),
          output_pattern({}),
          scales(scales),
          axes(axes),
          pads_begin(pads_begin),
          pads_end(pads_end),
          operation_type(operation_type),
          shape_calc_mode(shape_calc_mode),
          antialias(antialias),
          cube_coeff(cube_coeff),
          coord_trans_mode(ctm),
          round_mode(nm) {
//        if (scales.size() != axes.size())
//            throw std::runtime_error("Resample's scales/axes count does not match");
    }

    /// @brief reshape with static pattern
    resample(const primitive_id& id,
             const primitive_id& input,
             const std::vector<int64_t>& output_pattern,
             const std::vector<float>& scales,
             const std::vector<int64_t>& axes,
             const std::vector<size_t>& pads_begin = {},
             const std::vector<size_t>& pads_end = {},
             int32_t antialias = 0,
             float cube_coeff = -0.75f,
             InterpolateOp::InterpolateMode operation_type = InterpolateOp::InterpolateMode::LINEAR,
             InterpolateOp::ShapeCalcMode shape_calc_mode = InterpolateOp::ShapeCalcMode::SIZES,
             InterpolateOp::CoordinateTransformMode ctm = InterpolateOp::CoordinateTransformMode::HALF_PIXEL,
             InterpolateOp::NearestMode nm = InterpolateOp::NearestMode::ROUND_PREFER_FLOOR,
             const padding& output_padding = padding())
        : primitive_base(id, {input}, output_padding),
          output_size(tensor()),
          num_filter(0),
          output_pattern(output_pattern),
          scales(scales),
          axes(axes),
          pads_begin(pads_begin),
          pads_end(pads_end),
          operation_type(operation_type),
          shape_calc_mode(shape_calc_mode),
          antialias(antialias),
          cube_coeff(cube_coeff),
          coord_trans_mode(ctm),
          round_mode(nm) {
//        if (scales.size() != axes.size())
//            throw std::runtime_error("Resample's scales/axes count does not match");
    }

    InterpolateOp::InterpolateAttrs get_attrs() const {
        return InterpolateOp::InterpolateAttrs(this->operation_type,
                                               this->shape_calc_mode,
                                               this->pads_begin,
                                               this->pads_end,
                                               this->coord_trans_mode,
                                               this->round_mode,
                                               static_cast<bool>(this->antialias),
                                               this->cube_coeff);
    }

    tensor output_size;
    /// @param num_filter Input filter. Only used by bilinear sample_type.
    uint32_t num_filter;
    /// @param output_pattern Describing output shape for spatial axes.
    std::vector<int64_t> output_pattern;
    /// @param scales Scales of spatial axes, i.e. output_shape / input_shape
    std::vector<float> scales;
    /// @param axes Interpolation axes.
    std::vector<int64_t> axes;
    /// @param pads_begin Begin paddings for input.
    std::vector<size_t> pads_begin;
    /// @param pads_end End paddings for input.
    std::vector<size_t> pads_end;
    /// @param operation_type Resample method (nearest neighbor/bilinear/caffe bilinear).
    InterpolateOp::InterpolateMode operation_type;
    /// @param shape_calc_mode Specifies which input, sizes or scales, is used to calculate an output shape.
    InterpolateOp::ShapeCalcMode shape_calc_mode;
    /// @param antialias is a flag that specifies whether to perform anti-aliasing.
    int32_t antialias;
    /// @param cube_coeff specifies the parameter a for cubic interpolation. cube_coeff is used only when mode == cubic.
    float cube_coeff;
    /// @param coord_trans_mode specifies how to transform the coordinate in the resized tensor to the coordinate in the original tensor
    InterpolateOp::CoordinateTransformMode coord_trans_mode;
    /// @param round_mode specifies round mode when mode == nearest and is used only when mode == nearest.
    InterpolateOp::NearestMode round_mode;
};
/// @}
/// @}
/// @}
}  // namespace cldnn
