// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/shape_of_scale.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/primitives/shape_of_scale.hpp"

namespace ov {
namespace op {
namespace internal {
using ShapeOfScale = ov::intel_gpu::op::ShapeOfScale;
}  // namespace internal
}  // namespace op
}  // namespace ov

namespace ov {
namespace intel_gpu {

static void CreateShapeOfScaleOp(ProgramBuilder& p, const std::shared_ptr<op::ShapeOfScale>& op) {
//    validate_inputs_count(op, {2});
    validate_inputs_count(op, {1});
    auto inputs = p.GetInputInfo(op);
    std::string primitive_name = layer_type_name_ID(op);

    auto get_output_data_types = [&]() {
        std::vector<cldnn::optional_data_type> output_data_types;
        auto type = op->get_output_element_type(0);
        output_data_types.push_back(cldnn::element_type_to_data_type(type));
        return output_data_types;
    };
    auto shape_of_scale = cldnn::shape_of_scale(primitive_name,
                          inputs[0]);
    shape_of_scale.output_data_types = get_output_data_types();
    p.add_primitive(*op, shape_of_scale);
}

REGISTER_FACTORY_IMPL(internal, ShapeOfScale);

}  // namespace intel_gpu
}  // namespace ov
