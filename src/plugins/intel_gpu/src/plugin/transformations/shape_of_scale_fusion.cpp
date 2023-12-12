// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_scale_fusion.hpp"

#include "intel_gpu/op/shape_of_scale.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

ShapeOfScaleFusion::ShapeOfScaleFusion() {
    using namespace ov::pass::pattern;

    // Detect ShapeOfScale decomposition pattern
    auto x = any_input();

    // shapeof
    auto shapeof = wrap_type<ov::op::v3::ShapeOf>({x});
    // gather
    auto indices = wrap_type<ov::op::v0::Constant>();
    auto axis = wrap_type<ov::op::v0::Constant>();
    auto gather = wrap_type<ov::op::v8::Gather>({shapeof, indices, axis});
    // convert
    auto convert = wrap_type<ov::op::v0::Convert>({gather});
    // sqrt
    // x^2
    auto sqrt = wrap_type<ov::op::v0::Sqrt>({convert});

    auto const_power = wrap_type<ov::op::v0::Constant>();
    auto power = wrap_type<ov::op::v1::Power>({sqrt, const_power});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto x_output = pattern_map.at(x);

        auto output_type = m.get_match_root()->get_output_element_type(0);

        auto shape_of_scale = std::make_shared<op::ShapeOfScale>(x_output,
                                             output_type);
        shape_of_scale->set_friendly_name("shape_of_scale_" + m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), shape_of_scale);
        ov::replace_node(m.get_match_root(), shape_of_scale);
        std::cout << "Matching done!" << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(power, "ShapeOfScaleFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
