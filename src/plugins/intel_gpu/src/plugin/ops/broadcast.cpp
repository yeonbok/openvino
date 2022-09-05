// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/broadcast.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/reshape.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCommonBroadcastOp(Program& p, const std::shared_ptr<ngraph::Node>& op, const ngraph::AxisSet axis_mapping) {
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    ov::op::BroadcastModeSpec mode = ov::op::BroadcastType::NONE;
    if (auto broadcast_v3 = std::dynamic_pointer_cast<ngraph::op::v3::Broadcast>(op)) {
        mode = broadcast_v3->get_broadcast_spec();
    } else if (auto broadcast_v1 = std::dynamic_pointer_cast<ngraph::op::v1::Broadcast>(op)) {
        switch (broadcast_v1->get_broadcast_spec().m_type) {
            case ov::op::AutoBroadcastType::NONE: mode = ov::op::BroadcastType::NONE; break;
            case ov::op::AutoBroadcastType::NUMPY: mode = ov::op::BroadcastType::NUMPY; break;
            case ov::op::AutoBroadcastType::PDPD: mode = ov::op::BroadcastType::PDPD; break;
            default:
                                                  throw ov::Exception("[GPU] Can't match Broadcast v1 mode with v3 version");
        }
    } else {
        throw ov::Exception("[GPU] Can't cast Broadcast operation to any supported version");
    }

    auto broadcastPrim = cldnn::broadcast(layerName,
            inputPrimitives[0],
            inputPrimitives[1], axis_mapping, mode, op->get_friendly_name());
    p.AddPrimitive(broadcastPrim);
    p.AddPrimitiveToProfiler(op);
    return;
}

static void CreateBroadcastOp(Program& p, const std::shared_ptr<ngraph::op::v1::Broadcast>& op) {
    p.ValidateInputs(op, {2, 3});
    if (op->get_broadcast_spec().m_type == ngraph::op::AutoBroadcastType::NONE && op->get_input_size() == 3) {
        auto axis_mapping_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        if (!axis_mapping_node)
            IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

        auto axis_mapping = axis_mapping_node->get_axis_set_val();
        CreateCommonBroadcastOp(p, op, axis_mapping);
    } else {
        // TODO: check if axis_mapping is not needed in these cases and prepending input shape with ones works fine in all cases
        CreateCommonBroadcastOp(p, op, {});
    }
}

static void CreateBroadcastOp(Program& p, const std::shared_ptr<ngraph::op::v3::Broadcast>& op) {
    p.ValidateInputs(op, {2, 3});
    ngraph::AxisSet axis_mapping;
    if (op->get_input_size() == 3) {
        auto axis_mapping_node = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(op->get_input_node_shared_ptr(2));
        if (!axis_mapping_node)
            IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";

        axis_mapping = axis_mapping_node->get_axis_set_val();
    }
    CreateCommonBroadcastOp(p, op, axis_mapping);
}

REGISTER_FACTORY_IMPL(v1, Broadcast);
REGISTER_FACTORY_IMPL(v3, Broadcast);

}  // namespace intel_gpu
}  // namespace ov
