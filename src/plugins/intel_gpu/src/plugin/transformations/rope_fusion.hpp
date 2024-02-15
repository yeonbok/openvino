// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "intel_gpu/op/rope.hpp"
#include "utils.hpp"

namespace ov {
namespace intel_gpu {

class RoPEFusionGPTNEOX : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionGPTNEOX", "0");
    RoPEFusionGPTNEOX();
};

class RoPEFusionGPTJ : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionGPTJ", "0");
    RoPEFusionGPTJ();
};

class RoPEFusionChatGLM : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionChatGLM", "0");
    RoPEFusionChatGLM(int split_output_id);
};

class RoPEFusionQwen : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionQwen", "0");
    RoPEFusionQwen(int split_output_id);
};

class RoPEFusionIOSlicing : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionIOSlicing", "0");
    RoPEFusionIOSlicing();
};

class RoPEFusionPreprocess : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionPreprocess", "0");
    RoPEFusionPreprocess();
};

class RoPEFusionCosSinPreprocess : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionCosSinPreprocess", "0");
    RoPEFusionCosSinPreprocess();
};

class EliminateStridedSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateStridedSlice", "0");
    EliminateStridedSlice();
};

class RoPEFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("RoPEFusion", "0");
    RoPEFusion() {
        add_matcher<RoPEFusionGPTNEOX>();
        add_matcher<RoPEFusionGPTJ>();
        // optional heads & tails are fused in separate matcher pass,
        // after RoPENode has been created.
        add_matcher<RoPEFusionCosSinPreprocess>();
        add_matcher<RoPEFusionIOSlicing>();
        add_matcher<RoPEFusionPreprocess>();

        add_matcher<RoPEFusionChatGLM>(0);
        add_matcher<RoPEFusionChatGLM>(1);

        add_matcher<RoPEFusionQwen>(0);
        add_matcher<RoPEFusionQwen>(1);
    }
};

}  // namespace intel_gpu
}  // namespace ov
