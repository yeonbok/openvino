// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "register.hpp"

namespace cldnn {
namespace cpu {

#define REGISTER_CPU(prim)                                \
    static detail::attach_##prim##_impl attach_##prim

void register_implementations() {
#if 0 // TODO(taylor)
    REGISTER_CPU(detection_output);
    REGISTER_CPU(proposal);
#endif
    REGISTER_CPU(non_max_suppression);
}
//
}  // namespace cpu
}  // namespace cldnn
