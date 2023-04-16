# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino.runtime as ov

#! [dynamic_shape_bounded]
core = ov.Core()


model = core.read_model("model.xml")
# reshape model with bounded dynamic shape
# where batch dimension is between 1 and 10 and feature dimension is between 32 and 64
model.reshape([(1, 10), (32, 64)])

# compile model and create infer request
compiled_model = core.compile_model(model, "GPU")
infer_request = compiled_model.create_infer_request()

B = 8
F = 48

# create input tensor with specific shape
input_tensor = ov.Tensor(model.input().element_type, [B, F])

# ...

infer_request.infer([input_tensor])

#! [dynamic_shape_bounded]
