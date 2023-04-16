# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino.runtime as ov

#! [dynamic_shape]
core = ov.Core()


model = core.read_model("model.xml")

# reshape static model to dynamic model
model.reshape([-1, -1])

# compile model and create infer request
compiled_model = core.compile_model(model, "GPU")
infer_request = compiled_model.create_infer_request()

B = 10
F = 384

# create input tensor with specific shape
input_tensor = ov.Tensor(model.input().element_type, [B, F])

# ...

infer_request.infer([input_tensor])

#! [dynamic_shape]
