# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino.inference_engine as ie
import numpy as np

#! [dynamic_batch_legacy]
core = ie.IECore()

MAX_B = 8
C = 3
H = 224
W = 224
dyn_config = {"DYN_BATCH_ENABLED" : "YES"}
core.set_config(config=dyn_config, device_name="GPU")
net = core.read_network("model.xml")

# Set max batch size
net.batch_size = MAX_B

# Load network and create infer request
exec_net = core.load_network(network=net, device_name="GPU", num_requests=1)

# create input tensor with specific batch size
INPUT_B = 2
img = np.random.rand(INPUT_B, C, H, W).astype('float32')
tensor_desc = ie.TensorDesc("FP32", [INPUT_B, C, H, W], "NCHW")
input_blob = ie.Blob(tensor_desc, img)
input_blob_name = next(iter(exec_net.input_info))

ireq = exec_net.requests[0]
ireq.set_batch(INPUT_B)
ireq.set_blob(input_blob_name, input_blob)

# ...

ireq.infer()

#! [dynamic_batch_legacy]
