// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"

KERNEL(reorder_weights_nf4)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    const unsigned o = (uint)get_global_id(0);
    const unsigned i = (uint)get_global_id(1);

    const unsigned o0 = 2*o + 0;
    const unsigned o1 = 2*o + 1;

    uint input0_offset = o0*(INPUT0_IFM_NUM/2) + i/2;
    uint input1_offset = o1*(INPUT0_IFM_NUM/2) + i/2;
    uint input_idx = i % 2;

    uchar in0 = (input[input0_offset] >> i*4) & 0x0F;
    uchar in1 = (input[input1_offset] >> i*4) & 0x0F;

    uchar packed_out_channels = in0 & (in1 << 4);


    const uint osv_size = 32;
    const uint osv_byte_size = osv_size / 2;
    const uint i_offset = osv_byte_size;
    const uint os_offset = i_offset * OUTPUT_IFM_NUM;
    const uint os_idx = (o + osv_byte_size - 1) / osv_byte_size;
    const uint ov_idx = o % osv_byte_size;

    uint output_idx = os_idx * os_offset + i * i_offset + ov_idx;

    output[output_idx] = packed_out_channels;
}
