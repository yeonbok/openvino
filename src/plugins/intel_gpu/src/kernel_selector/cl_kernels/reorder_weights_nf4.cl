// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_weights.cl"

KERNEL(reorder_weights_nf4)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
#if defined(INPUT0_LAYOUT_IOYX) && defined(OUTPUT_LAYOUT_OIYX)
    const uint out_byte_offset = get_global_id(0);

    const uint offset0 = out_byte_offset * 2 + 0;
    const uint offset1 = out_byte_offset * 2 + 1;

    const uint i0 = offset0 % OUTPUT_IFM_NUM;
    const uint i1 = offset1 % OUTPUT_IFM_NUM;

    const uint o0 = offset0 / OUTPUT_IFM_NUM;
    const uint o1 = offset1 / OUTPUT_IFM_NUM;

    const uint input0_offset = GET_FILTER_INDEX(INPUT0, 0, o0, i0, 0, 0);
    const uint input1_offset = GET_FILTER_INDEX(INPUT0, 0, o1, i1, 0, 0);

    const uint input0_idx = input0_offset % 2;
    const uint input1_idx = input1_offset % 2;

    INPUT0_TYPE in0 = (input[input0_offset / 2] >> input0_idx*4) & 0x0F;
    INPUT0_TYPE in1 = (input[input1_offset / 2] >> input1_idx*4) & 0x0F;

    OUTPUT_TYPE out = in0 | (in1 << 4);
    output[out_byte_offset] = out;
#else
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
#endif
}
