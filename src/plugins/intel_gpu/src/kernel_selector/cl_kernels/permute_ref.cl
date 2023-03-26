// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL (permute_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
#ifdef F_FIRST
    //gws(f, x * y, z * w * b)
    const uint gid_0 = get_global_id(0);
    const uint gid_1 = get_global_id(1);
    const uint gid_2 = get_global_id(2);
    const uint f = gid_0;
    const uint x = gid_1 / INPUT0_SIZE_Y;
    const uint y = gid_1 % INPUT0_SIZE_Y;
    #if INPUT0_DIMS == 4
        const uint b = gid_2;
    #elif INPUT0_DIMS == 5
        const uint b = gid_2 / INPUT0_SIZE_Z;
        const uint z = gid_2 % INPUT0_SIZE_Z;
    #else
        const uint b = gid_2 / (INPUT0_SIZE_W * INPUT0_SIZE_Z) % INPUT0_BATCH_NUM;
        const uint z = gid_2 / INPUT0_SIZE_W % INPUT0_SIZE_Z;
        const uint w = gid_2 % INPUT0_SIZE_W;
    #endif
#else
    //gws(x, y * z * w, b*f)
    const uint gid_0 = get_global_id(1);
    #if INPUT0_DIMS == 4
        const uint y = gid_0;
    #elif INPUT0_DIMS == 5
        const uint z = gid_0 / INPUT0_SIZE_Y;
        const uint y = gid_0 % INPUT0_SIZE_Y;
    #else
        const uint w = gid_0 / (INPUT0_SIZE_Y * INPUT0_SIZE_Z) % INPUT0_SIZE_W;
        const uint z = gid_0 / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
        const uint y = gid_0 % INPUT0_SIZE_Y;
    #endif

    const uint x = get_global_id(0);
    const uint f = (uint)get_global_id(2) % INPUT0_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / INPUT0_FEATURE_NUM;
#endif
    INPUT0_TYPE input_var = input[IN_IDX];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[OUT_IDX] = FUSED_OPS_RESULT;
#else
    output[OUT_IDX] = ACTIVATION(input[IN_IDX], ACTIVATION_PARAMS);
    printf("output[%d] = input[%d] (%f) , OUTPUT_X_PAD_BEFORE = %d, OFFSET = %d\n", OUT_IDX, IN_IDX, input[IN_IDX], OUTPUT_PAD_BEFORE_SIZE_X, OUTPUT_OFFSET);
#endif
}
