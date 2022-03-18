// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(out_order) OUTPUT_GET_INDEX(out_order)

#if INPUT0_DIMS == 4
    #define IN_ORDER in_b,in_f,in_y,in_x
#elif INPUT0_DIMS == 5
    #define IN_ORDER in_b,in_f,in_z,in_y,in_x
#else
    #define IN_ORDER in_b,in_f,in_w,in_z,in_y,in_x
#endif

#if INPUT1_DIMS == 4
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif INPUT1_DIMS == 5
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#else
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
#endif

#if OUTPUT_DIMS == 4
    #define OUT_ORDER out_b,out_f,out_y,out_x
#elif OUTPUT_DIMS == 5
    #define OUT_ORDER out_b,out_f,out_z,out_y,out_x
#else
    #define OUT_ORDER out_b,out_f,out_w,out_z,out_y,out_x
#endif

#define INDICES_MAX_DIM 6

KERNEL(gather_elements_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    //TODO: Calculate Index or INPUT0_GET_INDEX(b,f,y,x)같은거 사용?
    //아래는 임시로 아무값이나 줌
    #if INPUT1_DIMS == 4
        const uint idx_b = 1;
        const uint idx_f = dim2;
        const uint idx_x = dim0;
        const uint idx_y = dim1;
        const uint idx_z = 0;
        const uint idx_w = 0;
        const uint in_b = 1;
        const uint in_f = 1;
        const uint in_x = 1;
        const uint in_y = 1;

        const uint idx_arr[INPUT1_DIMS*2] = {idx_b, idx_f, idx_y, idx_x, 0, 0, 0, 0};
        const uint idx_dim[INPUT1_DIMS] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    #elif INPUT1_DIMS == 5
        //TODO
    #else
        //TODO
    #endif

    const int idx = GET_UPDATES_INDEX(INPUT1, IDX_ORDER);

    const uint data_idx = GET_UPDATES_INDEX(INPUT0, IN_ORDER);

    // Calculate output index
    const uint out_x = idx_x;
    const uint out_y = idx_y;
    const uint out_z = idx_z;
    const uint out_w = idx_w;
    const uint out_f = idx_f;
    const uint out_b = idx_b;

    const uint output_idx = GET_OUTPUT_INDEX(OUT_ORDER);

    // Copy data to output as slice size
    #if HAS_FUSED_OPS
        #if OUTPUT_DIMS == 4
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint f_pitch = y_pitch * OUTPUT_SIZE_Y;
        #elif OUTPUT_DIMS == 5
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint f_pitch = z_pitch * OUTPUT_SIZE_Z;
        #else
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint w_pitch = z_pitch * OUTPUT_SIZE_Z;
            const uint f_pitch = w_pitch * OUTPUT_SIZE_W;
        #endif
        const uint b_pitch = f_pitch * OUTPUT_FEATURE_NUM;
    #endif

    if(!dim0 && !dim1 && !dim2)
        printf("hello?\n");
}

#undef INDICES_MAX_DIM
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef OUT_ORDER
#undef IDX_ORDER
#undef IN_ORDER
