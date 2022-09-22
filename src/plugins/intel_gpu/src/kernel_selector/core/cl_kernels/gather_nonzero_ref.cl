// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"

#define INPUT0_GET_INDEX1(idx_order) INPUT0_GET_INDEX(idx_order)

KERNEL (gather_nonzero_ref)(const __global INPUT0_TYPE* input,
                            volatile __global INPUT1_TYPE* output_shape,
                            __global OUTPUT_TYPE* output)
{
    int pos = 0;
    for (int b = 0; b < INPUT0_BATCH_NUM; ++b) {
        for (int f = 0; f < INPUT0_FEATURE_NUM; ++f) {
            for (int w = 0; w < INPUT0_SIZE_W; ++w) {
                for (int z = 0; z < INPUT0_SIZE_Z; ++z) {
                    for (int y = 0; y < INPUT0_SIZE_Y; ++y) {
                        for (int x = 0; x < INPUT0_SIZE_X; ++x) {
                            int input_idx = GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, y, x);
                            if (input[input_idx] != INPUT0_VAL_ZERO) {
                            #if OV_INPUT_RANK == 1
                                output[pos++] = b;
                            #elif OV_INPUT_RANK == 2
                                output[pos++] = b;
                                output[pos++] = f;
                            #elif OV_INPUT_RANK == 3
                                output[pos++] = b;
                                output[pos++] = f;
                                output[pos++] = y;
                            #elif OV_INPUT_RANK == 4
                                output[pos++] = b;
                                output[pos++] = f;
                                output[pos++] = y;
                                output[pos++] = x;
                            #elif OV_INPUT_RANK == 5
                                output[pos++] = b;
                                output[pos++] = f;
                                output[pos++] = z;
                                output[pos++] = y;
                                output[pos++] = x;
                            #elif OV_INPUT_RANK == 6
                                output[pos++] = b;
                                output[pos++] = f;
                                output[pos++] = w;
                                output[pos++] = z;
                                output[pos++] = y;
                                output[pos++] = x;
                            #else
                                printf("unknown rank !\n");
                            #endif
                            }
                        }
                    }
                }
            }
        }
    }
}

#undef INPUT0_GET_INDEX1
#undef INPUT_ORDER
