// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"

#define VSIZE 8
#define VLOAD CAT(vload, VSIZE)
#define VSTORE CAT(vstore,VSIZE)
#define OUTPUT_VTYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, VSIZE)

KERNEL (gather_nonzero_ref)(const __global INPUT0_TYPE* input,
                            volatile __global INPUT1_TYPE* output_shape,
                            __global OUTPUT_TYPE* output)
{
    int local_offset = 0;
    const int local_mem_size = (64*1024) / (sizeof(OUTPUT_TYPE));
    __local OUTPUT_TYPE local_mem[local_mem_size];

    int b_inc, f_inc, y_inc, x_inc;

#if OV_INPUT_RANK == 1 // b
    #define ADD_IDXS \
        int b = input_idx_v; \
        local_mem[local_offset++] = b;
#elif OV_INPUT_RANK == 2 // bf
    #define ADD_IDXS \
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        local_mem[local_offset++] = b; \
        local_mem[local_offset++] = f;
#elif OV_INPUT_RANK == 3 // bfy
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_Y_PITCH; \
        local_mem[local_offset++] = b; \
        local_mem[local_offset++] = f; \
        local_mem[local_offset++] = y;
#elif OV_INPUT_RANK == 4 // bfyx
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_Y_PITCH; \
        int x = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (y * INPUT0_Y_PITCH)) / INPUT0_X_PITCH; \
        local_mem[local_offset++] = b; \
        local_mem[local_offset++] = f; \
        local_mem[local_offset++] = y; \
        local_mem[local_offset++] = x;
#elif OV_INPUT_RANK == 5 // bfzyx
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int z = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_Z_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (z * INPUT0_Z_PITCH)) / INPUT0_Y_PITCH; \
        int x = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (z * INPUT0_Z_PITCH) - (y * INPUT_0_Y_PITCH)) / INPUT0_X_PITCH; \
        local_mem[local_offset++] = b; \
        local_mem[local_offset++] = f; \
        local_mem[local_offset++] = z; \
        local_mem[local_offset++] = y; \
        local_mem[local_offset++] = x;
#elif OV_INPUT_RANK == 6 // bfwzyx
        int f = (input_idx_v - (b * INPUT0_BATCH_PITCH)) / INPUT0_FEATURE_PITCH; \
        int w = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH)) / INPUT0_W_PITCH; \
        int z = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (w * INPUT0_W_PITCH)) / INPUT0_Z_PITCH; \
        int y = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (w * INPUT0_W_PITCH) - (z * INPUT0_Z_PITCH)) / INPUT0_Y_PITCH; \
        int x = (input_idx_v - (b * INPUT0_BATCH_PITCH) - (f * INPUT0_FEATURE_PITCH) - (w * INPUT0_W_PITCH) - (z * INPUT0_Z_PITCH) - (y * INPUT0_Y_PITCH)) / INPUT0_X_PITCH; \
        local_mem[local_offset++] = b; \
        local_mem[local_offset++] = f; \
        local_mem[local_offset++] = w; \
        local_mem[local_offset++] = z; \
        local_mem[local_offset++] = y; \
        local_mem[local_offset++] = x;
#endif
    int input_idx = 0;
    int global_output_offset = 0;
    // load to local mem
    for (; input_idx + VSIZE <= TOTAL_DATA_SIZE; input_idx += VSIZE) {
        // flush if local mem is full
        if (local_offset + VSIZE > local_mem_size) {
            for (int tmp = 0; tmp + VSIZE <= local_mem_size; ++tmp) {
                vstore8(VLOAD(0, local_mem + tmp), 0, output + global_output_offset + tmp);
            }
            global_output_offset += local_mem_size;
            local_offset = 0;
        }
        MAKE_VECTOR_TYPE(INPUT0_TYPE, VSIZE) inputs = VLOAD(0, input + input_idx);
        for (int v = 0; v < VSIZE; ++v) {
            int input_idx_v = input_idx + v;
            if (inputs[v] != INPUT0_VAL_ZERO) {
                ADD_IDXS;
            }
        }
    }
    // leftovers
    for (;input_idx < TOTAL_DATA_SIZE; ++input_idx) {
        local_mem[local_offset++] = input_idx;
    }
    
    // write back to global mem
    int local_out_iter = 0;
    for (; local_out_iter + VSIZE < local_offset; local_out_iter += VSIZE) {
        vstore8(VLOAD(0, local_mem + local_out_iter), 0, output + global_output_offset + local_out_iter);
    }
    // leftover
    for (; local_out_iter < local_offset; ++local_out_iter) {
        output[global_output_offset + local_out_iter] = local_mem[local_out_iter];
    }
}

#undef INPUT_ORDER
