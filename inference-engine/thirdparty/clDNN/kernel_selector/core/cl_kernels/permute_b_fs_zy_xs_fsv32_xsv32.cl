// Copyright (c) 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/include_all.cl"
#define LOCAL_MEM_NUM_ENTRIES (TILE_SIZE_W*TILE_SIZE_H)

KERNEL (permute_b_fs_zy_xs_fsv32_xsv32)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    //|dim2:bf|dim1:yz|dim0:x
    const uint x = get_global_id(0);
    const uint z = get_global_id(1) / INPUT0_SIZE_Y;
    const uint y = get_global_id(1) % INPUT0_SIZE_Y;   
    const uint f = (uint)get_global_id(2) % (INPUT0_FEATURE_NUM/TILE_SIZE_H);
    const uint b = (uint)get_global_id(2) / (INPUT0_FEATURE_NUM/TILE_SIZE_H);

    __local float read_buf[TILE_SIZE_W * TILE_SIZE_H * LWS];
    __local float transpose_buf[TILE_SIZE_W * TILE_SIZE_H * LWS];
    int local_id_x = get_local_id(0);
    int local_buf_offset = local_id_x * TILE_SIZE_W * TILE_SIZE_H;
    // read partial data
    //barrier(CLK_LOCAL_MEM_FENCE);
    for(int lh = 0; lh < TILE_SIZE_H; ++lh) {
        for(int lw = 0; lw < TILE_SIZE_W; ++lw) {
            unsigned int input_idx = b * INPUT0_BATCH_PITCH +  (TILE_SIZE_H *f + lh) * INPUT0_FEATURE_PITCH + z * INPUT0_Z_PITCH + y * INPUT0_Y_PITCH +  (TILE_SIZE_W * x + lw) * INPUT0_X_PITCH;
            int target_idx =  lh * TILE_SIZE_W + lw;
            read_buf[local_buf_offset + target_idx] = input[input_idx];
        }
    }
    //barrier(CLK_LOCAL_MEM_FENCE);
    // transpose
    for(int lh = 0; lh < TILE_SIZE_H; ++lh) {
        for(int lw = 0; lw < TILE_SIZE_W; ++lw) {
            unsigned int src = local_buf_offset + lh * TILE_SIZE_W + lw;
            unsigned int dst = local_buf_offset + lw * TILE_SIZE_W + lh;
            transpose_buf[dst] = read_buf[src];
        }
    }
    //barrier(CLK_LOCAL_MEM_FENCE);
    // write to ddr
    for(int lh = 0; lh < TILE_SIZE_W; ++lh) {
        for(int lw = 0; lw < TILE_SIZE_H; ++lw) {
            // b, f, z, x, y
            //unsigned int output_idx = OUTPUT_GET_INDEX(b, z, y, x * TILE_SIZE_W + lh, f * TILE_SIZE_H + lw);
            unsigned int output_idx = b * OUTPUT_BATCH_PITCH + z * OUTPUT_FEATURE_PITCH + y * OUTPUT_Z_PITCH + (x * TILE_SIZE_W + lh) * OUTPUT_Y_PITCH +  (TILE_SIZE_H *f + lw) * OUTPUT_X_PITCH;
            output[output_idx] = (float)transpose_buf[local_buf_offset + lh * TILE_SIZE_H + lw];
        }
    }

    //barrier(CLK_LOCAL_MEM_FENCE);
}
