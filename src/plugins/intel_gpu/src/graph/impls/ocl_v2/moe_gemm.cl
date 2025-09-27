/*******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "include/batch_headers/generic_vector_ops.cl"
#include "include/batch_headers/tile_ops.cl"

#define N INPUT0_BATCH_NUM // input_M
#define K INPUT1_SIZE_Y
#define M INPUT1_FEATURE_NUM // output_N
#define NUM_EXPERTS INPUT1_BATCH_NUM

DECLARE_2D_TILE(c_tile_type_half, half, SUBGROUP_SIZE, 8, 4, 2, 1)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm)(OPTIONAL_SHAPE_INFO_ARG
        const global half *input_ptr, const global half *weight_ptr, global half *out_ptr,
        const global int *input_offsets, const global int *weight_offsets, const global int* output_offsets,
        const global int *tokens_lengths) {
    const uint active_expert_id = get_global_id(2);
    input_ptr += input_offsets[active_expert_id];
    weight_ptr += weight_offsets[active_expert_id];
    out_ptr += output_offsets[active_expert_id];
 
    const int cur_token_len = tokens_lengths[active_expert_id]; // TODO
    if (cur_token_len == 0)
        return;        
    const uint sg_i = sub_group_broadcast(get_local_id(0)/SUBGROUP_SIZE, 0);
    const uint sg_j = sub_group_broadcast(get_local_id(1), 0);

    // start points of this sg
    const uint wg_i0 = get_group_id(0) * ugemm_moe_wg_tile_m;
    const uint wg_j0 = get_group_id(1) * ugemm_moe_wg_tile_n;
    const uint sg_i0 = wg_i0 + sg_i * ugemm_moe_sg_tile_m;
    const uint sg_j0 = wg_j0 + sg_j * ugemm_moe_sg_tile_n;

    if (sg_j0 >= cur_token_len || sg_i0 >= M)
        return;     /* early exit if outside batch */

    const int ld_weight = K;
    const int ld_input = K;

    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, M, N, K, wg_i0, wg_j0, 0, sg_i, sg_j);
    printf("gid[%d,%d,%d] == sg[%d, %d], sg_i0:%d, sg_j0:%d input_offset: %d, token len : %d weight_offset: %d output offset: %d w[0] : %f c_tile[0][0] : %f\n", \
        get_global_id(0), get_global_id(1), get_global_id(2), sg_i, sg_j, sg_i0, sg_j0, input_offsets[active_expert_id], cur_token_len, \
        weight_offsets[active_expert_id], output_offsets[active_expert_id], weight_ptr[0], c_tile.x[0][0]);

    c_tile_type_half c_tile_half;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            c_tile_half.x[i][j] = convert_half(c_tile.x[i][j]);
        }
    }
    tile_store(c_tile_half, out_ptr, M, N, sg_i0, sg_j0);     // note oneDNN version needs a leading dimension parameter
}
