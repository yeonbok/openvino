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


//DECLARE_2D_TILE(c_tile_type_half, half, SUBGROUP_SIZE, 8, 4, 2, 1)

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL(moe_gemm)(OPTIONAL_SHAPE_INFO_ARG
        const global half *input_ptr, const global half *weight_ptr, global float *out_ptr,
        const global int *input_offsets, const global int *weight_offsets, const global int* output_offsets,
        const global int *n_array, int m, int k) {
    uint batch = get_group_id(2);
    input_ptr += input_offsets[batch];
    weight_ptr += weight_offsets[batch];
    out_ptr += output_offsets[batch];
    //printf("m : %d n : %d k : %d\n", m, n, k);
    int n = n_array[batch];

    int ld_weight = k;
    int ld_input = k;

    uint sg_i = sub_group_broadcast(get_local_id(0)/SUBGROUP_SIZE, 0);
    uint sg_j = sub_group_broadcast(get_local_id(1), 0);

    // start points of this sg
    uint wg_i0 = get_group_id(0) * ugemm_moe_wg_tile_m;
    uint wg_j0 = get_group_id(1) * ugemm_moe_wg_tile_n;
    uint sg_i0 = wg_i0 + sg_i * ugemm_moe_sg_tile_m;
    uint sg_j0 = wg_j0 + sg_j * ugemm_moe_sg_tile_n;

    if (wg_j0 >= n)
        return;     /* early exit if outside batch */
    // ugemm_moe_c_type ugemm_moe(const global half* a, int lda, const global half* b, int ldb, int m, int n, int k, int i0, int j0, int h0, int local_id_m, int local_id_n) {$
    ugemm_moe_c_type c_tile = ugemm_moe(weight_ptr, ld_weight, input_ptr, ld_input, m, n, k, wg_i0, wg_j0, 0, sg_i, sg_j);
    //printf("wg_i0 : %d wg_j0 : %d c_tile %f\n", wg_i0, wg_j0, c_tile.x[0][0]); // debug
    tile_store(c_tile, out_ptr, m, n, sg_i0, sg_j0);     // note oneDNN version needs a leading dimension parameter
}