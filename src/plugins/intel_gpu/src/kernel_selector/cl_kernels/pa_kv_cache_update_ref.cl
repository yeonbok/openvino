// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

inline void FUNC(quantize_and_save_per_token)(__global const INPUT0_TYPE* in_data,
                                    const uint in_data_offset,
                                    __global OUTPUT_TYPE* out_data,
                                    const uint out_data_offset,
                                    const uint out_data_pitch,
                                    const uint comp_offset,
                                    const uint token_pos_in_block,
                                    const uint sglid,
                                    const uint num_groups,
                                    INPUT0_TYPE* input_data) {
    INPUT0_TYPE grp_max = 0.001;
    INPUT0_TYPE max_value = INPUT0_VAL_MIN;
    INPUT0_TYPE min_value = INPUT0_VAL_MAX;
    unroll_for (uint i = 0; i < num_groups; i++) {
        input_data[i] = BLOCK_READN(INPUT0_TYPE, 1, in_data, in_data_offset + i * SUBGROUP_SIZE);
        max_value = fmax(max_value, input_data[i]);
        min_value = fmin(min_value, input_data[i]);
    }

    min_value = sub_group_reduce_min(min_value);
    max_value = sub_group_reduce_max(max_value);

    // If the range of input data is zero, it is adjusted to the minimum value(0.001).
    #define ACCUMULATOR_TYPE float
    ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
    ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
    ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
    INPUT0_TYPE scale = (INPUT1_TYPE)(scale_tmp);
    INPUT0_TYPE zp = (INPUT1_TYPE)(zp_tmp);
    #undef ACCUMULATOR_TYPE

    unroll_for (uint i = 0; i < num_groups; i++) {
        OUTPUT_TYPE res = convert_char_rte(input_data[i] * scale + zp);

        uint offset = out_data_offset + (i * SUBGROUP_SIZE + sglid) * out_data_pitch;
        out_data[offset] = res;
    }

    INPUT0_TYPE* comp_ptr = out_data + comp_offset;

    if (sglid == 0) {
        comp_ptr[token_pos_in_block] = 1.0 / scale;
        comp_ptr[PAGED_ATTENTION_BLOCK_SIZE + token_pos_in_block] = zp;
    }
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUBGROUP_SIZE)))
KERNEL(pa_kv_cache_update)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* key_data,
    __global const INPUT1_TYPE* value_data,
    __global const INPUT2_TYPE* past_lens,
    __global const INPUT3_TYPE* block_indices,
    __global const INPUT4_TYPE* block_indices_begins,
    __global const INPUT5_TYPE* subsequence_begins,
    __global OUTPUT_TYPE* key_cache_data,
    __global OUTPUT1_TYPE* value_cache_data,
    const __global int* blocked_indexes_start,
    const __global int* blocked_indexes_end,
    const __global int* gws_seq_indexes_correspondence,
    const int is_prefill_stage
) {
    // If the the number of new tokens equals to the number of past_lens elements,
    // then it's the 2nd+ iteration
    const uint KEY_IN_STRIDE = KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM;
    const uint VAL_IN_STRIDE = KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM;
    if (!is_prefill_stage) {
        // 2nd+ token
        const uint seq_idx = (uint)get_global_id(0);
        const uint head_idx = (uint)get_global_id(1);
        const uint sglid = (uint)get_global_id(2);

        const uint past_seq_len = past_lens[seq_idx];
        const uint current_token_pos_in_block = past_seq_len % PAGED_ATTENTION_BLOCK_SIZE;
        const uint seq_block_idx = block_indices_begins[seq_idx] + past_seq_len / PAGED_ATTENTION_BLOCK_SIZE;
        const uint block_idx = block_indices[seq_block_idx];
        // printf("seq_idx : %d, past_seq_len : %d, current_token_pos_in_block %d\n", seq_idx, past_seq_len, current_token_pos_in_block);
        uint key_in_offset = INPUT0_OFFSET + seq_idx * KEY_IN_STRIDE + head_idx * K_HEAD_SIZE;
        uint value_in_offset = INPUT1_OFFSET + seq_idx * VAL_IN_STRIDE + head_idx * V_HEAD_SIZE;

        #ifdef IS_KEY_BY_CHANNEL
        uint block_k_base_offset = block_idx * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE + head_idx * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        #else // can it be shared?
        uint block_k_base_offset = block_idx * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + head_idx * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        #endif
        uint block_v_base_offset = block_idx * KV_HEADS_NUM * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + head_idx * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint key_out_offset = block_k_base_offset + current_token_pos_in_block;
        uint value_out_offset = block_v_base_offset + current_token_pos_in_block * V_HEAD_SIZE;


#if !IS_KV_COMPRESSED
        #define READ_K_BLOCK_SIZE GENERATE_STAGE_K_BLOCK_SIZE
        for (uint head_idx_index = 0; head_idx_index < K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_K_BLOCK_SIZE) {
            #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_K_BLOCK_SIZE, ptr, offset);
            #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_K_BLOCK_SIZE)

            DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

            unroll_for (uint i = 0; i < READ_K_BLOCK_SIZE; i++) {
                uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                #if READ_K_BLOCK_SIZE == 1
                    key_cache_data[key_offset] = input_data;
                #else
                    key_cache_data[key_offset] = input_data[i];
                #endif
            }
        }

        #define READ_V_BLOCK_SIZE GENERATE_STAGE_V_BLOCK_SIZE
        for (uint head_idx_index = 0; head_idx_index < V_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_V_BLOCK_SIZE) {
            #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_V_BLOCK_SIZE, ptr, offset);
            #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_V_BLOCK_SIZE)

            DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + head_idx_index);

            unroll_for (uint i = 0; i < READ_V_BLOCK_SIZE; i++) {
                uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                #if READ_V_BLOCK_SIZE == 1
                    value_cache_data[value_offset] = input_data;
                #else
                    value_cache_data[value_offset] = input_data[i];
                #endif
            }
        }
#else // IS_KV_COMPRESSED
        #ifdef IS_KEY_BY_CHANNEL
        const int hidden_stride = ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        const int comp_offset = PAGED_ATTENTION_BLOCK_SIZE;
        for (int i = 0; i < K_HEAD_SIZE / SUBGROUP_SIZE; i++) {
            // read k input
            const int hidden_idx = i * SUBGROUP_SIZE + sglid;
            const uint key_out_offset_per_wi = block_k_base_offset + hidden_idx * hidden_stride;
            const uint comp_k_offset = key_out_offset_per_wi + comp_offset;
            // read scale and zp
            INPUT0_TYPE* comp_ptr = key_cache_data + comp_k_offset;
            const INPUT0_TYPE orig_scale = comp_ptr[0];
            const INPUT0_TYPE orig_zp = comp_ptr[1];
            INPUT0_TYPE new_token = BLOCK_READN(INPUT0_TYPE, 1, key_data, key_in_offset + i * SUBGROUP_SIZE);
            // read key cache in this block and decompress
            // TODO : current block size is 16, but when the block size becomes different, this should be updated as well
            char16 key_cache_data_vec = vload16(0, key_cache_data + key_out_offset_per_wi);
//            if (get_global_id(0) == 0 && get_global_id(1) == 0)
//                printf("gid: %d %d %d sglid : %d, hidden_idx : %d, key_in_offset : %d, key_cache_data[%d], orig_scale : %f, orig_zp : %f\n", get_global_id(0), get_global_id(1), get_global_id(2), sglid, hidden_idx, key_in_offset, comp_k_offset, orig_scale, orig_zp);
            half16 key_cache_data_vec_half16;
            INPUT0_TYPE max_value = new_token;
            INPUT0_TYPE min_value = new_token;
            for (int j = 0; j < current_token_pos_in_block; ++j) {
                INPUT0_TYPE decompressed_key_cache_val = ((INPUT0_TYPE)key_cache_data_vec[j] - orig_zp) * orig_scale;
                key_cache_data_vec_half16[j] = decompressed_key_cache_val;
                max_value = fmax(max_value, decompressed_key_cache_val);
                min_value = fmin(min_value, decompressed_key_cache_val);
            }

            // requantize and store
            {
                #define ACCUMULATOR_TYPE float
                INPUT0_TYPE grp_max = 0.001;
                ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
                ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
                ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
                INPUT0_TYPE scale = (INPUT1_TYPE)(scale_tmp);
                INPUT0_TYPE zp = (INPUT1_TYPE)(zp_tmp);
                #undef ACCUMULATOR_TYPE
                for (uint token = 0; token <= current_token_pos_in_block; ++token) {
                    OUTPUT_TYPE quantized_key = convert_char_rte(key_cache_data_vec_half16[token] * scale + zp);
                    key_cache_data[key_out_offset_per_wi + token] = quantized_key;
                }
                comp_ptr[0] = 1.0/scale;
                comp_ptr[1] = zp;
//                printf("requantize result) gid %d %d %d head %d hidden %d cur_token_pos : %d key_out_offset: %d comp_k_offset : %d original scale : %f orig zp %f => req_scale = %f req_zp = %f\n", \
//                    get_global_id(0), get_global_id(1), get_global_id(2), get_global_id(1), hidden_idx, current_token_pos_in_block, key_out_offset_per_wi, comp_k_offset, orig_scale, orig_zp, comp_ptr[0], comp_ptr[1]);
            }
        } 

        #else
        {
            const uint comp_k_offset = block_k_base_offset + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
            // key processing
            INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
            FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE, comp_k_offset,
                current_token_pos_in_block, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
        }
        #endif
        // value processing
        {
            const uint comp_v_offset = block_v_base_offset + V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
            INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
            FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1, comp_v_offset,
                current_token_pos_in_block, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
        }
#endif // IS_KV_COMPRESSED
    } else {
        // 1st token
        const uint block_idx = get_global_id(0);
        const uint head_idx = get_global_id(1);
        const uint sglid = get_global_id(2);

    
        const uint subsequence_idx = gws_seq_indexes_correspondence[block_idx];
        const uint subsequence_begin_idx = subsequence_begins[subsequence_idx];

        const uint block_start_pos = blocked_indexes_start[block_idx];
        const uint block_end_pos = blocked_indexes_end[block_idx];
        const uint tokens_num = block_end_pos - block_start_pos;
        const uint past_len = past_lens[subsequence_idx];
        const uint token_start_pos_key = (past_len + block_start_pos - subsequence_begin_idx) % ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        const uint token_start_pos_val = (past_len + block_start_pos - subsequence_begin_idx) % PAGED_ATTENTION_BLOCK_SIZE;

        uint key_in_offset = INPUT0_OFFSET + block_start_pos * KEY_IN_STRIDE + head_idx * K_HEAD_SIZE;

        uint value_in_offset = INPUT1_OFFSET + block_start_pos * VAL_IN_STRIDE + head_idx * V_HEAD_SIZE;

        const uint current_block_idx = (past_len + block_start_pos - subsequence_begin_idx) / PAGED_ATTENTION_BLOCK_SIZE;

        const uint block_offset = block_indices_begins[subsequence_idx] + current_block_idx;


        #if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
        uint block_k_base_offset = block_indices[block_offset] * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE +
                                 head_idx * ADJUSTED_K_HEAD_SIZE * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
        uint key_out_offset = block_k_base_offset;
        #else
        uint block_k_base_offset = block_indices[block_offset] * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                 head_idx * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint key_out_offset = block_k_base_offset;
        const uint comp_k_offset = block_k_base_offset + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        key_out_offset += token_start_pos_key;
        #endif

        uint block_v_base_offset = block_indices[block_offset] * KV_HEADS_NUM * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                 head_idx * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint comp_v_offset = block_v_base_offset + V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint value_out_offset = block_v_base_offset;
        value_out_offset += token_start_pos_val * V_HEAD_SIZE;

        if (tokens_num == PAGED_ATTENTION_BLOCK_SIZE) {
        // block is full
        #if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            // key by channel
            for (uint i = 0; i < K_HEAD_SIZE / SUBGROUP_SIZE; i++) {
                uint key_in_offset_tmp = key_in_offset + i * SUBGROUP_SIZE;
                INPUT0_TYPE input_data[PAGED_ATTENTION_BLOCK_SIZE]; // to make this as local memory
                INPUT0_TYPE max_value = INPUT0_VAL_MIN;
                INPUT0_TYPE min_value = INPUT0_VAL_MAX;
                // Read 16 tokens x 16 hidden
                unroll_for (uint token_num = 0; token_num < PAGED_ATTENTION_BLOCK_SIZE; token_num++) {
                    input_data[token_num] = BLOCK_READN(INPUT0_TYPE, 1, key_data, key_in_offset_tmp + sglid);
                    max_value = fmax(max_value, input_data[token_num]);
                    min_value = fmin(min_value, input_data[token_num]);
                    key_in_offset_tmp += KEY_IN_STRIDE;
                }
                #define ACCUMULATOR_TYPE float
                INPUT0_TYPE grp_max = 0.001;
                ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
                ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
                ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
                INPUT0_TYPE scale = (INPUT1_TYPE)(scale_tmp);
                INPUT0_TYPE zp = (INPUT1_TYPE)(zp_tmp);
                #undef ACCUMULATOR_TYPE
                // quantize and save each hidden dim
                // TODO to store as vector write 
                uint key_out_offset_per_wi = key_out_offset + sglid * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
                const uint comp_k_offset = key_out_offset_per_wi + PAGED_ATTENTION_BLOCK_SIZE;
                // store comp_data
                INPUT0_TYPE* comp_ptr = (INPUT0_TYPE*) (&key_cache_data[comp_k_offset]);
                comp_ptr[0] = 1.0 / scale;
                comp_ptr[1] = zp;
//                if (get_global_id(1) == 0)
//                    printf("wrote scale to key_cache_data[%d] diff : %f scale_tmp : %f scale = %f %f, zp = %f\n", comp_k_offset, diff_value, scale_tmp, scale, comp_ptr[0], comp_ptr[1]);

                // store quantized key
                unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
                    OUTPUT_TYPE res = convert_char_rte(input_data[token_num] * scale + zp);
                    key_cache_data[key_out_offset_per_wi] = res;
                    key_out_offset_per_wi++;
                }
                key_in_offset += SUBGROUP_SIZE;
                key_out_offset += ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE * SUBGROUP_SIZE;
            }
            // value per token
            unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
                INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                    comp_v_offset, token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += V_HEAD_SIZE;
            }
        #else // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            unroll_for (uint token_num = 0; token_num < PAGED_ATTENTION_BLOCK_SIZE; token_num++) {
            #if !IS_KV_COMPRESSED
            {
                uint head_idx_index = 0;

                #define READ_BLOCK_SIZE 8
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 4
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 2
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 1
                for (; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data;
                    }
                }
            }
            {

                uint v_head_idx_index = 0;

                #define READ_BLOCK_SIZE 8
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 4
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }

                #define READ_BLOCK_SIZE 2
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data[i];
                    }
                }


                #define READ_BLOCK_SIZE 1
                for (; v_head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; v_head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                    #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)

                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + v_head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + v_head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data;
                    }
                }
            }
            #else // IS_KV_COMPRESSED
            {
                // key processing
                INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE,
                    comp_k_offset, token_num, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
            }
            {
                // value processing
                INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                    comp_v_offset, token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
            }
            #endif // IS_KV_COMPRESSED
                key_in_offset += (KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM);
                key_out_offset += 1;
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += V_HEAD_SIZE;
            }
        #endif // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
        } else {
        // block with leftover tokens
        #if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            // key processing by channel
            for (uint i = 0; i < K_HEAD_SIZE / SUBGROUP_SIZE; i++) {
                uint key_in_offset_tmp = key_in_offset + i * SUBGROUP_SIZE;
                INPUT0_TYPE input_data[PAGED_ATTENTION_BLOCK_SIZE]; // to make this as local memory
                INPUT0_TYPE max_value = INPUT0_VAL_MIN;
                INPUT0_TYPE min_value = INPUT0_VAL_MAX;
                // Read num_tokens x 16 hidden
                unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
                    input_data[token_num] = BLOCK_READN(INPUT0_TYPE, 1, key_data, key_in_offset_tmp + sglid);
                    max_value = fmax(max_value, input_data[token_num]);
                    min_value = fmin(min_value, input_data[token_num]);
                    key_in_offset_tmp += KEY_IN_STRIDE;
                }
                #define ACCUMULATOR_TYPE float
                INPUT0_TYPE grp_max = 0.001;
                ACCUMULATOR_TYPE diff_value = max_value == min_value ? (grp_max) : (max_value - min_value);
                ACCUMULATOR_TYPE scale_tmp = (ACCUMULATOR_TYPE)((CHAR_MAX - CHAR_MIN) / diff_value);
                ACCUMULATOR_TYPE zp_tmp = (ACCUMULATOR_TYPE)(-min_value * scale_tmp) + CHAR_MIN;
                INPUT0_TYPE scale = (INPUT1_TYPE)(scale_tmp);
                INPUT0_TYPE zp = (INPUT1_TYPE)(zp_tmp);
                #undef ACCUMULATOR_TYPE
                // quantize and save each hidden dim
                // TODO to store as vector write 
                const uint key_out_offset_per_wi = key_out_offset + sglid * ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;
                const uint comp_k_offset = key_out_offset_per_wi + PAGED_ATTENTION_BLOCK_SIZE;
                // store comp_data
                // store comp_data
                INPUT0_TYPE* comp_ptr = (INPUT0_TYPE*) (&key_cache_data[comp_k_offset]);
                comp_ptr[0] = 1.0 / scale;
                comp_ptr[1] = zp;
//                if (get_global_id(1) == 0)
//                    printf("gid: %d %d %d) wrote scale to key_cache_data[%d] block_k_base_offset : %d key_out_offset : %d diff : %f scale_tmp : %f scale = %f %f, zp = %f min_val :%f max_Val : %f diff : %f CHAR_MIN: %d\n", \
//                        get_global_id(0), get_global_id(1), get_global_id(2), \
//                        comp_k_offset, block_k_base_offset, key_out_offset, diff_value, scale_tmp, scale, comp_ptr[0], comp_ptr[1], min_value, max_value, diff_value,  CHAR_MIN);

                unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
                    OUTPUT_TYPE res = convert_char_rte(input_data[token_num] * scale + zp);
                    key_cache_data[key_out_offset_per_wi + token_num] = res;
                }
                key_in_offset += SUBGROUP_SIZE;
                key_out_offset += ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE * SUBGROUP_SIZE;
            }
            // value processing per token
            for (uint token_num = 0; token_num < tokens_num; token_num++) {
                INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
//                printf("Processing value: gid (%d %d %d) value_in_offset : %d value_out_offset : %d\n", get_global_id(0), get_global_id(1), get_global_id(2), value_in_offset, value_out_offset);
                FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                    comp_v_offset, token_start_pos_val + token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += V_HEAD_SIZE;
            }
        #else // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
            for (uint token_num = 0; token_num < tokens_num; token_num++) {
                uint head_idx_index = 0;

#if !IS_KV_COMPRESSED
                #define READ_BLOCK_SIZE 1
                #define BLOCK_READ(ptr, offset) BLOCK_READN(INPUT0_TYPE, READ_BLOCK_SIZE, ptr, offset);
                #define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_BLOCK_SIZE)
                for (uint head_idx_index = 0; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= K_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    DATA_VEC input_data = BLOCK_READ(key_data, key_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint key_offset = key_out_offset + (head_idx_index + sglid + SUBGROUP_SIZE * i) * PAGED_ATTENTION_BLOCK_SIZE;
                        key_cache_data[key_offset] = input_data;
                    }
                }

                for (uint head_idx_index = 0; head_idx_index + (READ_BLOCK_SIZE * SUBGROUP_SIZE) <= V_HEAD_SIZE; head_idx_index += SUBGROUP_SIZE * READ_BLOCK_SIZE) {
                    DATA_VEC input_data = BLOCK_READ(value_data, value_in_offset + head_idx_index);

                    unroll_for (uint i = 0; i < READ_BLOCK_SIZE; i++) {
                        uint value_offset = value_out_offset + head_idx_index + sglid + SUBGROUP_SIZE * i;
                        value_cache_data[value_offset] = input_data;
                    }
                }

#else // IS_KV_COMPRESSED
                {
                    // key processing
//                    if (get_global_id(0) == 1 && get_global_id(1) == 0 && get_global_id(2) == 4) {
//                        printf("gid:%d,%d,%d current_block_idx: %d, block_offset : %d, subsequence_idx : %d, subsequence_begin_idx : %d, block_start_pos : %d, \
//                        block_end_pos = %d,  token_start_pos = %d, token_num (sub): %d, cur_token: %d key_in_offset : %d, group_num : %d\n", get_global_id(0), get_global_id(1), \
//                        get_global_id(2), current_block_idx, block_offset, subsequence_idx, subsequence_begin_idx, block_start_pos, block_end_pos, token_start_pos_key, tokens_num, token_num, key_in_offset, K_HEAD_SIZE/SUBGROUP_SIZE);
//                    }
                    INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
                    FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE,
                        comp_k_offset, token_start_pos_key + token_num, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                }
                {
                    // value processing
                    INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                    FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                        comp_v_offset, token_start_pos_val + token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                }
#endif // IS_KV_COMPRESSED

                key_in_offset += (KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM);
                key_out_offset += 1;
                value_in_offset += (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM);
                value_out_offset += V_HEAD_SIZE;
            }
        #endif // defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
        }
    }
}
