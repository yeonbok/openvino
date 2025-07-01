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
    if (!is_prefill_stage) {
        if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
            printf("2nd token\n");
        }

        // 2nd+ token
        const uint seq_idx = (uint)get_global_id(0);
        const uint head_idx = (uint)get_global_id(1);
        const uint sglid = (uint)get_global_id(2);

        const uint seq_len = past_lens[seq_idx];
        const uint current_token_pos_in_block = seq_len % PAGED_ATTENTION_BLOCK_SIZE;
        const uint seq_block_idx = block_indices_begins[seq_idx] + seq_len / PAGED_ATTENTION_BLOCK_SIZE;
        const uint block_idx = block_indices[seq_block_idx];

        uint key_in_offset = INPUT0_OFFSET +
                             seq_idx * (KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM) +
                             head_idx * K_HEAD_SIZE;
        uint value_in_offset = INPUT1_OFFSET +
                               seq_idx * (KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_BEFORE_FEATURE_NUM + INPUT1_PAD_AFTER_FEATURE_NUM) +
                               head_idx * V_HEAD_SIZE;

        uint block_k_base_offset = block_idx * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE + head_idx * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
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
        const uint comp_k_offset = block_k_base_offset + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint comp_v_offset = block_v_base_offset + V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        {
            // key processing
            INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
            FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE, comp_k_offset,
                current_token_pos_in_block, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
        }
        {
            // value processing
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
        const uint token_start_pos = (past_len + block_start_pos - subsequence_begin_idx) % ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE;

        const uint KEY_IN_STRIDE = KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM;
        uint key_in_offset = INPUT0_OFFSET + block_start_pos * KEY_IN_STRIDE + head_idx * K_HEAD_SIZE;

        const int VAL_IN_STRIDE = KV_HEADS_NUM * V_HEAD_SIZE + INPUT1_PAD_AFTER_FEATURE_NUM + INPUT1_PAD_BEFORE_FEATURE_NUM;
        uint value_in_offset = INPUT1_OFFSET + block_start_pos * VAL_IN_STRIDE + head_idx * V_HEAD_SIZE;

        const uint current_block_idx = (past_len + block_start_pos - subsequence_begin_idx) / PAGED_ATTENTION_BLOCK_SIZE;

        const uint block_offset = block_indices_begins[subsequence_idx] + current_block_idx;


        #if defined(IS_KV_COMPRESSED) && defined(IS_KEY_BY_CHANNEL)
        uint block_k_base_offset = block_indices[block_offset] * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                 head_idx * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint key_out_offset = block_k_base_offset;
//        const uint comp_k_offset = block_k_base_offset + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
 //       key_out_offset += token_start_pos;
        #else
        uint block_k_base_offset = block_indices[block_offset] * KV_HEADS_NUM * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                 head_idx * ADJUSTED_K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint key_out_offset = block_k_base_offset;
        const uint comp_k_offset = block_k_base_offset + K_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        key_out_offset += token_start_pos;
        #endif

        uint block_v_base_offset = block_indices[block_offset] * KV_HEADS_NUM * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                                 head_idx * ADJUSTED_V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        const uint comp_v_offset = block_v_base_offset + V_HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE;
        uint value_out_offset = block_v_base_offset;
        value_out_offset += token_start_pos * V_HEAD_SIZE;

        if (tokens_num == PAGED_ATTENTION_BLOCK_SIZE) {

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
                unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
                    OUTPUT_TYPE res = convert_char_rte(input_data[token_num] * scale + zp);
                    key_cache_data[key_out_offset_per_wi] = res;
                    key_out_offset_per_wi++;
                }
                // store comp_data
                INPUT0_TYPE* comp_ptr = key_cache_data + key_out_offset_per_wi;
                comp_ptr[0] = 1.0 / scale;
                comp_ptr[1] = zp;
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
                    uint key_in_offset_tmp = key_in_offset + (KV_HEADS_NUM * K_HEAD_SIZE + INPUT0_PAD_AFTER_FEATURE_NUM + INPUT0_PAD_BEFORE_FEATURE_NUM);
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
                unroll_for (uint token_num = 0; token_num < tokens_num; token_num++) {
                    OUTPUT_TYPE res = convert_char_rte(input_data[token_num] * scale + zp);
                    key_cache_data[key_out_offset_per_wi] = res;
                    key_out_offset_per_wi++;
                }
                // store comp_data
                INPUT0_TYPE* comp_ptr = key_cache_data + key_out_offset_per_wi;
                comp_ptr[0] = 1.0 / scale;
                comp_ptr[1] = zp;
                key_in_offset += SUBGROUP_SIZE;
                key_out_offset += ADJUSTED_PAGED_ATTENTION_BLOCK_SIZE * SUBGROUP_SIZE;
            }
            // value processing per token
            for (uint token_num = 0; token_num < tokens_num; token_num++) {
                INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                    comp_v_offset, token_start_pos + token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
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
                    if (get_global_id(0) == 1 && get_global_id(1) == 0 && get_global_id(2) == 4) {
                        printf("gid:%d,%d,%d current_block_idx: %d, block_offset : %d, subsequence_idx : %d, subsequence_begin_idx : %d, block_start_pos : %d, \
                        block_end_pos = %d,  token_start_pos = %d, token_num (sub): %d, cur_token: %d key_in_offset : %d, group_num : %d\n", get_global_id(0), get_global_id(1), \
                        get_global_id(2), current_block_idx, block_offset, subsequence_idx, subsequence_begin_idx, block_start_pos, block_end_pos, token_start_pos, tokens_num, token_num, key_in_offset, K_HEAD_SIZE/SUBGROUP_SIZE);
                    }
                    INPUT0_TYPE input_data[K_HEAD_SIZE / SUBGROUP_SIZE];
                    FUNC_CALL(quantize_and_save_per_token)(key_data, key_in_offset, key_cache_data, key_out_offset, PAGED_ATTENTION_BLOCK_SIZE,
                        comp_k_offset, token_start_pos + token_num, sglid, K_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
                }
                {
                    // value processing
                    INPUT0_TYPE input_data[V_HEAD_SIZE / SUBGROUP_SIZE];
                    FUNC_CALL(quantize_and_save_per_token)(value_data, value_in_offset, value_cache_data, value_out_offset, 1,
                        comp_v_offset, token_start_pos + token_num, sglid, V_HEAD_SIZE / SUBGROUP_SIZE, &input_data[0]);
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
