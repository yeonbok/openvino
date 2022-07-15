// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/utils.hpp"
#include <vector>

using namespace cldnn;
using namespace ::tests;


TEST(LRUCache_test, basic01)
{
    const size_t cap = 4;
    LRUCache<int, int> ca(cap * sizeof(int));

    std::vector<int> inputs = {1, 2, 3, 4, 2, 1, 5};
    std::vector<std::pair<int, int>> input_values;
    for (auto i :  inputs) {
        input_values.push_back(std::make_pair(i, i + 10));
    }

    std::vector<bool> expected_hitted = {false, false, false, false, true, true, false};
    for (size_t i = 0; i < input_values.size(); i++) {
        auto& in = input_values[i];
        int data = 0;
        bool hitted = true;
        std::tie(data, hitted) = ca.get(in.first, [in](){
            return LRUCache<int, int>::CacheEntry{in.second, sizeof(in.second)};
        });
        EXPECT_EQ(data, in.second);
        EXPECT_EQ(hitted, (bool)expected_hitted[i]);
    }

    EXPECT_EQ(cap, ca.count());

    std::vector<std::pair<int, int>> expected_value;
    for (size_t i = cap; i > 0; i--) {  // 5, 1, 2, 4
        int idx = input_values.size() - i;
        expected_value.push_back(input_values[idx]);
    }

    int idx = expected_value.size() - 1;
    for (auto key : ca.get_all_keys()) {
        EXPECT_EQ(key, expected_value[idx--].first);
    }
}

class LRUCacheTestData {
public:
    LRUCacheTestData(int a, int b, int c) : x(a), y(b), z(c) {
        key = "key_" + std::to_string(a) + "_" + std::to_string(b) + "_" + std::to_string(c);
    }

    bool operator==(const LRUCacheTestData&rhs) {
        return (this->x == rhs.x && this->y == rhs.y && this->z == rhs.z);
    }

    bool operator!=(const LRUCacheTestData&rhs) {
        return (this->x != rhs.x || this->y != rhs.y || this->z != rhs.z);
    }

    operator std::string() {
        return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
    }

    std::string key;
    int x;
    int y;
    int z;

};

TEST(LRUCache_test, basic02) {
    const size_t cap = 4;
    LRUCache<std::string, std::shared_ptr<LRUCacheTestData>> ca(4 * sizeof(LRUCacheTestData));

    std::vector<std::shared_ptr<LRUCacheTestData>> inputs;
    inputs.push_back(std::make_shared<LRUCacheTestData>(1, 21, 11));
    inputs.push_back(std::make_shared<LRUCacheTestData>(2, 22, 12));
    inputs.push_back(std::make_shared<LRUCacheTestData>(3, 23, 13));
    inputs.push_back(std::make_shared<LRUCacheTestData>(4, 24, 14));
    inputs.push_back(std::make_shared<LRUCacheTestData>(2, 22, 12));
    inputs.push_back(std::make_shared<LRUCacheTestData>(1, 21, 11));
    inputs.push_back(std::make_shared<LRUCacheTestData>(3, 23, 13));
    inputs.push_back(std::make_shared<LRUCacheTestData>(5, 25, 15));

    std::vector<bool> expected_hitted = {false, false, false, false, true, true, true, false};

    for (size_t i = 0; i < inputs.size(); i++) {
        auto& in = inputs[i];
        std::shared_ptr<LRUCacheTestData> p_data;
        bool hitted = true;
        std::tie(p_data, hitted) = ca.get(in->key, [in](){
            return LRUCache<std::string, std::shared_ptr<LRUCacheTestData>>::CacheEntry{in, sizeof(LRUCacheTestData)};
        });
        EXPECT_EQ(p_data->key, in->key);
        EXPECT_EQ(hitted, (bool)expected_hitted[i]);
    }

    EXPECT_EQ(cap, ca.count());

    std::vector<std::string> expected_keys;
    for (size_t i = cap; i > 0; i--) {
        expected_keys.push_back(inputs[inputs.size() - i]->key);
    }

    int idx = expected_keys.size() - 1;
    for (auto key : ca.get_all_keys()) {
        EXPECT_EQ(key, expected_keys[idx--]);
    }
}
