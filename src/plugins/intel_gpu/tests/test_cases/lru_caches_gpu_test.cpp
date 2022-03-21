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
    for (int i = 0; i < input_values.size(); i++) {
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

    for (int i = 0; i < inputs.size(); i++) {
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

TEST(LRUCache_test, capacity_check)
{
    const std::vector<std::string> inputs = {
        "LRUCache_test",
        "[short cache 1]",
        "[short cache 2]",
        "[short cache 3]",
        "LRUCache_test",
        "last string",
        "Long data should delete short cache 1 2 3",
        "last string"};

    const std::vector<std::string> expected_final_cache = {
        "last string",
        "Long data should delete short cache 1 2 3",
        "LRUCache_test"};

    const size_t capacity = inputs[1].size() + 
                            inputs[2].size() + 
                            inputs[3].size() + 
                            inputs[4].size() +
                            inputs[5].size();

    LRUCache<std::string, std::string> cache(capacity);

    // check hitted result
    std::vector<bool> expected_hitted = {false, false, false, false, true, false, false, true};
    for (int i = 0; i < inputs.size(); i++) {
        auto& in = inputs[i];
        
        std::string data = "";
        bool hitted = false;
        std::tie(data, hitted) = cache.get(in, [in](){
            return LRUCache<std::string, std::string>::CacheEntry{in, in.size()};
        });
        EXPECT_EQ(hitted, (bool)expected_hitted[i]);
    }

    // compare final cache datas
    auto final_cache = cache.get_all_keys();
    EXPECT_EQ(final_cache.size(), expected_final_cache.size());

    int32_t i = 0;
    for (auto& it : final_cache) {
        EXPECT_EQ(it, expected_final_cache[i]);
        i++;
    }
}
