#pragma once
#include <algorithm>
#include <string>
#include <unordered_map>

template<typename T>
struct LoRAWeight {
    std::unordered_map<int, std::pair<T*, T*>> lora_map;
    std::unordered_map<int, int>               lora_ranks;
    int                                        max_rank = 0;

    std::pair<T*, T*> getLoRAWeight(int lora_id) const {
        auto it = lora_map.find(lora_id);
        if (it != lora_map.end()) {
            return it->second;
        }
        return std::make_pair<T*, T*>(nullptr, nullptr);
    }

    int getLoRARank(int lora_id) const {
        auto it = lora_ranks.find(lora_id);
        if (it != lora_ranks.end()) {
            return it->second;
        }
        return 0;
    }

    void setLoRAWeight(int lora_id, T* lora_a, T* lora_b, int lora_rank = 0) {
        lora_map[lora_id] = std::make_pair(lora_a, lora_b);
        if (lora_rank) {
            this->lora_ranks[lora_id] = lora_rank;
            if (lora_rank > max_rank) {
                max_rank = lora_rank;
            }
        }
    }
    void removeLoRAWeight(int lora_id) {
        if (lora_map.find(lora_id) == lora_map.end()) {
            return;
        }
        lora_map.erase(lora_id);
        int lora_rank = lora_ranks[lora_id];
        if (lora_rank == max_rank) {
            lora_ranks.erase(lora_id);
            if (lora_map.size() > 0) {
                auto it = std::max_element(
                    lora_ranks.begin(),
                    lora_ranks.end(),
                    [](const std::pair<int, int>& p1, const std::pair<int, int>& p2) { return p1.second < p2.second; });
                max_rank = it->second;
            } else {
                max_rank = 0;
            }
        }
    }
};