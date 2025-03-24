#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>

namespace rtp_llm {

template<typename StatusType, typename InputType>
class BaseDFA {
public:
    BaseDFA() = default;
    virtual ~BaseDFA() = default;
    virtual bool isFinished() = 0;
    virtual StatusType next(InputType input) = 0;
};

template<typename StatusType, typename InputType>
class StringContainDFA : public BaseDFA<StatusType, InputType> {
public:
    bool isFinished() override {
        std::cout << "Generic ContainDFA isFinished()\n";
        return false;
    }
    StatusType next(InputType input) override {
        std::cout << "Generic ContainDFA next()\n";
        return StatusType();
    }
};

template<typename InputType>
class StringContainDFA<size_t, InputType> : public BaseDFA<size_t, InputType> {
public:
    StringContainDFA(): contain_string_(), status_(0) {}

    StringContainDFA(std::vector<InputType> input_list)
        : contain_string_(input_list), status_(0) {
        computeNextArray(contain_string_);
    }

    void compile(const std::vector<InputType>& input_list) {
        contain_string_ = input_list;
        computeNextArray(input_list);
    }

    bool isFinished() override {
        return status_ == contain_string_.size();
    }

    size_t status() {
        return status_;
    }

    void forceSetStatus(size_t status) {
        status_ = status;
    }

    size_t next(InputType input) override { // find next available suffix by kmp
        while(status_ > 0 && contain_string_[status_] != input) {
            status_ = next_array_[status_ - 1];
        }
        if(contain_string_[status_] == input) {
            ++status_;
        }
        return status_;
    }

private:
    void computeNextArray(const std::vector<InputType>& input_list) {
        size_t n = input_list.size();
        next_array_.resize(n, 0);
        size_t j = 0;

        for(size_t i = 1; i < n; ++i) {
            while(j > 0 && input_list[i] != input_list[j]) {
                j = next_array_[j - 1];
            }
            if(input_list[i] == input_list[j]) {
                ++j;
            }
            next_array_[i] = j;
        }
    }

private:
    std::vector<InputType> contain_string_;
    size_t status_;
    std::vector<size_t> next_array_;
};

template<typename InputType>
void dfaStatusForward(StringContainDFA<size_t, InputType>& dfa, ft::Buffer new_tokens, int num_new_tokens, 
    std::vector<int> template_token_ids, bool enforce) 
{
    size_t original_status = dfa.status();
    for (size_t j = 0; j < num_new_tokens; ++j) {
        auto current_token_id = *new_tokens.dataWithOffset<int>(j);
        if (!dfa.isFinished()) {
            dfa.next(current_token_id);
        }
    }
    if (!dfa.isFinished() && enforce) {
        int offset = 0;
        for (size_t pos = original_status; pos < template_token_ids.size() && offset < num_new_tokens; pos++, offset++) {
            *new_tokens.dataWithOffset<int>(offset) = template_token_ids[pos];
            dfa.forceSetStatus(pos + 1);
        }
    }
}

}