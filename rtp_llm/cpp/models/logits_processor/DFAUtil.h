#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cassert>
#include <cstring>
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

namespace rtp_llm {

template<typename StatusType, typename InputType>
class BaseDFA {
public:
    BaseDFA()                                = default;
    virtual ~BaseDFA()                       = default;
    virtual bool       isFinished()          = 0;
    virtual StatusType next(InputType input) = 0;
};

template<typename StatusType, typename InputType>
class StringContainDFA: public BaseDFA<StatusType, InputType> {
public:
    bool isFinished() override {
        std::stringstream ss;
        ss << "Generic ContainDFA isFinished()\n";
        RTP_LLM_LOG_INFO("%s", ss.str().c_str());
        return false;
    }
    StatusType next(InputType input) override {
        std::stringstream ss;
        ss << "Generic ContainDFA next()\n";
        RTP_LLM_LOG_INFO("%s", ss.str().c_str());
        return StatusType();
    }
};

template<typename InputType>
class StringContainDFA<size_t, InputType>: public BaseDFA<size_t, InputType> {
public:
    StringContainDFA(): contain_string_(), status_(0) {}

    StringContainDFA(std::vector<InputType> input_list): contain_string_(input_list), status_(0) {
        computeNextArray(contain_string_);
    }

    StringContainDFA(StringContainDFA&)  = default;
    StringContainDFA(StringContainDFA&&) = default;

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

    size_t next(InputType input) override {  // find next available suffix by kmp
        if (isFinished()) {
            return status_;
        }
        while (status_ > 0 && contain_string_[status_] != input) {
            status_ = next_array_[status_ - 1];
        }
        if (contain_string_[status_] == input) {
            ++status_;
        }
        return status_;
    }

private:
    void computeNextArray(const std::vector<InputType>& input_list) {
        size_t n = input_list.size();
        next_array_.resize(n, 0);
        size_t j = 0;

        for (size_t i = 1; i < n; ++i) {
            while (j > 0 && input_list[i] != input_list[j]) {
                j = next_array_[j - 1];
            }
            if (input_list[i] == input_list[j]) {
                ++j;
            }
            next_array_[i] = j;
        }
    }

private:
    std::vector<InputType> contain_string_;
    size_t                 status_;
    std::vector<size_t>    next_array_;
};

template<typename StatusType, typename InputType>
class TreeDFA: public BaseDFA<StatusType, InputType> {
public:
    bool isFinished() override {
        std::stringstream ss;
        ss << "Generic TreeDFA isFinished()\n";
        RTP_LLM_LOG_INFO("%s", ss.str().c_str());
        return false;
    }
    StatusType next(InputType input) override {
        std::stringstream ss;
        ss << "Generic TreeDFA next()\n";
        RTP_LLM_LOG_INFO("%s", ss.str().c_str());
        return StatusType();
    }
};

template<typename InputType>
class TreeDFA<std::string, InputType>: public BaseDFA<std::string, InputType> {
public:
    TreeDFA(PrefixToCandidateTokensPtr prefixToCandidateTokensPtr):
        prefixToCandidateTokensPtr_(prefixToCandidateTokensPtr),
        status_(std::to_string(prefixToCandidateTokensPtr->startTokenId())),
        input_list_(10) {}

    TreeDFA(TreeDFA&)  = default;
    TreeDFA(TreeDFA&&) = default;

    bool isFinished() override {
        if (input_list_.empty()) {
            return false;
        }
        return input_list_[input_list_.size() - 1] == prefixToCandidateTokensPtr_->endTokenId();
    }

    std::string status() {
        return status_;
    }

    void forceSetStatus(std::string status) {
        status_ = status;
    }

    std::string next(InputType input) override {
        if (isFinished()) {
            return status_;
        }
        std::string new_status = prefixToCandidateTokensPtr_->generateNextKey(status_, input);
        if (prefixToCandidateTokensPtr_->isValidStatus(new_status)
            || std::to_string(input) == std::to_string(prefixToCandidateTokensPtr_->endTokenId())) {
            input_list_.push_back(input);
            status_ = new_status;
        } else {
            std::stringstream ss;
            ss << "Generated invalid status, status[" << status_ << "], input_id[" << input << "]";
            RTP_LLM_LOG_ERROR("%s, return status[%s].", ss.str().c_str(), status_.c_str());
            input_list_.clear();
        }
        return status_;
    }

    std::vector<size_t> getCandidateTokenIds() {
        std::vector<size_t> token_ids;
        for (auto token_id : prefixToCandidateTokensPtr_->getCandidateTokens(status_)) {
            if (token_id >= 0) {
                token_ids.push_back(static_cast<size_t>(token_id));
            } else {
                throw std::out_of_range("Negative token ID encountered");
            }
        }
        if (token_ids.empty()) {
            token_ids.push_back(prefixToCandidateTokensPtr_->endTokenId());
        }
        return token_ids;
    }

private:
    PrefixToCandidateTokensPtr prefixToCandidateTokensPtr_;
    std::string                status_;
    std::vector<InputType>     input_list_;
};

}  // namespace rtp_llm
