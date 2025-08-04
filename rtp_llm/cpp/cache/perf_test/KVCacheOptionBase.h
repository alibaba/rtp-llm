#pragma once

#include "autil/OptionParser.h"

namespace rtp_llm {

class KVCacheOptionBase {
public:
    KVCacheOptionBase(): opt_parser_("usage placeholder") {
        opt_parser_.addOption("", "--read", "read", read);
        opt_parser_.addOption("", "--write", "write", write);
        opt_parser_.addOption("", "--model_name", "model_name", model_name);
        opt_parser_.addOption("", "--layer_num", "layer_num", layer_num);
        opt_parser_.addOption("", "--block_num", "block_num", block_num);
        opt_parser_.addOption("", "--local_head_num_kv", "local_head_num_kv", local_head_num_kv);
        opt_parser_.addOption("", "--size_per_head", "size_per_head", size_per_head);
        opt_parser_.addOption("", "--seq_size_per_block", "seq_size_per_block", seq_size_per_block);
        opt_parser_.addOption("", "--block_size", "block_size", block_size);
        opt_parser_.addOption("", "--block_stride", "block_stride", block_stride);
        opt_parser_.addOption("", "--data_type", "data_type", data_type);
        opt_parser_.addOption("", "--wait_mr_time_sec", "wait_mr_time_sec", wait_mr_time_sec);
    }

public:
    bool parseOptions(int argc, char* argv[]) {
        std::ostringstream ss;
        ss << "Usage:" << std::endl << "\t-h|--help show usage" << std::endl;
        opt_parser_.updateUsageDescription(ss.str());

        if (!opt_parser_.parseArgs(argc, argv)) {
            return false;
        }

        opt_parser_.getOptionValue("read", read);
        opt_parser_.getOptionValue("write", write);
        opt_parser_.getOptionValue("model_name", model_name);
        opt_parser_.getOptionValue("layer_num", layer_num);
        opt_parser_.getOptionValue("block_num", block_num);
        opt_parser_.getOptionValue("local_head_num_kv", local_head_num_kv);
        opt_parser_.getOptionValue("size_per_head", size_per_head);
        opt_parser_.getOptionValue("seq_size_per_block", seq_size_per_block);
        opt_parser_.getOptionValue("block_size", block_size);
        opt_parser_.getOptionValue("block_stride", block_stride);
        opt_parser_.getOptionValue("data_type", data_type);
        opt_parser_.getOptionValue("wait_mr_time_sec", wait_mr_time_sec);

        if (!read && !write) {
            RTP_LLM_LOG_ERROR("read or write option must be set!");
            return false;
        }
        return true;
    }

    void addInt32Option(const std::string& option_name, int default_value) {
        opt_parser_.addOption("", "--" + option_name, option_name, default_value);
    }

    void addStringOption(const std::string& option_name, const std::string& default_value) {
        opt_parser_.addOption("", "--" + option_name, option_name, default_value);
    }

    void addBoolOption(const std::string& option_name, bool default_value) {
        opt_parser_.addOption("", "--" + option_name, option_name, default_value);
    }

    template<typename T>
    bool getOptionValue(const std::string& option_name, T& option_value) {
        return opt_parser_.getOptionValue(option_name, option_value);
    }

    std::string toString() const {
        std::ostringstream ss;
        ss << "read: " << read << ", "
           << "write: " << write << ", "
           << "model_name: " << model_name << ", layer_num: " << layer_num << ", block_num: " << block_num
           << ", local_head_num_kv: " << local_head_num_kv << ", size_per_head: " << size_per_head
           << ", seq_size_per_block: " << seq_size_per_block << ", block_size: " << block_size
           << ", block_stride: " << block_stride << ", data_type: " << data_type
           << ", wait_mr_time_sec: " << wait_mr_time_sec;
        return ss.str();
    }

public:
    bool read{false};   // true for read test
    bool write{false};  // true for write test

    // kvcache related
    std::string model_name{"TestModel"};
    int         layer_num{80};
    int         block_num{2048};
    int         local_head_num_kv{2};
    int         size_per_head{128};
    int         seq_size_per_block{128};
    int         block_size{1 << 20};  // 1MB
    int         block_stride{65536};  // 64KB
    std::string data_type{"fp16"};    // int8/fp8/fp16/fp32/bf16
    int         wait_mr_time_sec{3};  // sleep time for iov mr, in second

private:
    autil::OptionParser opt_parser_;
};

}  // namespace rtp_llm