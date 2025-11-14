#pragma once
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <iostream>
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {
namespace fmha {
class FmhaProfParam {
public:
    FmhaProfParam() {}
    void initialize_args(const std::string& op_name, bool dir) {
        op_name_ = op_name;
        op_dir_  = dir ? "Forward" : "Backward";
        add_argument("data_type");   // string
        add_argument("batch_size");  // int
        add_argument("num_heads");
        add_argument("num_heads_k");
        add_argument("head_dim");
        add_argument("head_dim_value");
        add_argument("seqlen_q");
        add_argument("seqlen_k");
        add_argument("custom_mask");  // bool
        add_argument("dropout");      // float
        add_argument("scale");
        add_argument("is_fixed_seqs");  // bool
        // add_argument("alibi"); //bool
        // add_argument("window_size_left");
        // add_argument("window_size_right");
    }
    template<typename T>
    void add_mha_params(const std::string& key, const T& val) {
        if (args_.find(key) == args_.end()) {
            args_.insert(std::make_pair(key, val_to_string(val)));
        }
        args_.at(key) = val_to_string(val);
    }
    void set_flash_attn_params(bool  dir,
                               bool  is_bf16,
                               bool  is_causal,
                               int   batch_size,
                               int   num_heads,
                               int   num_heads_k,
                               int   head_dim,
                               int   head_dim_value,
                               int   seqlen_q,
                               int   seqlen_k,
                               float dropout,
                               float scale,
                               int   window_size_left,
                               int   window_size_right,
                               bool  is_fixed_seqs,
                               bool  alibi) {
        const std::string op_name = "flash_attn_v2.5.6";
        initialize_args(op_name, dir);
        std::string data_type = is_bf16 ? "bf16" : "fp16";
        if (std::abs(scale * scale * head_dim - 1) > 1e-3) {
            // in general, scale can be get form inner computation.
            add_mha_params("scale", scale);
        }
        add_mha_params("custom_mask", is_causal);
        if (dropout < 1.f) {
            add_mha_params("dropout", dropout);
        }
        add_mha_params("seqlen_q", seqlen_q);
        add_mha_params("is_fixed_seqs", is_fixed_seqs);
        add_mha_params("head_dim", head_dim);
        add_mha_params("head_dim_value", head_dim_value);
        add_mha_params("num_heads_k", num_heads_k);
        add_mha_params("num_heads", num_heads);
        add_mha_params("batch_size", batch_size);
        add_mha_params("seqlen_k", seqlen_k);
        add_mha_params("data_type", data_type);
        if (is_causal) {
            if (!(window_size_left == seqlen_k && window_size_right == 0)) {
                add_mha_params("window_size_left", window_size_left);
                add_mha_params("window_size_right", window_size_right);
            }
        } else {
            if (!(window_size_left == -1 && window_size_right == -1)) {
                add_mha_params("window_size_left", window_size_left);
                add_mha_params("window_size_right", window_size_right);
            }
        }
        if (alibi) {
            add_mha_params("alibi", alibi);
        }
    }
    std::string format() {
        std::stringstream ss;
        ss << "[FMHA] --format=" << op_name_ << ',' << op_dir_;
        for (auto& iter : args_) {
            if (iter.second != "") {
                ss << ',' << iter.first << ':' << iter.second;
            }
        }
        ss << '.';
        return ss.str();
    }

private:
    std::string val_to_string(int val) {
        return std::to_string(val);
    }
    std::string val_to_string(float val) {
        std::stringstream float_str;
        float_str << std::fixed << std::setprecision(4) << val;
        return float_str.str();
    }
    std::string val_to_string(bool val) {
        return std::to_string(int(val));
    }
    std::string val_to_string(const std::string& val) {
        return val;
    }
    void add_argument(const std::string& name) {
        std::string init_val = "";
        if (args_.find(name) != args_.end()) {
            std::cout << "[" << name << "] already exists." << std::endl;
            throw std::runtime_error("Add argument fail.");
        }
        args_.insert(std::make_pair(name, init_val));
    }

protected:
    std::string                                  op_name_;
    std::string                                  op_dir_;
    std::unordered_map<std::string, std::string> args_;
};
class ProfilingInterface {
public:
    ProfilingInterface(ProfilingInterface const&)                   = delete;
    void                       operator=(ProfilingInterface const&) = delete;
    static ProfilingInterface& Instance(FMHAConfig fmha_config) {
        static ProfilingInterface instance;
        instance.use_nvtx_ = false;
        return instance;
    }
    bool get_op_info() {
        return use_nvtx_;
    }
    void instrument(bool start, FmhaProfParam& fmha_params) {
        if (!get_op_info()) {
            return;
        }
        if (start) {
            std::string op_name = fmha_params.format();
            if (use_nvtx_) {
                nvtxEventAttributes_t eventAttrib = {0};
                eventAttrib.version               = NVTX_VERSION;
                eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
                eventAttrib.message.ascii         = op_name.c_str();
                nvtxDomainRangePushEx(domain_, &eventAttrib);
            }
        } else {
            if (use_nvtx_) {
                nvtxDomainRangePop(domain_);
            }
        }  // if start
    }

private:
    ProfilingInterface() {
        // TODO: add print log
        domain_ = nvtxDomainCreateA("fmha");
    }
    ~ProfilingInterface() {
        nvtxDomainDestroy(domain_);
    }
    bool               use_nvtx_;
    nvtxDomainHandle_t domain_;
};
}  // namespace fmha
}  // namespace rtp_llm