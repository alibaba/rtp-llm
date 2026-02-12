#include "rtp_llm/cpp/models/elastic_ep_manager/ElasticEPManager.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

ElasticEPManager::ElasticEPManager(int ep_size, int ep_rank) {
    elastic_ep_stats_.ep_size_ = ep_size;
    elastic_ep_stats_.ep_rank_ = ep_rank;
    last_active_ranks_num_     = ep_size;
}

void ElasticEPManager::updateElasticEPStats() {
    try {
        py::gil_scoped_acquire acquire;
        py::module             deepep         = py::module::import("rtp_llm.models_py.distributed.deepep_wrapper");
        py::object             deepep_wrapper = deepep.attr("get_deepep_wrapper_if_initialized")();
        if (deepep_wrapper.is_none()) {
            RTP_LLM_LOG_ERROR("ElasticEPManager: DeepEP wrapper is not initialized");
            return;
        }

        // Update active ranks tensor
        py::object query_result = deepep_wrapper.attr("query_active_ranks")();
        if (query_result.is_none()) {
            RTP_LLM_LOG_ERROR("ElasticEPManager: query_active_ranks returned None");
            return;
        }

        elastic_ep_stats_.active_ranks_tensor_cpu_ = query_result.cast<torch::Tensor>().cpu();
        // Update active ranks count
        const auto tensor_cpu   = elastic_ep_stats_.active_ranks_tensor_cpu_.contiguous();
        int        active_count = 0;
        const int* data_ptr     = tensor_cpu.data_ptr<int>();
        if (data_ptr == nullptr) {
            RTP_LLM_LOG_ERROR("ElasticEPManager: Failed to get data pointer from tensor");
            return;
        }
        const int64_t numel = tensor_cpu.numel();
        for (int64_t i = 0; i < numel; ++i) {
            if (data_ptr[i] == 1) {
                active_count++;
            }
        }
        elastic_ep_stats_.active_ranks_num_ = active_count;

        // Update is downscale flag
        elastic_ep_stats_.is_downscale_ = (last_active_ranks_num_ > elastic_ep_stats_.active_ranks_num_);

        // Update is rank active flag
        elastic_ep_stats_.is_rank_active_ =
            (elastic_ep_stats_.active_ranks_tensor_cpu_[elastic_ep_stats_.ep_rank_].item<int>() == 1);
    } catch (py::error_already_set& e) {
        RTP_LLM_LOG_ERROR("ElasticEPManager::updateElasticEPStats Python error: %s", e.what());
        py::gil_scoped_acquire acquire;
        e.restore();
        PyErr_Clear();
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("ElasticEPManager::updateElasticEPStats C++ error: %s", e.what());
    }
}

void ElasticEPManager::stepForward(ElasticEPStats& stats) {
    updateElasticEPStats();

    // Log downscale event if detected
    if (elastic_ep_stats_.is_downscale_) {
        RTP_LLM_LOG_WARNING("ElasticEPManager::stepForward: active ranks decrease from %d to %d",
                            last_active_ranks_num_,
                            elastic_ep_stats_.active_ranks_num_);
    }
    last_active_ranks_num_ = elastic_ep_stats_.active_ranks_num_;
    stats                  = elastic_ep_stats_;
}
}  // namespace rtp_llm