#pragma once

#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/dataclass/Query.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/utils/PyUtils.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/RRLoadBalancer.h"
#include "rtp_llm/cpp/utils/Cm2Config.h"

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/th_op/ConfigModules.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(py::object mm_process_engine, rtp_llm::GptInitParameter params):
        MultimodalProcessor(mm_process_engine, params) {
        initLoadBalancer();
    }

private:
    MultimodalRpcPool                 pool_;
    std::string                       vit_cluster_name_;
    std::shared_ptr<BaseLoadBalancer> load_balancer_;

    void initLoadBalancer() {
        auto config    = makeConfig();
        load_balancer_ = std::make_shared<RRLoadBalancer>();
        RTP_LLM_CHECK_WITH_INFO(load_balancer_->init(config), "load_balancer init failed");
        RTP_LLM_LOG_INFO("load balancer init success");
    }

    LoadBalancerInitParams makeConfig() {
        SubscribeServiceConfig subscribe_config;
        if (gpt_init_parameter_.service_discovery_config.use_local) {
            // fake test
            std::string remote_ip = gpt_init_parameter_.service_discovery_config.remote_vit_server_ip;
            RTP_LLM_CHECK_WITH_INFO(!remote_ip.empty(), "multimodal server ip must be not empty");
            uint32_t remote_port = gpt_init_parameter_.remote_rpc_server_port_;
            RTP_LLM_LOG_INFO("remote rpc server addr: %s:%d", remote_ip.c_str(), remote_port);

            vit_cluster_name_ = "LOCAL";
            LocalNodeJsonize            node1(vit_cluster_name_, remote_ip, remote_port);
            LocalSubscribeServiceConfig local_config;
            local_config.nodes.push_back(node1);
            subscribe_config.local_configs.push_back(local_config);
        } else {
            std::string vit_cm2_config_str =
                gpt_init_parameter_.service_discovery_config.rtp_llm_multimodal_part_cm2_config;
            RTP_LLM_CHECK_WITH_INFO(!vit_cm2_config_str.empty(), "vit_cm2_config must be not empty");

            Cm2ClusterConfig vit_cm2_config;
            try {
                FromJsonString(vit_cm2_config, vit_cm2_config_str);
            } catch (autil::legacy::ExceptionBase& e) {
                RTP_LLM_CHECK_WITH_INFO("create json from str[%s] failed", vit_cm2_config_str.c_str());
            }
            vit_cluster_name_ = vit_cm2_config.cluster_name;
            CM2SubscribeServiceConfig cm2_service_config;
            cm2_service_config.zk_host       = vit_cm2_config.zk_host;
            cm2_service_config.zk_path       = vit_cm2_config.zk_path;
            cm2_service_config.zk_timeout_ms = 10 * 1000;
            cm2_service_config.clusters      = {vit_cm2_config.cluster_name};
            subscribe_config.cm2_configs.push_back(cm2_service_config);
        }
        LoadBalancerInitParams params;
        params.subscribe_config        = subscribe_config;
        params.update_interval_ms      = 100;
        params.sync_status_interval_ms = 10;
        return params;
    }

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs) {
        auto host = load_balancer_->chooseHost(vit_cluster_name_);

        if (!host || host->ip.empty()) {
            return ErrorInfo(ErrorCode::GET_HOST_FAILED, "get host for vit cluster " + vit_cluster_name_ + " failed");
        }
        auto vit_addr          = host->ip + ":" + std::to_string(host->rpc_port);
        auto connection_status = pool_.getConnection(vit_addr);
        if (!connection_status.ok()) {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto& connection = connection_status.value();

        auto                stub = connection.stub;
        MultimodalOutputsPB output_pb;
        grpc::ClientContext context;
        auto status = stub->RemoteMultimodalEmbedding(&context, QueryConverter::transMMInputsPB(mm_inputs), &output_pb);
        if (!status.ok()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, status.error_message());
        }
        return QueryConverter::transMMOutput(&output_pb);
        ;
    }
};

}  // namespace rtp_llm
