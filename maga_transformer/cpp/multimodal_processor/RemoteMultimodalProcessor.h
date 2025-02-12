#pragma once

#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/dataclass/Query.h"
#include "maga_transformer/cpp/utils/ErrorCode.h"
#include "maga_transformer/cpp/utils/StatusUtil.h"
#include "maga_transformer/cpp/utils/PyUtils.h"
#include "maga_transformer/cpp/model_rpc/RPCPool.h"
#include "maga_transformer/cpp/multimodal_processor/MultimodalProcessor.h"
#include "maga_transformer/cpp/model_rpc/QueryConverter.h"
#include "maga_transformer/cpp/disaggregate/load_balancer/RRLoadBalancer.h"
#include "maga_transformer/cpp/utils/Cm2Config.h"

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

namespace ft = fastertransformer;
namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(py::object mm_process_engine, ft::GptInitParameter params): 
    MultimodalProcessor(mm_process_engine, params) {
        initLoadBalancer();    
    }

private:
    MultimodalRpcPool pool_;
    std::string vit_cluster_name_;
    std::shared_ptr<BaseLoadBalancer> load_balancer_;

    void initLoadBalancer() {
        auto config = makeConfig();
        load_balancer_ = std::make_shared<RRLoadBalancer>();
        FT_CHECK_WITH_INFO(load_balancer_->init(config), "load_balancer init failed");
        FT_LOG_INFO("load balancer init success");
    }

    LoadBalancerInitParams makeConfig() {
        char* use_local_env = std::getenv("USE_LOCAL");
        SubscribeServiceConfig subscribe_config;
        if (use_local_env) {
            // fake test
            char* remote_vit_server_ip_env = std::getenv("REMOTE_VIT_SERVER_IP");
            FT_CHECK_WITH_INFO(remote_vit_server_ip_env, "multimodal server ip must be not empty");
            std::string remote_ip = std::string(remote_vit_server_ip_env);
            uint32_t remote_port = gpt_init_parameter_.remote_rpc_server_port_;
            FT_LOG_INFO("remote rpc server addr: %s:%d", remote_ip.c_str(), remote_port);

            vit_cluster_name_ = "LOCAL";
            LocalNodeJsonize node1(vit_cluster_name_, remote_ip, remote_port);
            LocalSubscribeServiceConfig local_config;
            local_config.nodes.push_back(node1);
            subscribe_config.local_configs.push_back(local_config);
        } else {
            char* vit_cm2_config_env = std::getenv("RTP_LLM_MULTIMODAL_PART_CM2_CONFIG");
            FT_CHECK_WITH_INFO(vit_cm2_config_env, "vit_cm2_config_env must be not empty");
            std::string vit_cm2_config_str = std::string(vit_cm2_config_env);

            Cm2ClusterConfig vit_cm2_config;
            try {
                FromJsonString(vit_cm2_config, vit_cm2_config_str);
            } catch (autil::legacy::ExceptionBase &e) {
                FT_CHECK_WITH_INFO("create json from str[%s] failed", vit_cm2_config_str.c_str());
            }
            vit_cluster_name_ = vit_cm2_config.cluster_name;
            CM2SubscribeServiceConfig cm2_service_config;
            cm2_service_config.zk_host = vit_cm2_config.zk_host;
            cm2_service_config.zk_path = vit_cm2_config.zk_path;
            cm2_service_config.zk_timeout_ms = 10 * 1000;
            cm2_service_config.clusters = {vit_cm2_config.cluster_name};
            subscribe_config.cm2_configs.push_back(cm2_service_config);
        }
        LoadBalancerInitParams params;
        params.subscribe_config = subscribe_config;
        params.update_interval_ms = 100;
        params.sync_status_interval_ms = 10;
        return params;
    }

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs) {
        auto host = load_balancer_->chooseHost(vit_cluster_name_);

        if (!host || host->ip.empty()) {
            return ErrorInfo(ErrorCode::GET_HOST_FAILED, "get host for vit cluster " + vit_cluster_name_ + " failed");
        }
        auto vit_addr = host->ip + ":" + std::to_string(host->rpc_port);
        auto connection_status = pool_.getConnection(vit_addr);
        if (!connection_status.ok()) {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto& connection = connection_status.value();

        auto stub = connection.stub;
        MultimodalOutputsPB output_pb;
        grpc::ClientContext context;
        auto status = stub->RemoteMultimodalEmbedding(&context, QueryConverter::transMMInputsPB(mm_inputs), &output_pb);
        if (!status.ok()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, status.error_message());
        }
        return QueryConverter::transMMOutput(&output_pb);;
    }

};

}
