#include "src/fastertransformer/th_op/multi_gpu_gpt/RtpLLMOp.h"
#include "autil/Log.h"
#include "c10/util/intrusive_ptr.h"
#include "maga_transformer/cpp/common/torch_bind.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "src/fastertransformer/core/Types.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"
#include "maga_transformer/cpp/metrics/RtpLLMMetrics.h"
#include <tuple>

using namespace std;

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {

RtpLLMOp::RtpLLMOp() {}

void RtpLLMOp::init(const ft::GptInitParameter& gpt_init_params,
                    py::object                  py_layers_weights,
                    py::object                  py_global_weights,
                    py::object                  mm_process_engine) {
    AUTIL_ROOT_LOG_CONFIG();
    AUTIL_ROOT_LOG_SETLEVEL(INFO);
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);

    auto convert = rtp_llm::WeightsConverter(false, gpt_init_params.quant_algo_);
    // TODO(xinfei.sxf) fix py_linear_bias_slopes 传递方式
    rtp_llm::EngineInitParams params(gpt_init_params,
                                     std::move(*convert.createGptWeights(py_layers_weights, py_global_weights)));
    if (gpt_init_params.tp_rank_ == 0) {
    // kmon metric init
        (void)rtp_llm::initKmonitorFactory();
        auto kmon_tags = rtp_llm::getHippoTags();
        params.metrics_reporter.reset(new kmonitor::MetricsReporter("", "", kmon_tags));
    }
    grpc_server_thread_ = std::thread(&RtpLLMOp::_init, this, gpt_init_params.model_rpc_port_, std::move(params), std::move(mm_process_engine));
    grpc_server_thread_.detach();
    while (!is_server_ready_) {
        sleep(1);  // wait 1s for server ready
    }
}

std::tuple<int64_t, int64_t> RtpLLMOp::getKVCacheInfo() {
    auto info = model_rpc_server_->getKVCacheInfo();
    return std::make_tuple(info.available_kv_cache, info.total_kv_cache);
}

void RtpLLMOp::_init(const int64_t model_rpc_port, const rtp_llm::EngineInitParams params, py::object mm_process_engine) {
    std::string server_address("0.0.0.0:" + std::to_string(model_rpc_port));
    model_rpc_server_.reset(new rtp_llm::ModelRpcServiceImpl(params, mm_process_engine));
    if (model_rpc_port < 0) {
        is_server_ready_ = true;
        return;
    }
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(model_rpc_server_.get());
    grpc_server_ = builder.BuildAndStart();

    FT_LOG_INFO("Server listening on %s", server_address.c_str());
    is_server_ready_ = true;
    grpc_server_->Wait();
    is_server_shutdown_ = true;
}

void RtpLLMOp::stop() {
    if (!is_server_shutdown_) {
        if (grpc_server_) {
            grpc_server_->Shutdown();
        }
        model_rpc_server_.reset();
    }
}

RtpLLMOp::~RtpLLMOp() {
    stop();
}

void registerRtpLLMOp(const py::module& m) {
    pybind11::class_<torch_ext::RtpLLMOp>(m, "RtpLLMOp")
        .def(pybind11::init<>())
        .def("init", &torch_ext::RtpLLMOp::init)
        .def("get_kv_cache_info", &torch_ext::RtpLLMOp::getKVCacheInfo)
        .def("stop", &torch_ext::RtpLLMOp::stop);
}

}  // namespace torch_ext
