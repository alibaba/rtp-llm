#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/disaggregate/load_balancer/WRRLoadBalancer.h"

namespace rtp_llm {

WorkerAwaredLoadBalancer::~WorkerAwaredLoadBalancer() {
}

void WorkerAwaredLoadBalancer::stop() {  
    if (!sync_worker_status_stop_) {
        sync_worker_status_stop_ = true;
        sync_worker_status_thread_->join();
        sync_worker_status_thread_.reset();
        service_discovery_thread_->stop();
        service_discovery_thread_.reset();

        RTP_LLM_LOG_INFO("destroy WorkerAwaredLoadBalancer done");
    }
}

bool WorkerAwaredLoadBalancer::init(const LoadBalancerInitParams& params) {
    if (!sync_worker_status_stop_) {
        RTP_LLM_LOG_WARNING("WorkerAwaredLoadBalancer should init multi times");
        return false;
    }
    sync_worker_status_stop_ = false;
    subscribe_service_manager_.reset(new SubscribeServiceManager);
    heartbeat_synchronizer_.reset(new HeartbeatSynchronizer);
    if (!subscribe_service_manager_->init(params.subscribe_config)) {
        RTP_LLM_LOG_WARNING("subscribe service manager init failed, WorkerAwaredLoadBalancer init failed");
        return false;
    }
    if (!heartbeat_synchronizer_->init()) {
        RTP_LLM_LOG_WARNING("heartbeat synchronizer init failed, WorkerAwaredLoadBalancer init failed");
        return false;
    }

    service_discovery_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&WRRLoadBalancer::discovery, this), params.update_interval_ms * 1000, "discovery");

    sync_worker_status_interval_ms_ = params.sync_status_interval_ms;
    sync_worker_status_thread_ =
        autil::Thread::createThread(std::bind(&WRRLoadBalancer::syncWorkerThread, this), "sync_worker_status");

    RTP_LLM_LOG_INFO("WorkerAwaredLoadBalancer init done");
    return true;
}

void WorkerAwaredLoadBalancer::syncWorkerThread() {
    while (!sync_worker_status_stop_) {
        int64_t                            start_time_us = autil::TimeUtility::currentTime();
        syncWorkerStatus();
        int64_t end_time_us  = autil::TimeUtility::currentTime();
        int     wait_time_us = sync_worker_status_interval_ms_ * 1000 - (end_time_us - start_time_us);
        if (wait_time_us > 0) {
            usleep(wait_time_us);
        }
    }
}

void WorkerAwaredLoadBalancer::syncWorkerStatus() {
    std::shared_lock<std::shared_mutex> lock(biz_hosts_mutex_);
    auto result = heartbeat_synchronizer_->getHeartbeatFromHost(biz_hosts_, sync_worker_status_interval_ms_);
    updateWorkerStatusImpl(result);
}


}  // namespace rtp_llm