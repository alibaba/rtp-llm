#pragma once

#include "aios/navi_sched/scheduler/MasterScheduleContextKernel.h"
#include "aios/navi_sched/scheduler/MasterScheduleContextData.h"

namespace navi_sched {

extern const std::string MASTER_SCHEDULE_RESULT_TYPE_ID;

class MasterScheduleResultState {
    friend class ::NaviSchedResourceTest;
public:
    struct Worker {
        std::vector<size_t> nodes;
        std::vector<size_t> requests;
        std::vector<size_t> failures;
        int64_t reserveUs = 0;
        int64_t cleanupUs = 0;
        int64_t reserveBeginUs = 0;
        int64_t cleanupBeginUs = 0;
        bool success = true;
    };
    struct DeliveryWorker {
        size_t begin = 0;
        size_t end = 0;
        std::vector<size_t> failures;
        int64_t elapsedUs = 0;
        int64_t beginUs = 0;
        bool success = true;
    };

    explicit MasterScheduleResultState(NaviSchedScheduleContextPtr value);
    uint64_t getScheduleId() const { return context->_scheduleId; }
    void partition(size_t workerCount);
    void reserveWorker(size_t workerIndex, MasterStateR &masterState);
    void prepareDelivery(size_t workerCount);
    void deliverWorker(size_t workerIndex);
    void collectDeliveryFailures();
    void cleanupWorker(size_t workerIndex);
    void merge();

    NaviSchedScheduleContextPtr context;
    std::shared_ptr<void> completion;
    std::vector<Worker> workers;
    std::vector<DeliveryWorker> deliveryWorkers;
    int64_t beginUs = 0;
    int64_t reserveDispatchUs = 0;
    int64_t reserveBarrierUs = 0;
    int64_t deliveryDispatchUs = 0;
    int64_t deliveryBarrierUs = 0;
    int64_t cleanupDispatchUs = 0;
    int64_t cleanupBarrierUs = 0;
    bool serialFinished = false;

private:
    std::vector<std::vector<size_t>> _nodeRequests;
    std::vector<size_t> _requestWorkers;
};

class MasterScheduleResultData : public navi::Data {
public:
    explicit MasterScheduleResultData(std::shared_ptr<MasterScheduleResultState> value)
        : navi::Data(MASTER_SCHEDULE_RESULT_TYPE_ID), state(std::move(value)) {}
    std::shared_ptr<MasterScheduleResultState> state;
};

class MasterScheduleResultPrepareKernel : public MasterScheduleResultKernel {
public:
    static const std::string KERNEL_ID;
    static constexpr size_t MIN_PARALLEL_REQUESTS = 64;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
private:
    size_t _workerCount = 1;
};

class MasterScheduleResultWorkerKernel : public navi::Kernel {
public:
    static const std::string KERNEL_ID;
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
private:
    size_t _workerIndex = 0;
    int32_t _phase = 0;
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(MasterStateR, _masterStateR);
};

class MasterScheduleResultBarrierKernel : public navi::Kernel {
public:
    static const std::string KERNEL_ID;
    bool isExpensive() const override { return false; }
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
private:
    size_t _workerCount = 1;
    int32_t _phase = 1;
};

}
