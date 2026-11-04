#include "navi/engine/ScheduleInfo.h"
#include "navi/util/CommonUtil.h"

namespace navi {

thread_local extern size_t current_thread_id;
thread_local extern size_t current_thread_counter;
thread_local extern size_t current_thread_wait_counter;

std::shared_ptr<ScheduleInfo> ScheduleInfo::makeInlineSchedInfo() {
    auto time = CommonUtil::getTimelineTimeNs();
    if (unlikely(0 == current_thread_id)) {
        current_thread_id = (long)syscall(SYS_gettid);
    }
    auto threadId = current_thread_id;
    auto schedInfoPtr = std::make_shared<ScheduleInfo>();
    auto &schedInfo = *schedInfoPtr;
    schedInfo.enqueueTime = time;
    schedInfo.dequeueTime = time;
    schedInfo.schedTid = threadId;
    schedInfo.signalTid = threadId;
    schedInfo.processTid = threadId;
    schedInfo.runningThread = 0;
    schedInfo.threadCounter = current_thread_counter;
    schedInfo.threadWaitCounter = current_thread_wait_counter;
    schedInfo.queueSize = 0;
    return schedInfoPtr;
}

void ScheduleInfo::fillProto(ScheduleInfoDef *info) const {
    info->set_enqueue_time(enqueueTime);
    info->set_dequeue_time(dequeueTime);
    info->set_running_thread(runningThread);
    info->set_sched_tid(schedTid);
    info->set_signal_tid(signalTid);
    info->set_process_tid(processTid);
    info->set_thread_counter(threadCounter);
    info->set_thread_wait_counter(threadWaitCounter);
    info->set_queue_size(queueSize);
    info->set_inline_type(inlineType);
}

}
