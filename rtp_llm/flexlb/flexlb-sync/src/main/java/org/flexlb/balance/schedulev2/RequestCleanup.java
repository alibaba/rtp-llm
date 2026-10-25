package org.flexlb.balance.schedulev2;

import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.dao.route.RoleType;

/**
 * 归还请求占用的队列名额和计数。新链路里所有归还都走这里，每一项都只会归还一次。
 */
public final class RequestCleanup {

    private final RequestTable requestTable;
    private final MainQueue mainQueue;
    private final QueueCapacity queueCapacity;
    private final LoopSignal mainLoopSignal;

    public RequestCleanup(RequestTable requestTable,
                          MainQueue mainQueue,
                          QueueCapacity queueCapacity,
                          LoopSignal mainLoopSignal) {
        this.requestTable = requestTable;
        this.mainQueue = mainQueue;
        this.queueCapacity = queueCapacity;
        this.mainLoopSignal = mainLoopSignal;
    }

    /**
     * 请求失败或取消后调用：取消超时任务，移出主队列，归还主队列名额和排队总数，最后从请求表移除。
     */
    public void releaseEnded(Request request) {
        request.cancelDeadlineTask();
        mainQueue.remove(request);
        releaseQueueSlot(request);
        queueCapacity.decreaseTotalQueueCount(request.future());
        requestTable.remove(request);
    }

    /**
     * 归还请求占用的主队列名额，并唤醒主队列循环去出队下一个请求。没占名额时什么也不做。
     */
    public void releaseQueueSlot(Request request) {
        RoleType role = request.takeQueueSlot();
        if (role == null) {
            return;
        }
        queueCapacity.releaseEngineSubQueueSlot(role);
        mainLoopSignal.wake();
    }
}
