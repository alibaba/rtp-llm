package org.flexlb.balance.schedulev2;

import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.schedulev2.decode.DecodeQueue;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestTable;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.dao.route.RoleType;

/**
 * 归还请求占用的队列名额和计数。新链路里所有归还都走这里，每一项都只会归还一次。
 */
public final class RequestCleanup {

    private final RequestTable requestTable;
    private final MainQueue mainQueue;
    private final QueueCapacity queueCapacity;
    private final LoopSignal mainLoopSignal;
    private final PrefillQueueRegistry prefillQueues;
    private final DecodeQueue decodeQueue;
    private final LoopSignal decodeLoopSignal;

    public RequestCleanup(RequestTable requestTable,
                          MainQueue mainQueue,
                          QueueCapacity queueCapacity,
                          LoopSignal mainLoopSignal,
                          PrefillQueueRegistry prefillQueues,
                          DecodeQueue decodeQueue,
                          LoopSignal decodeLoopSignal) {
        this.requestTable = requestTable;
        this.mainQueue = mainQueue;
        this.queueCapacity = queueCapacity;
        this.mainLoopSignal = mainLoopSignal;
        this.prefillQueues = prefillQueues;
        this.decodeQueue = decodeQueue;
        this.decodeLoopSignal = decodeLoopSignal;
    }

    /**
     * 请求失败或取消后调用：取消超时任务，移出主队列、Prefill 队列和 Decode 队列的 waiting，删掉 Prefill 的 dispatch 记录，
     * 归还子队列名额和排队总数，最后从请求表移除。
     * 在 Decode 上 reserve 了的请求留在 Decode 队列的 reserved 里，由 Decode 队列线程归还 reserve，这里只唤醒它。
     */
    public void releaseEnded(Request request) {
        request.cancelDeadlineTask();
        mainQueue.remove(request);
        PrefillQueue prefillQueue = prefillQueues.find(request);
        if (prefillQueue != null) {
            prefillQueue.removeWaiting(request);
            prefillQueue.removeDispatched(request);
        }
        decodeQueue.removeWaiting(request);
        if (request.decodeChoice() != null) {
            decodeLoopSignal.wake();
        }
        releaseQueueSlot(request);
        queueCapacity.decreaseTotalQueueCount(request.future());
        requestTable.remove(request);
    }

    /**
     * 请求发布后调用：归还子队列名额和排队总数，取消超时任务，从请求表移除。
     * Prefill 的 dispatch 记录和 Decode 的 reserve 保留，引擎做完后由各自的上报清掉。
     */
    public void releasePublished(Request request) {
        releaseQueueSlot(request);
        queueCapacity.decreaseTotalQueueCount(request.future());
        request.cancelDeadlineTask();
        requestTable.remove(request);
    }

    /**
     * Prefill 不可用时退回主队列，保留总排队名额和原顺序号。调用方已在 Decode 线程归还可能存在的预占。
     */
    public void returnToMain(Request request, Stage expected) {
        if (request.stage() != expected) {
            return;
        }
        PrefillQueue queue = prefillQueues.find(request);
        if (queue != null) {
            queue.removeWaiting(request);
            queue.removeDispatched(request);
        }
        releaseQueueSlot(request);
        request.setPrefillChoice(null);
        request.setPrefillDecidedAtMs(0L);
        request.setPrefillQueuedAtMs(0L);
        if (request.moveTo(expected, Stage.MAIN_WAITING)) {
            mainQueue.addBack(request);
        }
    }

    /**
     * 归还请求占用的子队列名额，并唤醒主队列循环去出队下一个请求。没占名额时什么也不做。
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
