package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.QueueCapacity;
import org.flexlb.balance.schedulev2.decode.DecodeQueue;
import org.flexlb.balance.schedulev2.main.MainQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueue;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
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
    private final LoopSignal prefillLoopSignal;
    private final DecodeQueue decodeQueue;
    private final LoopSignal decodeLoopSignal;

    public RequestCleanup(RequestTable requestTable,
                          MainQueue mainQueue,
                          QueueCapacity queueCapacity,
                          LoopSignal mainLoopSignal,
                          PrefillQueueRegistry prefillQueues,
                          LoopSignal prefillLoopSignal,
                          DecodeQueue decodeQueue,
                          LoopSignal decodeLoopSignal) {
        this.requestTable = requestTable;
        this.mainQueue = mainQueue;
        this.queueCapacity = queueCapacity;
        this.mainLoopSignal = mainLoopSignal;
        this.prefillQueues = prefillQueues;
        this.prefillLoopSignal = prefillLoopSignal;
        this.decodeQueue = decodeQueue;
        this.decodeLoopSignal = decodeLoopSignal;
    }

    /**
     * 请求失败或取消后调用：取消超时任务，移出主队列、Prefill 队列和 Decode 队列，删掉 Prefill 的 dispatch 记录，
     * 归还 Decode 的 reserve、主队列名额和排队总数，最后从请求表移除。
     */
    public void releaseEnded(Request request) {
        request.cancelDeadlineTask();
        mainQueue.remove(request);
        PrefillQueue prefillQueue = prefillQueues.find(request);
        if (prefillQueue != null) {
            prefillQueue.removeWaiting(request);
            if (prefillQueue.removeDispatched(request)) {
                prefillLoopSignal.wake();
            }
        }
        decodeQueue.remove(request);
        releaseDecodeReservation(request);
        releaseQueueSlot(request);
        queueCapacity.decreaseTotalQueueCount(request.future());
        requestTable.remove(request);
    }

    /**
     * 请求发布后调用：取消超时任务，从请求表移除。
     * Prefill 的 dispatch 记录和 Decode 的 reserve 保留，引擎做完后由各自的上报清掉。
     */
    public void releasePublished(Request request) {
        request.cancelDeadlineTask();
        requestTable.remove(request);
    }

    /**
     * 请求从 Prefill 队列 dispatch 出去后调用：不再算排队，归还主队列名额和排队总数。
     */
    public void releaseAtPrefillDispatch(Request request) {
        releaseQueueSlot(request);
        queueCapacity.decreaseTotalQueueCount(request.future());
    }

    /**
     * 归还请求在 Decode 上的 reserve，并唤醒 Decode 队列循环。没有 reserve 时什么也不做。
     * 只用于还没发布的请求，结果没发给调用方，所以按 NOT_SENT 归还。
     */
    public void releaseDecodeReservation(Request request) {
        DecodeChoice choice = request.takeDecodeChoice();
        if (choice == null) {
            return;
        }
        choice.endpoint().release(choice.reservation(), DecodeEndpoint.ReleaseReason.NOT_SENT);
        decodeLoopSignal.wake();
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
