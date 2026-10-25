package org.flexlb.balance.schedulev2.decode;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.WorkerDirectory;

/**
 * 在一个 group 里找空的 Decode。请求在选中的 Decode 上放不下时，有空的 Decode 就不看容量直接放过去，
 * 这样特别大的请求（超过 Decode 的 KV 额度，甚至超过总 KV）也不会一直放不下。
 * <p>
 * 空的意思是：引擎上报的 running 和 waiting 都是 0，本地在这台 Decode 上也没有 reserve 或引擎已确认的请求。
 * 引擎上报有间隔，刚放过去、引擎还没报上来的请求只在本地记录里，所以两边都要是 0。
 */
public final class EmptyDecodeFinder {

    private final WorkerDirectory workerDirectory;

    public EmptyDecodeFinder(WorkerDirectory workerDirectory) {
        this.workerDirectory = workerDirectory;
    }

    /**
     * 返回 group 里第一台空的 Decode，选中结果带着这台 Decode 的 hold 和返回给调用方的地址信息；没有时返回 null。
     */
    public SelectedRole find(String group, String requestId) {
        for (DecodeEndpoint.DecodeRoutingView view : workerDirectory.decodeRoutingSnapshot(group)) {
            if (!isEmpty(view)) {
                continue;
            }
            WorkerEndpoint.EndpointHold hold = workerDirectory.holdDecodeEndpoint(view);
            if (hold != null) {
                return DecodeSelector.buildSelectedRole(view, hold, requestId);
            }
        }
        return null;
    }

    static boolean isEmpty(DecodeEndpoint.DecodeRoutingView view) {
        WorkerStatus.EngineObservation engine = view.workerStatus().fields();
        return view.totalLoad() == 0 && engine.runningQueryLen() == 0L && engine.waitingQueryLen() == 0L;
    }
}
