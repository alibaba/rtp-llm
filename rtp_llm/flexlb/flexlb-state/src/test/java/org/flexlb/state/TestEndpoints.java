package org.flexlb.state;

import java.util.List;
import org.flexlb.state.spi.EngineObservation;
import org.flexlb.state.spi.EnginePhase;
import org.flexlb.state.spi.StateEndpointRef;
import org.flexlb.state.spi.StateRole;

/**
 * 测试用端点 / 引擎观察构造工具（仅测试源码，包内共享）。
 */
final class TestEndpoints {

    /** 最小端点实现。 */
    record Endpoint(long endpointId, StateRole role, long generation) implements StateEndpointRef {
    }

    private TestEndpoints() {
    }

    static Endpoint ep(long id, StateRole role, long generation) {
        return new Endpoint(id, role, generation);
    }

    static EngineObservation.RunningObservation running(long requestId, StateRole side,
                                                        EnginePhase phase, long batchId,
                                                        long kvTokens, long version) {
        return new EngineObservation.RunningObservation(requestId, side, phase, batchId, kvTokens, version);
    }

    static EngineObservation.FinishedObservation finished(long requestId, StateRole side,
                                                          int errorCode, long endTimeMs, long version) {
        return new EngineObservation.FinishedObservation(requestId, side, errorCode, endTimeMs, version);
    }

    /** 完整上报（detailCount = running.size()，上报完整性）。 */
    static EngineObservation observation(Endpoint ep, long round, long statusMs,
                                         List<EngineObservation.RunningObservation> running,
                                         List<EngineObservation.FinishedObservation> finished) {
        return new EngineObservation(ep, round, statusMs, running.size(), running, finished);
    }

    /** 单条 running 观察的完整上报。 */
    static EngineObservation runningOnly(Endpoint ep, long round, long statusMs,
                                         EngineObservation.RunningObservation r) {
        return observation(ep, round, statusMs, List.of(r), List.of());
    }

    /** 单条 finished 观察的完整上报。 */
    static EngineObservation finishedOnly(Endpoint ep, long round, long statusMs,
                                          EngineObservation.FinishedObservation f) {
        return observation(ep, round, statusMs, List.of(), List.of(f));
    }
}
