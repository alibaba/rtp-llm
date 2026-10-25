package org.flexlb.balance.endpoint.prefill;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Set;

/**
 * 一个 batch 当前还没结束的成员，每次需要时扫描请求表得到。
 */
final class LiveBatchMembers {
    private final BatchProgress batch;
    private final Set<RequestEntry> members = Collections.newSetFromMap(new IdentityHashMap<>());

    LiveBatchMembers(BatchProgress batch) {
        this.batch = batch;
    }

    BatchProgress batch() {
        return batch;
    }

    Set<RequestEntry> members() {
        return members;
    }

    void add(RequestEntry entry) {
        if (!members.add(entry)) {
            throw new IllegalStateException("duplicate live batch member request_id=" + entry.requestId);
        }
    }

    void remove(RequestEntry entry) {
        if (!members.remove(entry)) {
            throw new IllegalStateException("live batch members lost request_id=" + entry.requestId);
        }
    }

    /**
     * 如果这批结束上报能让所有成员都结束，返回整批的汇总；否则返回 null。不修改 batch 进度。
     */
    FinishedBatch finishedBatchIfAllFinish(Map<String, TerminalObservation> terminals) {
        long maxExecutionTimeMs = batch.maxExecutionTimeMs();
        boolean anyMemberSucceeded = batch.anyMemberSucceeded();
        boolean usableForLearning = batch.usableForLearning();
        for (RequestEntry member : members) {
            TerminalObservation terminal = terminals.get(member.requestId);
            if (terminal == null) {
                return null;
            }
            if (terminal.executionTimeMs() >= 0L) {
                maxExecutionTimeMs = Math.max(maxExecutionTimeMs, terminal.executionTimeMs());
            }
            anyMemberSucceeded |= terminal.workerObserved() && terminal.errorCode() == 0L;
            usableForLearning &= terminal.workerObserved() && terminal.errorCode() == 0L;
        }
        return new FinishedBatch(batch.batchId(), batch.featuresAtCommit(), batch.predictedWorkMs(), maxExecutionTimeMs,
                anyMemberSucceeded, usableForLearning);
    }
}
