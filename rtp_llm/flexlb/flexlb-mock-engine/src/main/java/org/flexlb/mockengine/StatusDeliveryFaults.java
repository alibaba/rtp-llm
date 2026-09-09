package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/** Test-only report corruption. All access is guarded by the service completion lock.
 * Delayed records have no version until publication, so a reader cannot acknowledge
 * a record that was withheld. Missing rounds count actual successful status replies.
 */
final class StatusDeliveryFaults {
    static final class Rule {
        int missingRounds;
        long delayMs;
        long releasedVersion;
        final List<Pending> pending = new ArrayList<>();
    }
    private record Pending(EngineRpcService.TaskInfoPB task, long readyNs) { }
    final Map<Long, Rule> rules = new LinkedHashMap<>();

    void configure(long rid, int missingRounds, long delayMs) {
        if (rid <= 0 || missingRounds < -1 || missingRounds > 1000
                || delayMs < -1 || delayMs > 60_000) {
            throw new IllegalArgumentException("require positive rid, rounds 0..1000, delay_ms 0..60000");
        }
        Rule rule = rules.computeIfAbsent(rid, ignored -> new Rule());
        if (missingRounds >= 0) rule.missingRounds = missingRounds;
        if (delayMs >= 0) rule.delayMs = delayMs;
    }

    boolean defer(EngineRpcService.TaskInfoPB task, long nowNs) {
        Rule rule = rules.get(task.getRequestId());
        if (rule == null || rule.delayMs == 0) return false;
        rule.pending.add(new Pending(task, nowNs + TimeUnit.MILLISECONDS.toNanos(rule.delayMs)));
        return true;
    }

    void releaseReady(long nowNs, Consumer<EngineRpcService.TaskInfoPB> publish) {
        for (Rule rule : rules.values()) {
            // Releasing into a still-suppressed reply would lose the record again.
            if (rule.missingRounds > 0) continue;
            var iterator = rule.pending.iterator();
            while (iterator.hasNext()) {
                Pending pending = iterator.next();
                if (rule.delayMs == 0 || nowNs - pending.readyNs >= 0) {
                    iterator.remove();
                    publish.accept(pending.task);
                }
            }
        }
    }

    List<Long> filter(EngineRpcService.WorkerStatusPB.Builder status) {
        List<Long> hidden = new ArrayList<>();
        rules.forEach((rid, rule) -> {
            if (rule.missingRounds == 0) return;
            rule.missingRounds--;
            hidden.add(rid);
            for (int i = status.getRunningTaskInfoCount() - 1; i >= 0; i--)
                if (status.getRunningTaskInfo(i).getRequestId() == rid) status.removeRunningTaskInfo(i);
            for (int i = status.getFinishedTaskListCount() - 1; i >= 0; i--)
                if (status.getFinishedTaskList(i).getRequestId() == rid) status.removeFinishedTaskList(i);
        });
        return hidden;
    }

    void clear(Consumer<EngineRpcService.TaskInfoPB> publish) {
        for (Rule rule : rules.values())
            for (Pending pending : rule.pending) publish.accept(pending.task);
        rules.clear();
    }
}
