package org.flexlb.service.monitor;

import org.flexlb.balance.scheduler.RequestState;
import org.flexlb.balance.schedulev2.QueueBlockReason;
import org.flexlb.balance.schedulev2.request.QueueProgress;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.balance.schedulev2.request.RequestTable;
import org.flexlb.balance.schedulev2.request.Stage;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.flexlb.constant.MetricConstant.AUTO_TPM_VICTIM_COUNT;

/**
 * V2 的实际等待人数与出队检查结果。人数按请求阶段采样，不把共享名额当作 Prefill 队列长度。
 * blocked.qps 统计新进入阻塞原因的请求，blocked_attempt.qps 统计每次未通过的检查。
 */
@Component
public class V2QueueReporter {

    public static final String DEPTH = "schedule_v2.queue.depth";
    public static final String OLDEST = "schedule_v2.queue.oldest_wait_ms";
    public static final String BLOCKED = "schedule_v2.queue.blocked.qps";
    public static final String ATTEMPTS = "schedule_v2.queue.blocked_attempt.qps";
    public static final String REJECTED = "schedule_v2.queue.rejected.qps";
    public static final String WAIT = "schedule_v2.queue.wait_ms";
    private final FlexMonitor monitor;
    private Set<QueueKey> previousQueues = Set.of();

    public V2QueueReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        monitor.register(DEPTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(OLDEST, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(WAIT, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        for (String metric : new String[]{BLOCKED, ATTEMPTS, REJECTED}) {
            monitor.register(metric, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        }
    }

    public void blocked(Request request, Stage stage, QueueBlockReason reason, String endpoint) {
        if (request.stage() != stage) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of("queue", queueName(stage), "reason", reason.name(), "endpoint",
                endpoint, "priority", String.valueOf(request.priority()));
        report(ATTEMPTS, tags, 1L);
        if (request.recordBlock(stage, reason.name(), endpoint)) {
            report(BLOCKED, tags, 1L);
        }
    }

    /**
     * 高优先级请求使 Prefill 等待请求退回主队列时，每次成功迁移上报一次。
     * 被退回请求继续排队，不计为拒绝或请求失败。
     */
    public void preemptedReturn(int preemptorPriority, int victimPriority, String endpoint) {
        reportPreemption("prefill_return_to_main", "prefill", "main", "requeue", preemptorPriority, victimPriority,
                endpoint);
    }

    public void mainPreempted(int preemptorPriority, int victimPriority) {
        reportPreemption("main_evicted", "main", "none", "reject", preemptorPriority, victimPriority, "all");
    }

    /**
     * 成功返回 Decode 抢占指令后，按每个受害者在选中时的阶段计数，不代表引擎已确认取消。
     */
    public void decodePreempted(int preemptorPriority, int victimPriority, DecodeTaskPhase phase, String endpoint) {
        String scenario = switch (phase) {
            case RUNNING -> "decode_running";
            case ACCEPTED_NOT_RUNNING -> "decode_accepted";
            case ENGINE_MAY_HAVE_SEEN -> "decode_unconfirmed";
            case MASTER_QUEUED_NOT_DISPATCHED -> throw new IllegalArgumentException("victim is still in master queue");
        };
        reportPreemption(scenario, "decode_engine", "none", "return_cancel_instruction", preemptorPriority,
                victimPriority, endpoint);
    }

    private void reportPreemption(String scenario,
                                  String fromQueue,
                                  String toQueue,
                                  String action,
                                  int preemptorPriority,
                                  int victimPriority,
                                  String endpoint) {
        report(AUTO_TPM_VICTIM_COUNT,
                FlexMetricTags.of("case", scenario, "stage", fromQueue, "from_queue", fromQueue, "to_queue", toQueue,
                        "action", action, "incoming_priority", String.valueOf(preemptorPriority), "victim_priority",
                        String.valueOf(victimPriority), "endpoint", endpoint),
                1L);
    }

    /**
     * future 的唯一完成回调调用此方法。取消、淘汰和失败都按离开时所在队列统计，成功只报排队耗时。
     */
    public void completed(Request request, Response response) {
        QueueProgress progress = request.queueProgress();
        if (progress.stage().isEnded() || response != null && !response.isSuccess()) {
            String reason = progress.stage() == Stage.CANCELLED ? "REQUEST_CANCELLED" : "UNKNOWN";
            if (response != null) {
                for (StrategyErrorType error : StrategyErrorType.values()) {
                    if (error.getErrorCode() == response.getCode()) {
                        reason = error.name();
                        break;
                    }
                }
            }
            if (request.toRequestState().state() == RequestState.Phase.TIMED_OUT) {
                reason = "REQUEST_TIMEOUT";
            }
            report(REJECTED, FlexMetricTags.of("queue", queueName(progress.previousStage()), "reason", reason,
                    "priority", String.valueOf(request.priority())), 1L);
        }
        String result = progress.stage() == Stage.PUBLISHED ? "success" : "failed";
        report(WAIT, FlexMetricTags.of("queue", "main", "result", result), progress.mainMs());
        report(WAIT, FlexMetricTags.of("queue", "prefill", "result", result), progress.prefillMs());
        report(WAIT, FlexMetricTags.of("queue", "decode", "result", result), progress.decodeMs());
    }

    /**
     * 每秒读取一次阶段，过渡中的请求只归属一个队列。已发布请求不计入等待。
     * 空队列继续报零，节点移除后的下一轮补零，避免大盘保留旧值。
     */
    public synchronized void sample(RequestTable requests, List<String> prefillEndpoints) {
        Map<QueueKey, Long> counts = new HashMap<>();
        Map<QueueKey, Long> oldest = new HashMap<>();
        for (Stage stage : new Stage[]{Stage.MAIN_WAITING, Stage.DECODE_WAITING, Stage.DECODE_RESERVED}) {
            counts.put(new QueueKey(queueName(stage), "all"), 0L);
        }
        for (String endpoint : prefillEndpoints) {
            counts.put(new QueueKey("prefill", endpoint), 0L);
        }
        long now = System.nanoTime();
        for (Request request : requests.list()) {
            QueueProgress progress = request.queueProgress();
            Stage stage = progress.stage();
            if (stage.isEnded() || stage == Stage.PUBLISHED) {
                continue;
            }
            var choice = request.prefillChoice();
            String endpoint = stage == Stage.PREFILL_WAITING && choice != null ? choice.endpoint().ipPort() : "all";
            QueueKey key = new QueueKey(queueName(stage), endpoint);
            counts.merge(key, 1L, Long::sum);
            long ageMs =
                    java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(Math.max(0L, now - progress.enteredAtNanos()));
            oldest.merge(key, ageMs, Math::max);
        }
        Set<QueueKey> keys = new HashSet<>(previousQueues);
        keys.addAll(counts.keySet());
        for (QueueKey key : keys) {
            FlexMetricTags tags = FlexMetricTags.of("queue", key.queue(), "endpoint", key.endpoint());
            report(DEPTH, tags, counts.getOrDefault(key, 0L));
            report(OLDEST, tags, oldest.getOrDefault(key, 0L));
        }
        previousQueues = Set.copyOf(counts.keySet());
    }

    public static String queueName(Stage stage) {
        return switch (stage) {
            case MAIN_WAITING -> "main";
            case PREFILL_WAITING -> "prefill";
            case DECODE_WAITING -> "decode_waiting";
            case DECODE_RESERVED -> "decode_reserved";
            default -> "none";
        };
    }

    private void report(String metric, FlexMetricTags tags, long value) {
        try {
            monitor.report(metric, tags, value);
        } catch (Throwable failure) {
            Logger.warn("V2 queue metric report failed: metric={}", metric, failure);
        }
    }

    private record QueueKey(String queue, String endpoint) {
    }
}
