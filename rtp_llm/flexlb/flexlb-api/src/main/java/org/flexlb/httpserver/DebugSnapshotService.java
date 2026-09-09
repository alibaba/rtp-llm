package org.flexlb.httpserver;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestRegistry;
import org.flexlb.balance.scheduler.RequestScheduler;
import org.flexlb.debug.DebugPage;
import org.flexlb.debug.DebugQuery;
import org.flexlb.debug.DebugRows;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.function.Function;

/** Read-only composition: component snapshots deliberately do not share a lock or timestamp. */
@Component
@ConditionalOnProperty(name = "flexlb.debug.enabled", havingValue = "true")
public class DebugSnapshotService {
    public static final Set<String> INCLUDES = Set.of("scheduler", "queues", "prefill", "decode", "engine");
    private final RequestRegistry registry;
    private final RequestScheduler scheduler;
    private final EndpointRegistry endpoints;
    private final String instanceId = UUID.randomUUID().toString();

    public DebugSnapshotService(RequestRegistry registry, RequestScheduler scheduler, EndpointRegistry endpoints) {
        this.registry = registry;
        this.scheduler = scheduler;
        this.endpoints = endpoints;
    }

    public record Snapshot(int schemaVersion, String instanceId, String snapshotId,
                           long captureStartedAtMs, long captureFinishedAtMs, String status,
                           boolean endpointDirectoryTruncated, String endpointScope,
                           Map<String, DebugPage> components) {
        public Snapshot {
            components = Map.copyOf(components);
        }
    }

    public Snapshot capture(DebugQuery query, Set<String> include, int endpointLimit) {
        if (include.isEmpty() || !INCLUDES.containsAll(include) || endpointLimit < 1 || endpointLimit > 256) {
            throw new IllegalArgumentException("invalid include or endpoint_limit (1..256)");
        }
        long started = System.currentTimeMillis();
        Map<String, DebugPage> result = new LinkedHashMap<>();
        Budget budget = new Budget(query);
        if (include.contains("scheduler")) {
            result.put("scheduler", budget.capture(registry::debugSnapshot));
        }
        if (include.contains("queues")) {
            result.put("queues", budget.capture(scheduler::debugQueueSnapshot));
        }
        boolean endpointRequested = include.stream().anyMatch(Set.of("prefill", "decode", "engine")::contains);
        List<WorkerEndpoint> directory = endpointRequested ? endpoints.debugEndpoints(endpointLimit + 1) : List.of();
        boolean directoryTruncated = directory.size() > endpointLimit;
        for (WorkerEndpoint endpoint : directory.subList(0, Math.min(directory.size(), endpointLimit))) {
            String generation = Long.toString(endpoint.getStatus().getGenerationId());
            Map<String, Object> identity = DebugRows.fields("endpoint", endpoint.ipPort(),
                    "endpoint_generation", generation);
            if (include.contains("prefill") && endpoint instanceof PrefillEndpoint prefill) {
                result.put("prefill/" + generation, withIdentity(budget.capture(prefill::debugSnapshot), identity));
            }
            if (include.contains("decode") && endpoint instanceof DecodeEndpoint decode) {
                result.put("decode/" + generation, withIdentity(budget.capture(decode::debugSnapshot), identity));
            }
            if (include.contains("engine")) {
                result.put("engine/" + generation,
                        withIdentity(budget.capture(q -> engineSnapshot(endpoint, q)), identity));
            }
        }
        boolean partial = directoryTruncated || result.values().stream()
                .anyMatch(page -> !Set.of("ok", "not_applicable").contains(page.status()));
        return new Snapshot(1, instanceId, UUID.randomUUID().toString(), started,
                System.currentTimeMillis(), partial ? "partial" : "ok", directoryTruncated,
                "advisory_registered_generations_excludes_detached_retirement", result);
    }

    private static DebugPage withIdentity(DebugPage page, Map<String, Object> identity) {
        Map<String, Object> metadata = new LinkedHashMap<>(page.metadata());
        metadata.putAll(identity);
        return new DebugPage(page.consistency(), page.status(), page.captureStartedAtMs(),
                page.captureFinishedAtMs(), page.scannedCount(), page.truncated(), page.rows(), metadata);
    }

    private static DebugPage engineSnapshot(WorkerEndpoint endpoint, DebugQuery query) {
        // One publication read binds fields to both status cursors.
        DebugRows rows = new DebugRows(query);
        var committed = endpoint.getStatus().committedWorkerStatus();
        for (var task : committed.fields().runningTaskList().values()) {
            if (!rows.visit()) {
                break;
            }
            if (query.matches(task.requestId())) {
                rows.add(DebugRows.fields("request_id", Long.toString(task.requestId()),
                        "batch_id", Long.toString(task.batchId()), "phase", task.phase().name(),
                        "dp_rank", task.dpRank()));
            }
        }
        return rows.finish("atomic_publication", DebugRows.fields("scope", "last_committed_engine_report",
                "status_version", Long.toString(committed.cursor().statusVersion()),
                "finished_task_version", Long.toString(committed.cursor().latestFinishedTaskVersion()),
                "reported_task_count", committed.fields().runningTaskList().size(),
                "freshness", "not_established_by_capture_time"));
    }

    /** Global per-response budget, including all endpoints and batch member rows. */
    private static final class Budget {
        private int remainingRows;
        private int remainingScans;
        private final Long requestId;

        private Budget(DebugQuery query) {
            remainingRows = query.limit();
            remainingScans = query.scanLimit();
            requestId = query.requestId();
        }

        private DebugPage capture(Function<DebugQuery, DebugPage> reader) {
            if (remainingRows == 0 || remainingScans == 0) {
                return DebugPage.unavailable("not_captured", "budget_exhausted", System.currentTimeMillis());
            }
            try {
                DebugPage page = reader.apply(new DebugQuery(
                        Math.min(remainingRows, remainingScans), remainingScans, requestId));
                remainingRows -= page.rows().size();
                remainingScans -= page.scannedCount();
                return page;
            } catch (RuntimeException failure) {
                // A failed owner may already have consumed its full scan allowance.
                remainingScans = 0;
                return DebugPage.unavailable("not_captured", "unavailable", System.currentTimeMillis());
            }
        }
    }
}
