package org.flexlb.balance.scheduler;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.config.VictimStage;

import java.util.EnumSet;

/** Test-only builders for the public scheduler and dispatcher variants. */
public final class SchedulingTestConfig {
    /** Standalone component fixtures use the same immutable inputs as registered requests. */
    public static RequestContext freezeInputs(RequestContext context) {
        if (context.getRequirements() == null) {
            org.springframework.test.util.ReflectionTestUtils.setField(context, "requirements", RequestRequirements.capture(context));
        }
        return context;
    }

    /** Construct protocol fixtures without exposing a production route factory that bypasses selection. */
    public static RequestRoute createRoute(
            RequestContext context, org.flexlb.dao.loadbalance.Response response,
            org.flexlb.dao.loadbalance.ServerStatus prefill, org.flexlb.dao.loadbalance.ServerStatus decode,
            org.flexlb.balance.endpoint.PrefillEndpoint prefillEndpoint,
            org.flexlb.balance.endpoint.DecodeEndpoint decodeEndpoint,
            org.flexlb.balance.endpoint.DecodeResources.ReservationHandle reservation, long enqueuedAtMs) {
        // Production selection always supplies Prefill; state-only fixtures may omit its behavior.
        if (prefillEndpoint == null) {
            prefillEndpoint = org.mockito.Mockito.mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
            org.mockito.Mockito.when(prefillEndpoint.getIp()).thenReturn(prefill == null ? "127.0.0.1" : prefill.getServerIp());
            org.mockito.Mockito.when(prefillEndpoint.getHttpPort()).thenReturn(prefill == null ? 8080 : prefill.getHttpPort());
            org.mockito.Mockito.when(prefillEndpoint.getGrpcPort()).thenReturn(prefill == null ? 8090 : prefill.getGrpcPort());
        }
        if (prefill == null) {
            prefill = new org.flexlb.dao.loadbalance.ServerStatus();
            prefill.setRequestId(context.getRequestId());
            prefill.setRole(org.flexlb.dao.route.RoleType.PREFILL);
            prefill.setServerIp(prefillEndpoint.getIp());
            prefill.setHttpPort(prefillEndpoint.getHttpPort());
            prefill.setGrpcPort(prefillEndpoint.getGrpcPort());
            prefill.setSuccess(true);
        }
        org.flexlb.balance.strategy.WorkerAssignment prefillAssignment = fixtureAssignment(prefill, prefillEndpoint);
        org.flexlb.balance.strategy.WorkerAssignment decodeAssignment = fixtureAssignment(decode, decodeEndpoint);
        try {
            var constructor = RequestRoute.class.getDeclaredConstructor(RequestContext.class,
                    org.flexlb.dao.loadbalance.Response.class, org.flexlb.balance.strategy.WorkerAssignment.class,
                    org.flexlb.balance.strategy.WorkerAssignment.class,
                    org.flexlb.balance.endpoint.DecodeResources.ReservationHandle.class);
            constructor.setAccessible(true);
            RequestRoute selected = constructor.newInstance(context, response, prefillAssignment, decodeAssignment, null);
            RequestRoute route = RequestRoute.create(context, selected, reservation);
            AbstractRequestScheduler.initializeWorkerQueue(context, enqueuedAtMs);
            return route;
        } catch (java.lang.reflect.InvocationTargetException failure) {
            org.flexlb.util.Failures.rethrow(failure.getCause(), "route fixture construction failed");
            throw new AssertionError(failure);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private static org.flexlb.balance.strategy.WorkerAssignment fixtureAssignment(
            org.flexlb.dao.loadbalance.ServerStatus metadata, org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {
        if (metadata == null && endpoint == null) { return null; }
        return org.mockito.Mockito.mock(org.flexlb.balance.strategy.WorkerAssignment.class, call -> switch (call.getMethod().getName()) {
            case "endpoint" -> endpoint;
            case "serverStatus" -> metadata;
            case "requestId" -> metadata == null ? 0L : metadata.getRequestId();
            case "role" -> metadata == null ? null : metadata.getRole();
            case "group" -> metadata == null ? null : metadata.getGroup();
            case "hitCache" -> metadata == null || metadata.getDebugInfo() == null
                    ? 0L : metadata.getDebugInfo().getHitCacheLen();
            default -> org.mockito.Mockito.RETURNS_DEFAULTS.answer(call);
        });
    }

    /** Read a fixture's nullable template without making it a production API. */
    public static org.flexlb.dao.loadbalance.Response routeResponse(RequestRoute route) {
        return org.flexlb.dao.loadbalance.Response.copyOf(
                (org.flexlb.dao.loadbalance.Response) org.springframework.test.util.ReflectionTestUtils.getField(route, "routeResponse"));
    }

    public static RequestRequirements decodeRequirements(int priority, long hardKv, long expectedKv,
            org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity) {
        return new RequestRequirements(99L, org.flexlb.dao.SchedulingMetadata.explicit(priority, Long.MAX_VALUE), expectedKv, capacity,
                RequestRequirements.DecodeMode.PREEMPT_AT_PLACEMENT,
                newConfig().getRouter().getRoles().getDecode().getCostEstimator().compiledFormula(),
                hardKv, null, java.util.List.of(), 0L, true, 0);
    }

    private SchedulingTestConfig() {
    }

    public static FlexlbConfig batchConfig() {
        FlexlbConfig config = newConfig();
        useBatchDispatcher(config);
        return config;
    }

    public static SchedulerConfig usePriorityQueue(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        QueueOrderingConfig priority = queue.getOrdering().getType()
                == QueueOrderingConfig.Type.PRIORITY
                ? queue.getOrdering() : QueueOrderingConfig.priority();
        queue.setOrdering(priority);
        config.setScheduler(queue);
        return queue;
    }

    public static SchedulerConfig useFifoQueue(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        queue.setOrdering(new QueueOrderingConfig());
        config.setScheduler(queue);
        return queue;
    }

    public static void useSingleDecision(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        queue.setDecision(DecisionPolicyConfig.single());
        config.setScheduler(queue);
    }

    public static DecisionPolicyConfig useFixedWindowDecision(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        if (queue.getDecision().getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW) {
            return queue.getDecision();
        }
        DecisionPolicyConfig fixedWindow = new DecisionPolicyConfig();
        queue.setDecision(fixedWindow);
        config.setScheduler(queue);
        return fixedWindow;
    }

    public static DispatcherConfig useBatchDispatcher(FlexlbConfig config) {
        configureRequiredValues(config);
        if (config.getDispatcher().getType() == DispatcherConfig.Type.BATCH) {
            return config.getDispatcher();
        }
        DispatcherConfig batch = new DispatcherConfig();
        batch.setMaxInflightPerPrefillWorker(2);
        config.setDispatcher(batch);
        return batch;
    }

    public static DispatcherConfig useNonBatchDispatcher(FlexlbConfig config) {
        configureRequiredValues(config);
        if (config.getDispatcher().getType()
                == DispatcherConfig.Type.NON_BATCH) {
            return config.getDispatcher();
        }
        DispatcherConfig nonBatch = DispatcherConfig.nonBatch();
        nonBatch.setMaxInflightPerPrefillWorker(64);
        config.setDispatcher(nonBatch);
        return nonBatch;
    }

    public static PreemptionConfig preemption(FlexlbConfig config) {
        usePriorityQueue(config);
        QueueOrderingConfig priority = config.priorityOrdering();
        if (priority.getPreemption() == null) {
            priority.setPreemption(new PreemptionConfig());
        }
        return priority.getPreemption();
    }

    public static void allowVictim(FlexlbConfig config, VictimStage stage) {
        PreemptionConfig preemption = preemption(config);
        EnumSet<VictimStage> stages = preemption.getAllowedVictimStages().isEmpty()
                ? EnumSet.noneOf(VictimStage.class)
                : EnumSet.copyOf(preemption.getAllowedVictimStages());
        stages.add(stage);
        preemption.setAllowedVictimStages(stages);
    }

    public static FlexlbConfig newConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.getDispatcher().setMaxInflightPerPrefillWorker(2);
        configureRequiredValues(config);
        return config;
    }

    public static void configureRequiredValues(FlexlbConfig config) {
        if (config.getRequestLifecycle().getRequest().getTimeoutMs() == null) {
            config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        }
        if (config.getRequestLifecycle().getDecision().getLifetime() == null) {
            config.getRequestLifecycle().getDecision().setLifetime(2.0);
        }
    }

    private static SchedulerConfig activeQueueOrNew(FlexlbConfig config) {
        configureRequiredValues(config);
        return config.isQueue() ? config.getScheduler() : new SchedulerConfig();
    }
}
