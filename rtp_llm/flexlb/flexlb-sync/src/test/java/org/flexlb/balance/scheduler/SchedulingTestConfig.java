package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigTestSupport;
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
    private static final java.util.concurrent.atomic.AtomicLong fixtureQueueSequence = new java.util.concurrent.atomic.AtomicLong();
    /** Construct protocol fixtures without exposing a production route factory that bypasses selection. */
    public static RequestRoute createRoute(
            RequestContext context, org.flexlb.dao.loadbalance.Response response,
            org.flexlb.dao.loadbalance.ServerStatus prefill, org.flexlb.dao.loadbalance.ServerStatus decode,
            org.flexlb.balance.endpoint.PrefillEndpoint prefillEndpoint,
            org.flexlb.balance.endpoint.DecodeEndpoint decodeEndpoint,
            org.flexlb.balance.endpoint.DecodeResources.ReservationHandle reservation, long enqueuedAtMs) {
        if (context.data().getSchedulingMetadata() == null) {
            context.data().setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(
                    context.data().getPriority(), context.data().getRequestExpiresAtMs()));
        }
        // Production selection always supplies Prefill; state-only fixtures may omit its behavior.
        if (prefillEndpoint == null) {
            prefillEndpoint = org.mockito.Mockito.mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
            org.mockito.Mockito.when(prefillEndpoint.getIp()).thenReturn(prefill == null ? "127.0.0.1" : prefill.getServerIp());
            org.mockito.Mockito.when(prefillEndpoint.getHttpPort()).thenReturn(prefill == null ? 8080 : prefill.getHttpPort());
            org.mockito.Mockito.when(prefillEndpoint.getGrpcPort()).thenReturn(prefill == null ? 8090 : prefill.getGrpcPort());
        }
        if (prefill == null) {
            prefill = new org.flexlb.dao.loadbalance.ServerStatus();
            prefill.setRequestId(context.data().getRequestId());
            prefill.setRole(org.flexlb.dao.route.RoleType.PREFILL);
            prefill.setServerIp(prefillEndpoint.getIp());
            prefill.setHttpPort(prefillEndpoint.getHttpPort());
            prefill.setGrpcPort(prefillEndpoint.getGrpcPort());
            prefill.setSuccess(true);
        }
        RequestRoute selected = new RequestRoute(context);
        org.springframework.test.util.ReflectionTestUtils.setField(selected, "routeResponse", response);
        // State-only fixtures bypass selection; seed its completed construction boundary explicitly.
        org.springframework.test.util.ReflectionTestUtils.setField(selected, "constructionState", (byte) 1);
        org.springframework.test.util.ReflectionTestUtils.setField(selected, "prefill", prefill);
        org.springframework.test.util.ReflectionTestUtils.setField(selected, "decode", decode);
        var pins = new org.flexlb.balance.endpoint.WorkerEndpoint.GenerationPin[3];
        org.springframework.test.util.ReflectionTestUtils.setField(selected, "pins", pins);
        pins[0] = fixturePin(prefillEndpoint);
        pins[1] = fixturePin(decodeEndpoint);
        org.springframework.test.util.ReflectionTestUtils.setField(selected, "prefillIndex", (byte) 0);
        org.springframework.test.util.ReflectionTestUtils.setField(selected, "decodeIndex", (byte) (decodeEndpoint == null ? -1 : 1));
        RequestRoute route = RequestRoute.create(context, selected, reservation);
        context.data().initializeWorkerQueue(enqueuedAtMs, fixtureQueueSequence.incrementAndGet());
        return route;
    }

    private static org.flexlb.balance.endpoint.WorkerEndpoint.GenerationPin fixturePin(
            org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {
        if (endpoint == null) { return null; }
        var pin = org.mockito.Mockito.mock(org.flexlb.balance.endpoint.WorkerEndpoint.GenerationPin.class);
        org.mockito.Mockito.when(pin.endpoint()).thenReturn(endpoint);
        return pin;
    }

    static org.flexlb.balance.PlacementResult<RequestRoute, PlacementKey> prepareRoute(RequestContext context,
            java.util.function.Function<RequestRoute, org.flexlb.balance.PlacementResult<RequestRoute, PlacementKey>> select) {
        RequestRoute route = new RequestRoute(context);
        boolean complete = false;
        Throwable primary = null;
        try {
            var result = select.apply(route);
            if (result.status() == org.flexlb.balance.PlacementResult.Status.SUCCESS) {
                if (result.value() != route) { throw new IllegalStateException("fixture returned a foreign route"); }
                route.prepare();
                complete = true;
            }
            return result;
        } catch (RuntimeException | Error failure) {
            primary = failure;
            throw failure;
        } finally {
            if (!complete) {
                Throwable cleanup = org.flexlb.util.Failures.close(route);
                if (primary != null) { org.flexlb.util.Failures.append(primary, cleanup); }
                else { org.flexlb.util.Failures.rethrow(cleanup, "fixture route cleanup failed"); }
            }
        }
    }

    /** Read a fixture's nullable template without making it a production API. */
    public static org.flexlb.dao.loadbalance.Response routeResponse(RequestRoute route) {
        return org.flexlb.dao.loadbalance.Response.copyOf(
                (org.flexlb.dao.loadbalance.Response) org.springframework.test.util.ReflectionTestUtils.getField(route, "routeResponse"));
    }

    /** Mutable metadata for deliberate fault injection into a state-only fixture. */
    static org.flexlb.dao.loadbalance.ServerStatus prefillTemplate(RequestRoute route) {
        return (org.flexlb.dao.loadbalance.ServerStatus)
                org.springframework.test.util.ReflectionTestUtils.getField(route, "prefill");
    }

    public static org.flexlb.config.ConfigService configService(FlexlbConfig config) {
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }

    public static org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity(FlexlbConfig config) {
        return new PlacementConfiguration().decodeAdmissionCapacity(configService(config));
    }

    public static org.flexlb.balance.eviction.EvictionPlanner.DecodeEvictionDecision planDecode(
            org.flexlb.dao.loadbalance.Request request, int priority,
            org.flexlb.balance.endpoint.DecodeResources.AdmissionCapacity capacity,
            org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot snapshot,
            PreemptionConfig preemption, java.util.Map<String, String> failures) {
        return org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                org.flexlb.balance.eviction.EvictionPlanner.class, "planDecode",
                request, priority, capacity, snapshot, preemption, failures);
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
                ? queue.getOrdering() : ConfigTestSupport.priority();
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
        queue.setDecision(ConfigTestSupport.single());
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
        DispatcherConfig nonBatch = ConfigTestSupport.nonBatch();
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
