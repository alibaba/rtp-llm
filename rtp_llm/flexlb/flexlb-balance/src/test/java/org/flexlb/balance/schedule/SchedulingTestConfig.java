package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.config.scheduler.QueueOrderingConfig;
import org.flexlb.config.scheduler.SchedulerConfig;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.EnumSet;

/**
 * Test-only builders for the public scheduler and dispatcher variants.
 */
public final class SchedulingTestConfig {
    private static final java.util.concurrent.atomic.AtomicLong fixtureQueueSequence =
            new java.util.concurrent.atomic.AtomicLong();

    private SchedulingTestConfig() {}

    /**
     * Construct protocol fixtures without exposing a production route factory that bypasses selection.
     */
    public static RequestResult createRoute(
            RequestContext context,
            org.flexlb.worker.endpoint.Response response,
            org.flexlb.worker.status.ServerStatus prefill,
            org.flexlb.worker.status.ServerStatus decode,
            PrefillEndpoint prefillEndpoint,
            DecodeEndpoint decodeEndpoint,
            DecodeEndpoint.RequestRegistration reservation,
            long enqueuedAtMs) {
        if (context.getSchedulingMetadata() == null) {
            context.setSchedulingMetadata(
                    org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                            context.getRequest().getRawPriority(),
                            context.getConfig()
                                    .resolveExpiresAtMs(context.getTiming().startedAtMs())));
        }
        // Production selection always supplies Prefill; state-only fixtures may omit its behavior.
        if (prefillEndpoint == null) {
            prefillEndpoint = org.mockito.Mockito.mock(PrefillEndpoint.class);
            org.mockito.Mockito.when(prefillEndpoint.getIp())
                    .thenReturn(prefill == null ? "127.0.0.1" : prefill.getServerIp());
            org.mockito.Mockito.when(prefillEndpoint.getHttpPort())
                    .thenReturn(prefill == null ? 8080 : prefill.getHttpPort());
            org.mockito.Mockito.when(prefillEndpoint.getGrpcPort())
                    .thenReturn(prefill == null ? 8090 : prefill.getGrpcPort());
        }
        if (prefill == null) {
            prefill = new org.flexlb.worker.status.ServerStatus();
            prefill.setRequestId(context.getRequestId());
            prefill.setRole(org.flexlb.config.service.RoleType.PREFILL);
            prefill.setServerIp(prefillEndpoint.getIp());
            prefill.setHttpPort(prefillEndpoint.getHttpPort());
            prefill.setGrpcPort(prefillEndpoint.getGrpcPort());
            prefill.setSuccess(true);
        }
        if (decode == null && decodeEndpoint != null) {
            decode = new org.flexlb.worker.status.ServerStatus();
            decode.setRequestId(context.getRequestId());
            decode.setRole(org.flexlb.config.service.RoleType.DECODE);
            decode.setServerIp(
                    java.util.Objects.requireNonNullElse(decodeEndpoint.getIp(), "127.0.0.1"));
            decode.setHttpPort(decodeEndpoint.getHttpPort());
            decode.setGrpcPort(decodeEndpoint.getGrpcPort());
            decode.setSuccess(true);
        }
        var roles = new java.util.ArrayList<RoleSelection>();
        prefill = org.flexlb.worker.status.ServerStatus.copyOf(prefill);
        prefill.setSuccess(true);
        prefill.setRequestId(context.getRequestId());
        if (prefill.getRole() == null) prefill.setRole(org.flexlb.config.service.RoleType.PREFILL);
        if (decode != null) {
            decode = org.flexlb.worker.status.ServerStatus.copyOf(decode);
            decode.setSuccess(true);
            decode.setRequestId(context.getRequestId());
            if (decode.getRole() == null) decode.setRole(org.flexlb.config.service.RoleType.DECODE);
        }
        roles.add(
                RoleSelection.prefill(
                        (PrefillEndpoint) fixtureWorker(prefillEndpoint), prefill, 0L, null));
        if (decode != null)
            roles.add(
                    RoleSelection.decode(
                            (DecodeEndpoint) fixtureWorker(decodeEndpoint), decode, reservation));
        RequestResult selected = new RequestResult(context, roles);

        return selected;
    }

    /**
     * Project a fixture's selected roles into an independent response.
     */
    public static org.flexlb.worker.endpoint.Response routeResponse(RequestResult route) {
        return route.successResponse();
    }

    public static org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity capacity(
            FlexlbConfig config) {
        return new SchedulerConfiguration().decodeAdmissionCapacity(configService(config));
    }

    public static org.flexlb.config.ConfigService configService(FlexlbConfig config) {
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        return service;
    }

    public static org.flexlb.balance.schedule.DecodePreemptionPolicy.DecodePreemptionDecision
            planDecode(
                    org.flexlb.worker.endpoint.Request request,
                    int priority,
                    org.flexlb.worker.endpoint.DecodeResources.AdmissionCapacity capacity,
                    org.flexlb.worker.endpoint.DecodeResources.ResourceSnapshot snapshot,
                    PreemptionConfig preemption,
                    java.util.Map<String, String> failures) {
        return new org.flexlb.balance.schedule.PriorityDecodePreemptionPolicy()
                .planDecode(request, priority, capacity, snapshot, preemption, failures);
    }

    public static FlexlbConfig newConfig() {
        FlexlbConfig config = new FlexlbConfig();
        configureRequiredValues(config);
        return config;
    }

    public static void configureRequiredValues(FlexlbConfig config) {
        if (config.getRequestLifecycle().getRequest().getTimeoutMs() == null) {
            config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        }
    }

    public static SchedulerConfig useFifoQueue(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        queue.setOrdering(new QueueOrderingConfig());
        config.setScheduler(queue);
        return queue;
    }

    public static void useSingleDecision(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        config.setScheduler(queue);
    }

    public static void allowVictim(FlexlbConfig config, PreemptionStage stage) {
        PreemptionConfig preemption = preemption(config);
        EnumSet<PreemptionStage> stages =
                preemption.getAllowedPreemptionStages().isEmpty()
                        ? EnumSet.noneOf(PreemptionStage.class)
                        : EnumSet.copyOf(preemption.getAllowedPreemptionStages());
        stages.add(stage);
        preemption.setAllowedPreemptionStages(stages);
    }

    public static PreemptionConfig preemption(FlexlbConfig config) {
        usePriorityQueue(config);
        QueueOrderingConfig priority = config.priorityOrdering();
        if (priority.getPreemption() == null) {
            priority.setPreemption(new PreemptionConfig());
        }
        return priority.getPreemption();
    }

    public static SchedulerConfig usePriorityQueue(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        QueueOrderingConfig priority =
                queue.getOrdering().getType() == QueueOrderingConfig.Type.PRIORITY
                        ? queue.getOrdering()
                        : ConfigTestSupport.priority();
        queue.setOrdering(priority);
        config.setScheduler(queue);
        return queue;
    }

    private static WorkerEndpoint fixtureWorker(WorkerEndpoint endpoint) {
        if (endpoint == null) return null;
        var mock = org.mockito.Mockito.mockingDetails(endpoint);
        if (mock.isMock()
                && !mock.isSpy()
                && mock.getStubbings().stream()
                        .noneMatch(
                                stub ->
                                        stub.getInvocation()
                                                .getMethod()
                                                .getName()
                                                .equals("acceptsNewRequests"))) {
            // Protocol fixtures represent selected workers; explicit unavailable stubs take
            // precedence.
            org.mockito.Mockito.lenient()
                    .when(endpoint.acceptsNewRequests())
                    .thenAnswer(
                            call ->
                                    !endpoint.hasStoppedAcceptingRequests()
                                            && (endpoint.getStatus() == null
                                                    || endpoint.getStatus()
                                                            .isAvailableForScheduling()));
        }
        return endpoint;
    }

    private static SchedulerConfig activeQueueOrNew(FlexlbConfig config) {
        configureRequiredValues(config);
        return config.isQueue() ? config.getScheduler() : new SchedulerConfig();
    }
}
