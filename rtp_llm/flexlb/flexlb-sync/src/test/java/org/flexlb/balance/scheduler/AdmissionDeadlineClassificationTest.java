package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** End-to-end coverage for typed Auto-TPM admission-deadline failures. */
class AdmissionDeadlineClassificationTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";

    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private PrefillEndpoint prefillEndpoint;
    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        ConfigService configService = mock(ConfigService.class);
        Router router = mock(Router.class);
        BatchDispatcher dispatcher = mock(BatchDispatcher.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);

        config = new FlexlbConfig();
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchFixedWaitMs(3_600_000);
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbBatchQueueMaxSize(100);
        when(configService.loadBalanceConfig()).thenReturn(config);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation ->
                route(invocation.<BalanceContext>getArgument(0).getRequestId()));

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        scheduler = new FlexlbBatchScheduler(configService, router, endpointRegistry,
                dispatcher, reporter, null, null);

        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, status);
        prefillEndpoint = endpointRegistry.getPrefill(PREFILL_IP_PORT);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void queuedDeadlineReportsHigherPriorityAhead() {
        long now = System.currentTimeMillis();
        BatchItem blocker = autoTpmItem(1L, 70, now);
        BatchItem victim = autoTpmItem(2L, 50, now + 1);
        enqueue(blocker);
        enqueue(victim);

        scheduler.onTimeout(victim, new TimeoutException("admission deadline exceeded"));

        assertAdmissionFailure(victim, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
        assertEquals(List.of(1L), queuedRequestIds());
    }

    @Test
    void queuedDeadlineReportsEarlierSamePriorityAhead() {
        long now = System.currentTimeMillis();
        BatchItem blocker = autoTpmItem(11L, 50, now);
        BatchItem victim = autoTpmItem(12L, 50, now + 1);
        enqueue(blocker);
        enqueue(victim);

        scheduler.onTimeout(victim, new TimeoutException("admission deadline exceeded"));

        assertAdmissionFailure(victim, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
        assertEquals(List.of(11L), queuedRequestIds());
    }

    @Test
    void queuedHeadDeadlineReportsResourceExhausted() {
        long now = System.currentTimeMillis();
        BatchItem victim = autoTpmItem(21L, 70, now);
        BatchItem lowerBehind = autoTpmItem(22L, 30, now + 1);
        enqueue(victim);
        enqueue(lowerBehind);

        scheduler.onTimeout(victim, new TimeoutException("admission deadline exceeded"));

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertEquals(List.of(22L), queuedRequestIds());
    }

    @Test
    void enqueueBatchTimeoutWithoutQueueEvidenceReportsResourceExhausted() {
        BatchItem victim = autoTpmItem(31L, 50, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(victim));

        scheduler.onTimeout(victim, new TimeoutException("EnqueueBatch deadline"));

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertTrue(queuedRequestIds().isEmpty());
    }

    @Test
    void inflightTtlWithoutQueueEvidenceReportsResourceExhausted() {
        BatchItem victim = autoTpmItem(41L, 50, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(victim));
        config.setFlexlbInflightTtlMs(-1);

        scheduler.cleanupInflight();

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertTrue(queuedRequestIds().isEmpty());
    }

    @Test
    void legacyRegistrationWithScheduleBudgetKeepsBatchSloExpired() {
        config.setAutoTpmEnabled(false);
        BalanceContext context = context(51L, 50);
        long now = System.currentTimeMillis();
        context.setBudget(ScheduleBudget.forDeadline(50, now, now + 3_600_000));
        CompletableFuture<Response> future = scheduler.submit(context);
        config.setFlexlbInflightTtlMs(-1);

        scheduler.cleanupInflight();

        Response response = future.getNow(null);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.UNSPECIFIED, response.getAdmissionRejectReason());
    }

    private void enqueue(BatchItem item) {
        assertTrue(scheduler.registerInflight(item));
        assertTrue(prefillEndpoint.getBatcher().tryOffer(item));
    }

    private List<Long> queuedRequestIds() {
        return prefillEndpoint.getBatcher().queueManager().snapshot().items().stream()
                .map(snapshot -> snapshot.requestId())
                .toList();
    }

    private static void assertAdmissionFailure(BatchItem item,
                                               StrategyErrorType errorType,
                                               AdmissionRejectReason reason) {
        Response response = item.future().getNow(null);
        assertFalse(response.isSuccess());
        assertEquals(errorType.getErrorCode(), response.getCode());
        assertEquals(reason, response.getAdmissionRejectReason());
    }

    private BatchItem autoTpmItem(long requestId, int priority, long enqueuedAtMs) {
        BalanceContext context = context(requestId, priority);
        context.setBudget(ScheduleBudget.forDeadline(
                priority, enqueuedAtMs, enqueuedAtMs + 3_600_000));
        return item(context, enqueuedAtMs);
    }

    private BatchItem item(BalanceContext context, long enqueuedAtMs) {
        Response route = route(context.getRequestId());
        return new BatchItem(context, new CompletableFuture<>(), route,
                FlexlbBatchScheduler.findServer(route, RoleType.PREFILL), null,
                prefillEndpoint, null, enqueuedAtMs);
    }

    private static BalanceContext context(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(new FlexlbConfig());
        return context;
    }

    private static Response route(long requestId) {
        ServerStatus prefill = new ServerStatus();
        prefill.setSuccess(true);
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8081);
        prefill.setRequestId(requestId);
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(prefill));
        return response;
    }
}
