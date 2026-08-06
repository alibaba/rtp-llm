package org.flexlb.autotpm;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Stage 3 switch matrix for {@link PriorityPressureController#tryPreempt}:
 * the master switch chain ({@code autoTpmEnabled} →
 * {@code autoTpmDecodeRunningPreemptEnabled}) short-circuits before any
 * endpoint inspection, the {@code autoTpmPreemptRateLimitPerMin=0} guardrail
 * never admits a Cancel, and the fully-enabled path reaches victim selection.
 */
class AutoTpmStage3SwitchMatrixTest {

    private static final String EP = "10.0.0.1:8080";
    private static final String EP_IP = "10.0.0.1";
    private static final int EP_GRPC_PORT = 8081;
    private static final long VICTIM_ID = 100L;
    private static final int VICTIM_PRIORITY = 30;
    private static final int INCOMING_PRIORITY = 70;

    private ConfigService configService;
    private EndpointRegistry endpointRegistry;
    private EngineGrpcClient grpcClient;
    private FlexlbMetricHelper metricHelper;
    private DecodeEndpoint decodeEp;
    private FlexlbConfig config;
    private PriorityPressureController controller;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        endpointRegistry = mock(EndpointRegistry.class);
        grpcClient = mock(EngineGrpcClient.class);
        metricHelper = mock(FlexlbMetricHelper.class);
        decodeEp = mock(DecodeEndpoint.class);

        config = new FlexlbConfig(); // library defaults: both switches off
        when(configService.loadBalanceConfig()).thenReturn(config);

        ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
        decodeEndpoints.put(EP, decodeEp);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(decodeEndpoints);
        when(endpointRegistry.getDecode(EP)).thenReturn(decodeEp);
        when(decodeEp.getIp()).thenReturn(EP_IP);
        when(decodeEp.getGrpcPort()).thenReturn(EP_GRPC_PORT);
        when(grpcClient.isCancelSupported(EP_IP, EP_GRPC_PORT)).thenReturn(true);

        controller = new PriorityPressureController(configService, endpointRegistry, grpcClient,
                mock(InflightStore.class), new PriorityRegistry(), metricHelper);
    }

    // ---- a) preempt switch off (library default) → short circuit ----

    @Test
    void tryPreempt_decodeRunningPreemptDisabled_shortCircuits_noCancelSent() {
        config.setAutoTpmEnabled(true);
        assertFalse(config.isAutoTpmDecodeRunningPreemptEnabled(), "library default must be off");
        when(decodeEp.snapshotRunningCandidates(any())).thenReturn(List.of(victimCandidate()));

        Optional<PreemptResult> result = controller.tryPreempt(incomingContext());

        assertTrue(result.isEmpty());
        verifyNoCancelRpc();
        // short circuit happens before any endpoint view is consulted
        verify(endpointRegistry, never()).getDecodeEndpoints();
        verify(decodeEp, never()).snapshotRunningCandidates(any());
    }

    // ---- b) master switch off overrides the preempt switch ----

    @Test
    void tryPreempt_autoTpmDisabled_shortCircuits_evenWithPreemptEnabled() {
        config.setAutoTpmEnabled(false);
        config.setAutoTpmDecodeRunningPreemptEnabled(true);
        when(decodeEp.snapshotRunningCandidates(any())).thenReturn(List.of(victimCandidate()));

        Optional<PreemptResult> result = controller.tryPreempt(incomingContext());

        assertTrue(result.isEmpty());
        verifyNoCancelRpc();
        verify(endpointRegistry, never()).getDecodeEndpoints();
    }

    // ---- c) rate limit 0 → guardrail never admits ----

    @Test
    void tryPreempt_rateLimitZero_guardrailNeverAdmits_noCancelSent() {
        enableAll();
        config.setAutoTpmPreemptRateLimitPerMin(0);
        when(decodeEp.snapshotRunningCandidates(any())).thenReturn(List.of(victimCandidate()));

        // even repeated attempts never pass the guardrail
        for (int i = 0; i < 3; i++) {
            assertTrue(controller.tryPreempt(incomingContext()).isEmpty());
        }

        verifyNoCancelRpc();
        verify(metricHelper, org.mockito.Mockito.times(3)).reportAutoTpmRunningCancel(
                VICTIM_PRIORITY, INCOMING_PRIORITY, PriorityPressureController.RESULT_RATE_LIMITED);
    }

    // ---- d) fully enabled → reaches victim selection ----

    @Test
    void tryPreempt_fullyEnabled_reachesSelector_emptyCandidatesReturnEmpty() {
        enableAll();
        when(decodeEp.snapshotRunningCandidates(any())).thenReturn(List.of());

        Optional<PreemptResult> result = controller.tryPreempt(incomingContext());

        assertTrue(result.isEmpty(), "no eligible victim → empty, no permit burned");
        // the selection flow was actually entered: candidates were snapshotted
        verify(decodeEp).snapshotRunningCandidates(any());
        // but with no victim there is nothing to cancel or rate-limit
        verifyNoCancelRpc();
        verify(metricHelper, never()).reportAutoTpmRunningCancel(anyInt(), anyInt(), any());
    }

    // ==================== fixtures ====================

    private void enableAll() {
        config.setAutoTpmEnabled(true);
        config.setAutoTpmDecodeRunningPreemptEnabled(true);
        config.setAutoTpmPreemptRateLimitPerMin(10);
        config.setAutoTpmEndpointPreemptQpsLimit(0);
        config.setAutoTpmCommitWaitReleaseTimeoutMs(30);
        config.setAutoTpmPreemptCriticalSectionMs(0);
    }

    private void verifyNoCancelRpc() {
        verify(grpcClient, never()).cancelAsync(any(), anyInt(), anyLong(), any(), anyLong());
        verify(grpcClient, never()).cancelAsync(any(), anyInt(), any(), anyLong());
    }

    private static VictimCandidate victimCandidate() {
        return new VictimCandidate(VICTIM_ID, VICTIM_PRIORITY, 5, 500,
                System.currentTimeMillis() - 60_000, EP);
    }

    private static BalanceContext incomingContext() {
        Request request = new Request();
        request.setRequestId(200L);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setPriority(INCOMING_PRIORITY);
        return ctx;
    }
}
