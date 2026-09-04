package org.flexlb.httpserver;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.service.RouteService;
import org.flexlb.service.grace.ActiveRequestCounter;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.flexlb.service.optimizer.OptimizerClient;
// dsv4 v1 stack: PrioritySchedulerReporter is the pre-rename name of
// intake3's RequestSchedulerReporter (same constructor slot).
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * Tier-2 follower forwarding four-state matrix for
 * {@link FlexlbServiceImpl#schedule} plus the ForwardGuard decision matrix of
 * the real {@link FlexlbGrpcForwarder}.
 *
 * <p>Matrix source (brief p6, use case master_forward_matrix). The follower
 * B's decision depends on its cached master view, which production derives
 * from {@code LBStatusConsistencyService.getMasterHostIpPort()}. Two layers
 * are covered:
 * <ol>
 *   <li><b>schedule() branch matrix</b> — {@code grpcForwarder} mocked, the
 *       four states injected as {@code MasterForwardResult}s (state ① is
 *       injected through {@code isMaster()}).</li>
 *   <li><b>ForwardGuard matrix of the real forwarder</b> — a real
 *       {@link FlexlbGrpcForwarder} with a mocked
 *       {@link LBStatusConsistencyService}; each state is injected by
 *       stubbing {@code getMasterHostIpPort()} (the Tier-2 thenAnswer
 *       construction from the brief), including a real transport failure to
 *       a dead master address (state ②) and the SELF_TARGET /
 *       MAX_FORWARD_HOPS=1 negative guards.</li>
 * </ol></p>
 *
 * <p>Terminal code: every failed forward answers 8511
 * ({@code StrategyErrorType.BATCH_SLO_EXPIRED}, canRetry=false — the
 * no-retry terminal code of the ambiguity window).</p>
 */
class ScheduleForwardMatrixTest {

    /**
     * The follower's declared IP. 127.0.0.5, not 127.0.0.1: the dead master
     * below reuses 127.0.0.1 for its instant TCP RST, and the SELF_TARGET
     * gate compares IPs only — a distinct declared IP keeps both states
     * reachable without any real interface binding.
     */
    private static final String FOLLOWER_IP = "127.0.0.5";
    private static final String LIVE_MASTER = "10.0.0.9:7001";
    /**
     * Dead master for state ②: 127.0.0.1:1 refuses instantly on loopback
     * (verified on this macOS: 127.0.0.2:1 is silently dropped — SYN
     * retries stall the RPC past any sane timeout). Its IP differs from
     * the follower's declared 127.0.0.5, so the SELF_TARGET gate (the
     * brief's same-machine pitfall) does not intercept before the RPC.
     */
    private static final String DEAD_MASTER = "127.0.0.1:1";
    private static final int TERMINAL_FORWARD_CODE =
            StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode();

    // ---- shared mocks for the schedule() matrix ----
    private RouteService routeService;
    private LBStatusConsistencyService consistency;
    private EngineHealthReporter engineHealthReporter;
    private ActiveRequestCounter activeRequestCounter;
    private ActiveRequestCounter.RequestToken requestToken;
    private FlexlbGrpcForwarder grpcForwarder;
    private FlexlbServiceImpl service;
    private ch.qos.logback.classic.Logger pvLogger;
    private ListAppender<ILoggingEvent> pvAppender;

    // ---- resources for the real-forwarder guard matrix ----
    private EventLoopGroup channelEventLoop;
    private ExecutorService channelExecutor;

    @BeforeEach
    void setUp() {
        routeService = mock(RouteService.class);
        consistency = mock(LBStatusConsistencyService.class);
        engineHealthReporter = mock(EngineHealthReporter.class);
        grpcForwarder = mock(FlexlbGrpcForwarder.class);

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());

        activeRequestCounter = mock(ActiveRequestCounter.class);
        requestToken = mock(ActiveRequestCounter.RequestToken.class);
        when(activeRequestCounter.acquire()).thenReturn(requestToken);
        CacheAwareService cacheAwareService = mock(CacheAwareService.class);
        when(cacheAwareService.prepareBlockCacheKeys(any())).thenReturn(CompletableFuture.completedFuture(null));

        service = new FlexlbServiceImpl(
                routeService,
                consistency,
                engineHealthReporter,
                activeRequestCounter,
                grpcForwarder,
                configService,
                mock(BatchSchedulerReporter.class),
                mock(ServerScheduleLatencyRecorder.class),
                mock(PrioritySchedulerReporter.class),
                cacheAwareService,
                mock(OptimizerClient.class));

        pvLogger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        pvAppender = new ListAppender<>();
        pvAppender.start();
        pvLogger.addAppender(pvAppender);
    }

    @AfterEach
    void tearDown() {
        pvLogger.detachAppender(pvAppender);
        pvAppender.stop();
        if (channelExecutor != null) {
            channelExecutor.shutdownNow();
        }
        if (channelEventLoop != null) {
            channelEventLoop.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        }
    }

    // ------------------------------------------------------------------
    // Part 1: schedule() four-state matrix (forwarder mocked)
    // ------------------------------------------------------------------

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("state ① follower became master (ZK re-election done): LOCAL_MASTER routes locally")
    void state1LocalMasterRoutesLocally() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(true);
        stubSuccessfulLocalRoute();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request("90_001"), observer);

        verify(grpcForwarder, never()).forwardScheduleToMaster(any());
        verify(routeService, times(1)).route(any());
        assertSuccessfulResponse(observer);
        assertSinglePvContains("\"scheduleOrigin\":\"LOCAL_MASTER\"");
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("state ② cached master is dead + forward fails: terminal 8511, never routed locally")
    void state2DeadMasterForwardFailsTerminallyWithoutLocalRetry() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.failed(
                                "UNAVAILABLE", DEAD_MASTER)));
        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request("90_002"), observer);

        // Terminal 8511, success=false, master host surfaced for observability.
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = capturedResponse(observer);
        assertFalse(response.getSuccess());
        assertEquals(TERMINAL_FORWARD_CODE, response.getCode());
        assertEquals(DEAD_MASTER, response.getRealMasterHost());
        assertEquals(8511, TERMINAL_FORWARD_CODE);

        // The failed forward is terminal: no local routing attempt and no
        // implicit cancel RPC. Explicit client cancellation uses cancel().
        verify(routeService, never()).route(any());
        verify(grpcForwarder, never()).forwardCancelToMaster(any());
        verify(requestToken, times(1)).close();
        assertSinglePvContains("\"code\":8511");
        assertSinglePvContains("\"scheduleOrigin\":\"FORWARD_FAILED\"");
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("state ③ MASTER_NULL election window: LOCAL_FALLBACK routes locally, no cancel")
    void state3MasterNullFallsBackToLocalRouting() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.noMaster()));
        stubSuccessfulLocalRoute();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request("90_003"), observer);

        verify(grpcForwarder, times(1)).forwardScheduleToMaster(any());
        verify(grpcForwarder, never()).forwardCancelToMaster(any());
        verify(routeService, times(1)).route(any());
        assertSuccessfulResponse(observer);
        assertSinglePvContains("\"scheduleOrigin\":\"LOCAL_FALLBACK\"");
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("state ④ cached master alive + forward succeeds: response passed through untouched, client cannot tell")
    void state4LiveMasterForwardIsTransparentToClient() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB masterResponse =
                FlexlbScheduleProtocol.FlexlbScheduleResponsePB.newBuilder()
                        .setSuccess(true)
                        .setCode(200)
                        .setEnqueuedByMaster(true)
                        .setRealMasterHost(LIVE_MASTER)
                        .build();
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.forwarded(
                                masterResponse, LIVE_MASTER)));

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request("90_004"), observer);

        // The exact master response object is delivered — byte-for-byte
        // passthrough, the client cannot perceive it talked to a follower.
        verify(observer, times(1)).onNext(masterResponse);
        verify(observer, times(1)).onCompleted();
        verify(observer, never()).onError(any());
        verify(routeService, never()).route(any());
        verify(grpcForwarder, never()).forwardCancelToMaster(any());
        verify(requestToken, times(1)).close();

        // The forwarding node writes no PV record: no local scheduling trace.
        assertTrue(pvAppender.list.isEmpty(),
                "forwarding follower must not write a local PV record");
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("no ZK consistency config: LOCAL_STANDALONE routes locally, forwarder dormant")
    void state0StandaloneRoutesLocallyWhenConsistencyDisabled() {
        when(consistency.isNeedConsistency()).thenReturn(false);
        stubSuccessfulLocalRoute();

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request("90_005"), observer);

        verifyNoInteractions(grpcForwarder);
        verify(routeService, times(1)).route(any());
        assertSuccessfulResponse(observer);
        assertSinglePvContains("\"scheduleOrigin\":\"LOCAL_STANDALONE\"");
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("negative: SELF_TARGET guard failure skips cancel reconciliation (guard rejected before any RPC)")
    void guardBlockedFailureSkipsCancelReconciliation() {
        when(consistency.isNeedConsistency()).thenReturn(true);
        when(consistency.isMaster()).thenReturn(false);
        // Pathological config: cached master equals self while isMaster=false.
        // The guard rejects it before any RPC, so ownership is unambiguous and
        // no cancel reconciliation may be started.
        when(grpcForwarder.forwardScheduleToMaster(any())).thenReturn(
                CompletableFuture.completedFuture(
                        FlexlbGrpcForwarder.MasterForwardResult.failed(
                                "SELF_FORWARD_BLOCKED", FOLLOWER_IP + ":7001")));

        StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer =
                mock(StreamObserver.class);

        service.schedule(request("90_006"), observer);

        verify(grpcForwarder, never()).forwardCancelToMaster(any());
        verify(routeService, never()).route(any());
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = capturedResponse(observer);
        assertFalse(response.getSuccess());
        assertEquals(TERMINAL_FORWARD_CODE, response.getCode());
    }

    // ------------------------------------------------------------------
    // Part 2: real FlexlbGrpcForwarder ForwardGuard matrix (Tier-2
    // thenAnswer injection through getMasterHostIpPort)
    // ------------------------------------------------------------------

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("guard matrix: MASTER_NULL view produces noMaster() without any RPC")
    void guardMasterNullProducesNoMasterWithoutRpc() throws Exception {
        try (RealForwarderFixture fixture = newRealForwarderFixture(null)) {
            FlexlbGrpcForwarder.MasterForwardResult result =
                    fixture.forwarder.forwardScheduleToMaster(request("90_101"))
                            .toCompletableFuture().get(5, TimeUnit.SECONDS);

            assertFalse(result.masterFound(), "no master selected");
            assertNull(result.response(), "no RPC attempted, no response");
            assertEquals("MASTER_NULL", result.failure());
            verify(fixture.engineHealthReporter).reportForwardToMasterResult(
                    "LOCAL", "MASTER_NULL");
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("guard matrix: SELF_TARGET (cached master == self, isMaster=false) is blocked without any RPC")
    void guardSelfTargetIsBlockedWithoutRpc() throws Exception {
        String selfTarget = FOLLOWER_IP + ":7001";
        try (RealForwarderFixture fixture = newRealForwarderFixture(selfTarget)) {
            FlexlbGrpcForwarder.MasterForwardResult result =
                    fixture.forwarder.forwardScheduleToMaster(request("90_102"))
                            .toCompletableFuture().get(5, TimeUnit.SECONDS);

            assertTrue(result.masterFound(), "a master was selected (self) before the guard");
            assertNull(result.response(), "SELF_TARGET must not attempt an RPC");
            assertEquals("SELF_FORWARD_BLOCKED", result.failure());
            assertEquals(selfTarget, result.masterHost());
            verify(fixture.engineHealthReporter).reportForwardToMasterResult(
                    FOLLOWER_IP, "SELF_TARGET");
        }
    }

    @Test
    @Timeout(value = 10, unit = TimeUnit.SECONDS)
    @DisplayName("guard matrix: forwardHop=1 hits MAX_FORWARD_HOPS=1 and is blocked (no forwarding storm)")
    void guardHopLimitBlocksSecondHop() throws Exception {
        try (RealForwarderFixture fixture = newRealForwarderFixture(LIVE_MASTER)) {
            FlexlbScheduleProtocol.FlexlbScheduleRequestPB alreadyForwardedOnce =
                    FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                            .setRequestId("90_103")
                            .setForwardHop(1)
                            .build();

            FlexlbGrpcForwarder.MasterForwardResult result =
                    fixture.forwarder.forwardScheduleToMaster(alreadyForwardedOnce)
                            .toCompletableFuture().get(5, TimeUnit.SECONDS);

            assertTrue(result.masterFound());
            assertNull(result.response(), "hop-limit violation must not attempt another RPC");
            assertEquals("FORWARD_HOP_LIMIT", result.failure());
            assertEquals(LIVE_MASTER, result.masterHost());
            verify(fixture.engineHealthReporter).reportForwardToMasterResult(
                    "10.0.0.9", "HOP_LIMIT");
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    @DisplayName("state ② real chain: forwarding to a dead master address yields UNAVAILABLE terminal failure")
    void guardDeadMasterAddressYieldsUnavailableFailure() throws Exception {
        try (RealForwarderFixture fixture = newRealForwarderFixture(DEAD_MASTER)) {
            FlexlbGrpcForwarder.MasterForwardResult result =
                    fixture.forwarder.forwardScheduleToMaster(request("90_104"))
                            .toCompletableFuture().get(20, TimeUnit.SECONDS);

            assertTrue(result.masterFound(), "a master address was selected");
            assertNull(result.response(), "dead master cannot answer");
            assertEquals("UNAVAILABLE", result.failure(),
                    "transport failure to the dead master must surface as UNAVAILABLE");
            assertEquals(DEAD_MASTER, result.masterHost());
            verify(fixture.engineHealthReporter).reportForwardToMasterResult(
                    "127.0.0.1", "GRPC_FAILED");
        }
    }

    // ------------------------------------------------------------------
    // fixtures and helpers
    // ------------------------------------------------------------------

    /** Real forwarder bound to a mocked consistency view; auto-closed. */
    private final class RealForwarderFixture implements AutoCloseable {
        final FlexlbGrpcForwarder forwarder;
        final EngineHealthReporter engineHealthReporter;

        private RealForwarderFixture(EngineHealthReporter engineHealthReporter,
                                     FlexlbGrpcForwarder forwarder) {
            this.engineHealthReporter = engineHealthReporter;
            this.forwarder = forwarder;
        }

        @Override
        public void close() {
            forwarder.shutdown();
        }
    }

    private RealForwarderFixture newRealForwarderFixture(String masterHostIpPort) {
        LBStatusConsistencyService consistencyView = mock(LBStatusConsistencyService.class);
        when(consistencyView.isNeedConsistency()).thenReturn(true);
        when(consistencyView.isMaster()).thenReturn(false);
        when(consistencyView.getLocalHostIp()).thenReturn(FOLLOWER_IP);
        // Tier-2 thenAnswer injection: the follower's cached master view is
        // exactly what the guard consumes in production.
        when(consistencyView.getMasterHostIpPort()).thenReturn(masterHostIpPort);

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(new FlexlbConfig());
        EngineHealthReporter healthReporter = mock(EngineHealthReporter.class);

        channelEventLoop = new NioEventLoopGroup(1);
        channelExecutor = Executors.newFixedThreadPool(2);
        FlexlbGrpcForwarder forwarder = new FlexlbGrpcForwarder(
                consistencyView, configService, healthReporter,
                channelEventLoop, channelExecutor);
        return new RealForwarderFixture(healthReporter, forwarder);
    }

    private static FlexlbScheduleProtocol.FlexlbScheduleRequestPB request(String requestId) {
        return FlexlbScheduleProtocol.FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(requestId)
                .setSeqLen(1024)
                .build();
    }

    private void stubSuccessfulLocalRoute() {
        Response response = new Response();
        response.setSuccess(true);
        response.setCode(200);
        when(routeService.route(any())).thenReturn(
                CompletableFuture.completedFuture(response));
    }

    private FlexlbScheduleProtocol.FlexlbScheduleResponsePB capturedResponse(
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
        org.mockito.ArgumentCaptor<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> captor =
                org.mockito.ArgumentCaptor.forClass(
                        FlexlbScheduleProtocol.FlexlbScheduleResponsePB.class);
        verify(observer, times(1)).onNext(captor.capture());
        verify(observer, times(1)).onCompleted();
        return captor.getValue();
    }

    private void assertSuccessfulResponse(
            StreamObserver<FlexlbScheduleProtocol.FlexlbScheduleResponsePB> observer) {
        FlexlbScheduleProtocol.FlexlbScheduleResponsePB response = capturedResponse(observer);
        assertTrue(response.getSuccess());
        assertEquals(200, response.getCode());
    }

    private void assertSinglePvContains(String expected) {
        assertEquals(1, pvAppender.list.size());
        assertTrue(pvAppender.list.get(0).getFormattedMessage().contains(expected),
                pvAppender.list.get(0).getFormattedMessage());
    }
}
