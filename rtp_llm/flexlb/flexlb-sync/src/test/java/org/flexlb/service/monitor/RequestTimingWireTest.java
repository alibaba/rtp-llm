package org.flexlb.service.monitor;

import io.grpc.Server;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelServiceConfiguration;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.discovery.StaticServiceDiscoveryProvider;
import org.flexlb.engine.grpc.client.RequestTimingClient;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.config.parser.ModelServiceConfigParser;
import org.flexlb.timing.grpc.PullRequestTimingsRequest;
import org.flexlb.timing.grpc.PullRequestTimingsResponse;
import org.flexlb.timing.grpc.RequestTimingEventKind;
import org.flexlb.timing.grpc.RequestTimingRecord;
import org.flexlb.timing.grpc.RequestTimingServiceGrpc;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestTimingWireTest {
    private static final String PREFIX = "app.frontend.ttft.stages.";
    private static final String FIRST_ATTEMPT = "request#call0123456789abcdef0123456789abcdef";

    @Test
    void failedDispatchIsSeparateFromRetryTtftWhileUnrelatedPrefillTimesOut() throws Exception {
        try (TimingServer frontend = new TimingServer("frontend", false);
             TimingServer prefill = new TimingServer("prefill", false);
             TimingServer unavailable = new TimingServer("prefill", true);
             CollectionFixture fixture = new CollectionFixture(List.of(frontend, prefill, unavailable))) {
            fixture.initialize(frontend, prefill);
            long receivedMs = fixture.clock.get();
            String secondAttempt = FIRST_ATTEMPT + "#attempt1";
            frontend.records = List.of(timing(1, FIRST_ATTEMPT, "frontend", receivedMs * 1000)
                    .setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                    .setFailedAtUs((receivedMs + 30) * 1000)
                    .build(), timing(2, secondAttempt, "frontend", (receivedMs + 35) * 1000).build());
            prefill.records = List.of(
                    timing(1, FIRST_ATTEMPT, "prefill", (receivedMs + 10) * 1000).build(),
                    timing(2, secondAttempt, "prefill", (receivedMs + 40) * 1000).build());
            fixture.tick();
            fixture.awaitSequence(frontend, 2);
            fixture.awaitSequence(prefill, 2);
            TaskInfo failed = task(FIRST_ATTEMPT, receivedMs + 11, receivedMs + 30, 0, 1, 1000, 2000, 3000, 2);
            TaskInfo successful = task(secondAttempt, receivedMs + 41, receivedMs + 90, receivedMs + 80, 0, 4000, 5000, 6000, 3);
            fixture.addWorkers(Map.of(FIRST_ATTEMPT, failed, secondAttempt, successful));
            fixture.clock.addAndGet(1500);
            fixture.tick();

            verify(fixture.monitor).report(eq(PREFIX + "failed.us"), any(), eq(30_000d));
            verify(fixture.monitor).report(eq(PREFIX + "failed.requests"), any(), eq(1d));
            verify(fixture.monitor).report(eq(PREFIX + "total.us"), any(), eq(45_000d));
            verify(fixture.monitor).report(eq(PREFIX + "before.prefill.us"), any(), eq(5000d));
            verify(fixture.monitor).report(eq(PREFIX + "kv.alloc.us"), any(), eq(4000d));
            verify(fixture.monitor).report(eq(PREFIX + "remote.kv.us"), any(), eq(3000d));
            verify(fixture.monitor).report(eq(PREFIX + "scheduler.wait.us"), any(), eq(5000d));
            verify(fixture.monitor).report(eq(PREFIX + "prefill.inflight.us"), any(), eq(6000d));
            verify(fixture.monitor).report(eq(PREFIX + "other.us"), any(), eq(22_000d));

            fixture.awaitIdle(unavailable);
            fixture.tick();
            await(() -> unavailable.requests.get() >= 2, "unavailable P must be retried after its deadline");
            fixture.awaitIdle(unavailable);
            fixture.tick();
            fixture.addWorkers(Map.of(FIRST_ATTEMPT, failed, secondAttempt, successful));
            fixture.clock.addAndGet(1500);
            fixture.tick();
            verify(fixture.monitor, times(1)).report(eq(PREFIX + "requests"), any(), eq(1d));
            verify(fixture.monitor, times(1)).report(eq(PREFIX + "total.us"), any(), anyDouble());
            verify(fixture.monitor, times(1)).report(eq(PREFIX + "failed.requests"), any(), eq(1d));
            verify(fixture.monitor, times(1)).report(eq(PREFIX + "failed.us"), any(), eq(30_000d));
        }
    }

    @Test
    void failedDispatchWithoutPrefillReceiptPublishesOnlyFailureMetricsOnce() throws Exception {
        try (TimingServer frontend = new TimingServer("frontend", false);
                TimingServer prefill = new TimingServer("prefill", false);
                CollectionFixture fixture = new CollectionFixture(List.of(frontend, prefill))) {
            fixture.initialize(frontend);
            long dispatchMs = fixture.clock.get();
            frontend.records = List.of(timing(1, FIRST_ATTEMPT, "frontend", dispatchMs * 1000)
                    .setEventKind(RequestTimingEventKind.DISPATCH_FAILED)
                    .setFailedAtUs((dispatchMs + 30) * 1000)
                    .build());
            fixture.tick();
            fixture.awaitSequence(frontend, 1);
            fixture.clock.addAndGet(1500);
            fixture.tick();
            verify(fixture.monitor, never()).report(eq(PREFIX + "failed.us"), any(), anyDouble());
            verify(fixture.monitor, never()).report(eq(PREFIX + "failed.requests"), any(), anyDouble());
            verify(fixture.monitor, never()).report(eq(PREFIX + "requests"), any(), anyDouble());
            verify(fixture.monitor, never()).report(eq(PREFIX + "total.us"), any(), anyDouble());

            frontend.records =
                    List.of(frontend.records.get(0), frontend.records.get(0).toBuilder().setSequence(2).build());
            fixture.tick();
            fixture.awaitSequence(frontend, 2);
            fixture.clock.addAndGet(120_000);
            fixture.tick();
            verify(fixture.monitor, times(1)).report(eq(PREFIX + "failed.us"), any(), eq(30_000d));
            verify(fixture.monitor, times(1)).report(eq(PREFIX + "failed.requests"), any(), eq(1d));
            verify(fixture.monitor, never()).report(eq(PREFIX + "requests"), any(), anyDouble());
            verify(fixture.monitor, never()).report(eq(PREFIX + "total.us"), any(), anyDouble());
        }
    }

    @Test
    void standaloneCollectorPublishesOnceAfterMissingPrefillRecordArrives() throws Exception {
        try (TimingServer frontend = new TimingServer("frontend", false);
             TimingServer prefill = new TimingServer("prefill", true);
             CollectionFixture fixture = new CollectionFixture(List.of(frontend, prefill))) {
            fixture.initialize(frontend);
            long receivedMs = fixture.clock.get();
            frontend.records = List.of(timing(1, FIRST_ATTEMPT, "frontend", receivedMs * 1000).build());
            fixture.tick();
            fixture.awaitSequence(frontend, 1);
            TaskInfo successful = task(FIRST_ATTEMPT, receivedMs + 11, receivedMs + 90, receivedMs + 80,
                    0, 4000, 5000, 6000, 3);
            fixture.addWorkers(Map.of(FIRST_ATTEMPT, successful));
            fixture.clock.addAndGet(1500);
            fixture.tick();
            verify(fixture.monitor, never()).report(eq(PREFIX + "requests"), any(), anyDouble());

            prefill.records = List.of(timing(1, FIRST_ATTEMPT, "prefill", (receivedMs + 10) * 1000).build());
            prefill.unavailable = false;
            fixture.awaitIdle(prefill);
            fixture.tick();
            fixture.awaitSequence(prefill, 1);
            verify(fixture.monitor, never()).report(eq(PREFIX + "requests"), any(), anyDouble());
            fixture.clock.addAndGet(1500);
            fixture.tick();
            verify(fixture.monitor).report(eq(PREFIX + "total.us"), any(), eq(80_000d));
            verify(fixture.monitor).report(eq(PREFIX + "requests"), any(), eq(1d));

            fixture.addWorkers(Map.of(FIRST_ATTEMPT, successful));
            fixture.clock.addAndGet(1500);
            fixture.tick();
            verify(fixture.monitor, times(1)).report(eq(PREFIX + "requests"), any(), eq(1d));
        }
    }

    private static final class TimingServer implements AutoCloseable {
        private final Server server;
        private final String role;
        private final AtomicInteger requests = new AtomicInteger();
        private volatile List<RequestTimingRecord> records = List.of();
        private volatile boolean unavailable;

        private TimingServer(String role, boolean unavailable) throws Exception {
            this.role = role;
            this.unavailable = unavailable;
            server = NettyServerBuilder.forPort(0).addService(new RequestTimingServiceGrpc.RequestTimingServiceImplBase() {
                @Override
                public void pullRequestTimings(PullRequestTimingsRequest request,
                                               StreamObserver<PullRequestTimingsResponse> output) {
                    requests.incrementAndGet();
                    if (TimingServer.this.unavailable) {
                        return;
                    }
                    List<RequestTimingRecord> snapshot = records;
                    var response = PullRequestTimingsResponse.newBuilder().setEpoch(role + "-epoch")
                            .setNextSequence(snapshot.isEmpty() ? 0 : snapshot.getLast().getSequence());
                    snapshot.stream().filter(record -> record.getSequence() > request.getAfterSequence())
                            .forEach(response::addRecords);
                    output.onNext(response.build());
                    output.onCompleted();
                }
            }).build().start();
        }

        private RequestTimingSettings.Source source() {
            return new RequestTimingSettings.Source(new GrpcTarget("127.0.0.1", server.getPort()), role);
        }

        @Override
        public void close() throws Exception {
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        }
    }

    private static final class CollectionFixture implements AutoCloseable {
        private final FlexMonitor monitor = mock(FlexMonitor.class);
        private final RequestTimingClient client;
        private final RequestTimingPoller poller;
        private final RequestTimingDiscovery discovery;
        private final int daemonCount;
        private final ScheduledThreadPoolExecutor executor;
        private final AtomicLong clock = new AtomicLong();
        private final WorkerStatus worker = WorkerStatus.createDiscovered(RoleType.PREFILL, "group", "127.0.0.1",
                8081, 8082, "site", "p-deployment");

        private CollectionFixture(List<TimingServer> servers) {
            GrpcChannelFactory channels = mock(GrpcChannelFactory.class);
            when(channels.create(any())).thenAnswer(invocation -> {
                GrpcTarget target = invocation.getArgument(0);
                return NettyChannelBuilder.forAddress(target.host(), target.port()).usePlaintext().build();
            });
            client = new RequestTimingClient(channels);
            ConfigService configs = mock(ConfigService.class);
            FlexlbConfig config = new FlexlbConfig();
            config.getObservability().getTtftStages().setEnabled(true);
            config.getObservability().getTtftStages().setRemoteKvBeforeScheduler(true);
            when(configs.loadBalanceConfig()).thenReturn(config);
            LBStatusConsistencyService leadership = mock(LBStatusConsistencyService.class);
            when(leadership.isNeedConsistency()).thenReturn(false);
            when(leadership.isMaster()).thenReturn(false);
            daemonCount = servers.size();
            String groups = servers.stream().map(server -> """
                    {"%s_endpoint":{"address":"daemon-%d","protocol":"grpc",
                     "discovery":{"type":"static-env","hosts":["127.0.0.1:%d"]}}}
                    """.formatted(server.role, server.server.getPort(), server.server.getPort()))
                    .collect(Collectors.joining(","));
            when(configs.modelServiceConfig()).thenReturn(ModelServiceConfigParser.parse("{\"role_endpoints\":[" + groups + "]}"));
            var provider = new RoutingServiceDiscovery(List.of(new StaticServiceDiscoveryProvider()));
            var model = new ModelServiceConfiguration().modelMetaConfig(configs, provider);
            discovery = new RequestTimingDiscovery(model, provider);
            poller = new RequestTimingPoller(configs, leadership, client, monitor, discovery);
            executor = (ScheduledThreadPoolExecutor) ReflectionTestUtils.getField(poller, "executor");
        }

        private void initialize(TimingServer... responding) throws Exception {
            tick();
            await(() -> discovery.sources().size() == daemonCount, "all configured daemons must be discovered");
            tick();
            for (TimingServer server : responding) {
                awaitSequence(server, 0);
            }
            clock.set(System.currentTimeMillis());
            var aggregator = new TtftStageAggregator(monitor, 100, 120_000, 1000, true, clock::get);
            executor.submit(() -> ReflectionTestUtils.setField(poller, "aggregator", aggregator)).get(5, TimeUnit.SECONDS);
        }

        private void tick() throws Exception {
            executor.submit(poller::tick).get(5, TimeUnit.SECONDS);
        }

        private void addWorkers(Map<String, TaskInfo> tasks) throws Exception {
            WorkerStatusResponse response = new WorkerStatusResponse();
            response.setRole(RoleType.PREFILL);
            response.setAlive(true);
            response.setRunningTaskInfo(Map.of());
            response.setFinishedTaskInfo(tasks);
            poller.addWorkerStatus(worker, worker.freezeStatusResponse(response));
            executor.submit(() -> {}).get(5, TimeUnit.SECONDS);
        }

        private void awaitSequence(TimingServer server, long sequence) throws Exception {
            awaitCursor(server, cursor -> !cursor.inFlight && cursor.sequence == sequence && !cursor.epoch.isEmpty());
        }

        private void awaitIdle(TimingServer server) throws Exception {
            awaitCursor(server, cursor -> !cursor.inFlight);
        }

        @SuppressWarnings("unchecked")
        private void awaitCursor(TimingServer server, java.util.function.Predicate<RequestTimingCursor> condition) throws Exception {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (System.nanoTime() < deadline) {
                boolean complete = executor.submit(() -> {
                    Map<RequestTimingSettings.Source, RequestTimingCursor> cursors = (Map<RequestTimingSettings.Source, RequestTimingCursor>)
                            ReflectionTestUtils.getField(poller, "cursors");
                    RequestTimingCursor cursor = cursors.get(server.source());
                    return cursor != null && condition.test(cursor);
                }).get(5, TimeUnit.SECONDS);
                if (complete) {
                    return;
                }
                Thread.sleep(10);
            }
            assertTrue(false, "gRPC source must finish the expected incremental response: " + server.source());
        }

        @Override
        public void close() throws Exception {
            poller.close();
            executor.awaitTermination(5, TimeUnit.SECONDS);
            discovery.shutdown();
            client.close();
        }
    }

    private static RequestTimingRecord.Builder timing(long sequence, String id, String role, long startUs) {
        var record = RequestTimingRecord.newBuilder()
                .setSequence(sequence)
                .setRequestId(id)
                .setRole(role)
                .setDeploymentName("frontend-deployment");
        if ("prefill".equals(role)) {
            return record.setEventKind(RequestTimingEventKind.PREFILL_RECEIVED).setReceivedTimeUs(startUs);
        }
        return record.setEventKind(RequestTimingEventKind.DISPATCH_COMPLETED).setDispatchStartUs(startUs);
    }

    private static TaskInfo task(String id, long receivedMs, long endMs, long firstTokenMs, long errorCode,
                                 long allocation, long scheduler, long inflight, long remoteMs) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(id);
        task.setRequestReceivedTimeMs(receivedMs);
        task.setEndTimeMs(endMs);
        task.setFirstTokenTimeMs(firstTokenMs);
        task.setErrorCode(errorCode);
        task.setHybridKvAllocWaitUs(allocation);
        task.setSchedulerWaitUs(scheduler);
        task.setPrefillInflightWallUs(inflight);
        task.setRemoteKvWaitMs(remoteMs);
        task.setSchedulerWaitForPleReadyUs(0L);
        return task;
    }

    private static void await(BooleanSupplier condition, String message) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            Thread.sleep(10);
        }
        assertTrue(condition.getAsBoolean(), message);
    }
}
