package org.flexlb.mock.engine.cache;

import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.HttpServer;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.stub.StreamObserver;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.BooleanSupplier;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.flexlb.kvcm.grpc.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Exercises direct and discovered KVCM endpoints, including leader changes and cancellation.
 */
class KvcmDiscoveryTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    @Test void unreachableAndInvalidCandidatesDoNotHideAnAvailableLeader() throws Exception {
        try (var invalid = new Meta(); var healthy = new Meta()) {
            invalid.valid = false;
            int unused;
            try (var socket = new ServerSocket(0)) { unused = socket.getLocalPort(); }
            var vip = new Vip(); vip.hosts.set(List.of(WorkerHost.of("127.0.0.1", unused, ""), invalid.host(), healthy.host()));
            try (var resolver = new KvcmDiscovery(null, "kvcm-domain", "shared_128", vip)) {
                assertEquals(healthy.endpoint(), resolver.get());
                assertEquals("shared_128", healthy.instance);
                assertEquals(1, invalid.lookups.get());
            }
            assertTrue(vip.closed);
        }
    }

    @Test void successfulLeaderStaysCachedUntilItsOwnEndpointFails() throws Exception {
        try (var first = new Meta(); var second = new Meta()) {
            var vip = new Vip(); vip.hosts.set(List.of(first.host()));
            try (var resolver = new KvcmDiscovery(null, "kvcm-domain", "shared_128", vip)) {
                URI old = resolver.get();
                vip.hosts.set(List.of(second.host()));
                Thread.sleep(1100); // Regression: the removed one-second TTL must not trigger another lookup.
                assertEquals(old, resolver.get()); assertEquals(1, first.lookups.get()); assertEquals(0, second.lookups.get());
                resolver.invalidate(old);
                assertEquals(second.endpoint(), resolver.get());
                resolver.invalidate(old); // Late failure from an older endpoint cannot clear the current leader.
                assertEquals(second.endpoint(), resolver.get()); assertEquals(1, second.lookups.get());
            }
        }
    }

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void httpFailureOrNotLeaderRediscoversAndRepairsTheSnapshot(boolean notLeader) throws Exception {
        try (var first = new Meta(); var second = new Meta(); var cache = new KvCacheManager(1, 4, 0)) {
            var vip = new Vip(); vip.hosts.set(List.of(first.host()));
            var resolver = new KvcmDiscovery(null, "kvcm-domain", "shared_128", vip);
            try (var lease = cache.acquire(1, List.of()).lease()) { lease.publish(List.of(99L), 1); }
            try (var reporter = new KvcmReporter(new KvcmReporter.Configuration(resolver, true, 600000),
                    new KvcmReporter.Identity("shared_128", "group", "127.0.0.1:18080", "full", "FP16", false, 1, 1, "hbm"), cache)) {
                try { await(() -> first.snapshots.get() > 0); }
                catch (AssertionError error) { throw new AssertionError("Initial reporter: " + reporter.status()
                    + ", lookups=" + first.lookups + ", registrations=" + first.registrations, error); }
                vip.hosts.set(List.of(second.host())); first.notLeader = notLeader; first.failHttp = true;
                try { await(() -> second.snapshots.get() > 0); }
                catch (AssertionError error) { throw new AssertionError("Recovery reporter: " + reporter.status()
                    + ", endpoint=" + resolver.current() + ", lookups=" + first.lookups + "/" + second.lookups
                    + ", registrations=" + second.registrations, error); }
                assertEquals(1, first.lookups.get()); assertEquals(1, second.lookups.get());
                assertEquals(1, second.snapshot.get().path("key_count").asInt());
                assertEquals("99", second.snapshot.get().path("blocks").get(0).path("block_key").asText());
                assertTimeoutPreemptively(java.time.Duration.ofSeconds(3), reporter::close);
            }
            assertTrue(vip.closed);
            assertThrows(IllegalStateException.class, resolver::get);
        }
    }

    @Test void directOverrideIsFixedAndValidated() {
        URI direct = URI.create("http://127.0.0.1:6382");
        try (var resolver = KvcmDiscovery.direct(direct)) {
            assertEquals(direct, resolver.get()); resolver.invalidate(direct); assertEquals(direct, resolver.get());
        }
        assertThrows(IllegalArgumentException.class, () -> KvcmDiscovery.direct(URI.create("file:///tmp/not-http")));
        assertThrows(IllegalArgumentException.class, () -> KvcmDiscovery.discover("", "shared_128", false));
    }

    @Test void closeCancelsBlockedVipLookupWithoutWaitingForItsMonitor() throws Exception {
        var entered = new CountDownLatch(1); var interrupted = new CountDownLatch(1);
        var vip = new Vip() {
            @Override public List<WorkerHost> getHosts(String domain) {
                entered.countDown();
                try { new CountDownLatch(1).await(); }
                catch (InterruptedException expected) { interrupted.countDown(); Thread.currentThread().interrupt(); }
                return List.of();
            }
        };
        var resolver = new KvcmDiscovery(null, "kvcm-domain", "shared_128", vip);
        var pending = CompletableFuture.supplyAsync(resolver::get);
        try {
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(1), resolver::close);
            assertThrows(ExecutionException.class, () -> pending.get(2, TimeUnit.SECONDS));
            assertTrue(interrupted.await(2, TimeUnit.SECONDS)); assertTrue(vip.closed);
        } finally { resolver.close(); }
    }

    @Test void closeCancelsAnOutstandingGrpcLookup() throws Exception {
        var entered = new CountDownLatch(1);
        var rpc = ServerBuilder.forPort(0).addService(new MetaServiceGrpc.MetaServiceImplBase() {
            @Override public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> out) {
                entered.countDown(); // Deliberately never complete this RPC.
            }
        }).build().start();
        var resolver = KvcmDiscovery.discover("127.0.0.1:" + rpc.getPort(), "shared_128", false);
        var pending = CompletableFuture.supplyAsync(resolver::get);
        try {
            assertTrue(entered.await(2, TimeUnit.SECONDS));
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(1), resolver::close);
            assertThrows(ExecutionException.class, () -> pending.get(2, TimeUnit.SECONDS));
        } finally { resolver.close(); rpc.shutdownNow().awaitTermination(2, TimeUnit.SECONDS); }
    }

    private static void await(BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(8);
        while (!condition.getAsBoolean()) {
            assertTrue(System.nanoTime() < deadline, "Condition timed out"); Thread.sleep(10);
        }
    }

    /**
     * Supplies bootstrap hosts and exposes discovery shutdown to the tests.
     */
    static class Vip implements ServiceDiscovery {
        final AtomicReference<List<WorkerHost>> hosts = new AtomicReference<>(List.of());
        boolean closed;
        public List<WorkerHost> getHosts(String domain) { assertEquals("kvcm-domain", domain); return hosts.get(); }
        public void listen(String domain, ServiceHostListener listener) { fail("No watcher is needed"); }
        public void shutdown() { closed = true; }
    }

    /**
     * Serves test Meta gRPC and HTTP endpoints to exercise leader discovery and reports.
     */
    static final class Meta implements AutoCloseable {
        final Server rpc;
        final HttpServer http;
        final AtomicInteger lookups = new AtomicInteger(), registrations = new AtomicInteger(), snapshots = new AtomicInteger();
        final AtomicReference<JsonNode> snapshot = new AtomicReference<>();

        volatile boolean valid = true, failHttp, notLeader;

        volatile String instance;

        Meta() throws Exception {
            http = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            http.createContext("/api/", exchange -> {
                try (exchange) {
                    var body = JSON.readTree(exchange.getRequestBody());
                    boolean registration = exchange.getRequestURI().getPath().equals("/api/registerInstance");
                    int status = failHttp && !notLeader ? 503 : 200;
                    int code = failHttp ? 9 : registration || registrations.get() > 0 ? 1 : 10;
                    if (status == 200 && code == 1) {
                        if (registration) registrations.incrementAndGet();
                        for (var event : body.path("events")) if (event.has("block_snapshot")) {
                            snapshot.set(event.path("block_snapshot")); snapshots.incrementAndGet();
                        }
                    }
                    byte[] response = ("{\"header\":{\"status\":{\"code\":" + code + "}}}").getBytes();
                    exchange.sendResponseHeaders(status, response.length); exchange.getResponseBody().write(response);
                }
            }); http.start();
            rpc = ServerBuilder.forPort(0).addService(new MetaServiceGrpc.MetaServiceImplBase() {
                @Override public void getClusterInfo(GetClusterInfoRequest request, StreamObserver<GetClusterInfoResponse> out) {
                    instance = request.getInstanceId(); lookups.incrementAndGet();
                    out.onNext(GetClusterInfoResponse.newBuilder()
                        .setHeader(CommonResponseHeader.newBuilder().setStatus(Status.newBuilder().setCode(ErrorCode.OK)))
                        .setLeaderNodeId("leader")
                        .setLeaderEndpoint(MetaNodeEndpoint.newBuilder().setNodeId(valid ? "leader" : "old-leader")
                            .setHost("127.0.0.1").setMetaHttpPort(http.getAddress().getPort())).build());
                    out.onCompleted();
                }
            }).build().start();
        }

        @Override public void close() throws Exception { http.stop(0); rpc.shutdownNow().awaitTermination(3, TimeUnit.SECONDS); }

        WorkerHost host() { return WorkerHost.of("127.0.0.1", rpc.getPort(), ""); }

        URI endpoint() { return URI.create("http://127.0.0.1:" + http.getAddress().getPort()); }
    }
}
