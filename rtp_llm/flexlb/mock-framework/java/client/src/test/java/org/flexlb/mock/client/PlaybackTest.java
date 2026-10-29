package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.flexlb.mock.client.ClientProtocol.*;

class PlaybackTest {
    @TempDir Path directory;
    static org.flexlb.mock.client.input.ClientInput.Request input() { return InputFixture.request(); }
    Shard shard(String master, int count, double qps, int slots, int queue) throws Exception {
        var start = InputFixture.plan(directory, List.of(master), count, qps, slots, queue, 3000, 1000);
        var part = RequestFiles.split(start, directory.resolve(UUID.randomUUID().toString()), 1, () -> false).getFirst();
        return new Shard(1, start.masters(), part.path(), part.count(), 0, 1, 1, part.sha256(), qps, slots, queue, 3000, 1000);
    }
    Path topology(NativeClientTest.Harness h) throws Exception {
        var path = directory.resolve(UUID.randomUUID() + ".json");
        new ObjectMapper().writeValue(path.toFile(), List.of(h.endpoint())); return path;
    }
    static String master(NativeClientTest.Harness h) { return "127.0.0.1:" + h.master.getPort(); }
    static void eventually(java.util.function.BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(8);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) Thread.sleep(10);
        assertTrue(condition.getAsBoolean(), "Condition did not become true");
    }
    @Test void naturalEndDrainsQueueAndConsumesRealNativeResponses() throws Exception {
        try (var h = new NativeClientTest.Harness(); var load = new LoadClient(shard(master(h), 3, 10, 2, 4), new ClientMetrics(), topology(h), directory.resolve(UUID.randomUUID() + ".binpb.zst"), "client-0")) {
            load.begin(System.currentTimeMillis() + 30);
            var result = load.await(5000);
            assertTrue(result.done()); assertFalse(result.stopped()); assertTrue(result.errors().isEmpty(), result.toString());
            assertEquals(3, result.issued()); assertEquals(3, result.finished()); assertEquals(0, result.unsent());
            assertEquals(3, h.schedules.get());
        }
    }
    @Test void fullQueueFailsInsteadOfBlockingDroppingOrExceedingConcurrency() throws Exception {
        try (var h = new NativeClientTest.Harness(); var load = new LoadClient(shard(master(h), 30, 500, 1, 1), new ClientMetrics(), topology(h), directory.resolve(UUID.randomUUID() + ".binpb.zst"), "client-0")) {
            h.tokenDelay = 300;
            load.begin(System.currentTimeMillis() + 30);
            var result = load.await(5000);
            assertTrue(result.errors().getFirst().contains("queue full"));
            assertTrue(result.issued() <= 1); assertEquals(result.issued(), result.finished());
            assertEquals(30 - result.issued(), result.unsent());
        }
    }
    @Test void stopBeforeBeginDoesNotSendAndIsIdempotent() throws Exception {
        try (var h = new NativeClientTest.Harness(); var load = new LoadClient(shard(master(h), 3, 10, 1, 2), new ClientMetrics(), topology(h), directory.resolve(UUID.randomUUID() + ".binpb.zst"), "client-0")) {
            load.halt(); load.halt();
            var result = load.await(5000);
            assertEquals(0, result.issued()); assertEquals(3, result.unsent()); assertTrue(result.errors().isEmpty());
            assertThrows(IllegalStateException.class, () -> load.begin(System.currentTimeMillis() + 30));
        }
    }
    @Test void stopClosesAdmissionAndDrainTimeoutCancelsInflight() throws Exception {
        try (var h = new NativeClientTest.Harness()) {
            h.tokenDelay = 5000;
            var plan = shard(master(h), 10, 20, 1, 10);
            plan = new Shard(1, plan.masters(), plan.requests_file(), plan.count(), 0, 1, 1, plan.sha256(), plan.qps(), 1, 10, 10000, 80);
            try (var load = new LoadClient(plan, new ClientMetrics(), topology(h), directory.resolve(UUID.randomUUID() + ".binpb.zst"), "client-0")) {
                load.begin(System.currentTimeMillis() + 30);
                assertTrue(h.scheduleEntered.await(2, TimeUnit.SECONDS));
                load.halt(); long issued = load.state().issued();
                var result = load.await(5000);
                assertEquals(issued, result.issued()); assertEquals(issued, result.finished());
                assertEquals(0, result.inflight()); assertTrue(result.errors().getFirst().contains("drain timed out"));
            }
        }
    }
    @Test void businessErrorsAreRecordedWithoutFailingPlayback() throws Exception {
        try (var h = new NativeClientTest.Harness()) {
            h.route = r -> r.toBuilder().setSuccess(false).setCode(429).clearLifecycle().build();
            var metrics = new ClientMetrics();
            try (var load = new LoadClient(shard(master(h), 3, 10, 1, 4), metrics, topology(h), directory.resolve(UUID.randomUUID() + ".binpb.zst"), "client-0")) {
                load.begin(System.currentTimeMillis() + 30);
                var result = load.await(5000);
                assertEquals(3, result.finished()); assertTrue(result.errors().isEmpty());
                assertTrue(metrics.scrape().contains("mock_client_requests_finished_total{role=\"client\",outcome=\"error\",} 3.0"));
            }
        }
    }
    @Test void threeClientsSplitGlobalCountAndUseTcpEvenForClientZero() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var h = new NativeClientTest.Harness();
             var zero = new ClientNode("client-0", topology, directory.resolve("zero"));
             var one = new ClientNode("client-1", topology, directory.resolve("one"));
             var two = new ClientNode("client-2", topology, directory.resolve("two"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "client", "control", zero.url()), Map.of("role", "client", "control", one.url()), Map.of("role", "client", "control", two.url())));
            var start = InputFixture.plan(directory, null, 7, 10, 4, 4, 3000, 1000);
            assertThrows(IllegalStateException.class, () -> one.start(start));
            zero.start(start);
            assertThrows(IllegalStateException.class, () -> zero.start(start));
            eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done() && two.state(new Sequence(1)).done());
            var result = zero.stop(new Empty());
            assertTrue(result.errors().isEmpty(), result.toString()); assertEquals(3, result.clients().size());
            assertEquals(7, result.clients().values().stream().mapToLong(State::issued).sum());
            assertEquals(7, h.schedules.get()); assertEquals(result, zero.stop(new Empty()));
            zero.start(start);
            assertThrows(IllegalStateException.class, () -> one.halt(new Sequence(1)));
            zero.stop(new Empty());
            assertTrue(Files.exists(directory.resolve("one/playback/1/result.json")));
            assertTrue(Files.exists(directory.resolve("zero/coordinator/2/result.json")));
        }
    }
    @Test void oneWorkerOverflowStopsWholeGroupAndCannotBecomeSuccessfulSummary() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var h = new NativeClientTest.Harness();
             var zero = new ClientNode("client-0", topology, directory.resolve("zero"));
             var one = new ClientNode("client-1", topology, directory.resolve("one"))) {
            h.tokenDelay = 500;
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "client", "control", zero.url()), Map.of("role", "client", "control", one.url())));
            zero.start(InputFixture.plan(directory, List.of(master(h)), 100, 500, 2, 1, 3000, 1000));
            eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done());
            var result = zero.stop(new Empty());
            assertFalse(result.errors().isEmpty()); assertTrue(result.clients().values().stream().allMatch(State::done));
            assertTrue(result.clients().values().stream().mapToLong(State::issued).sum() < 100);
        }
    }
    @Test void unreachablePeerStopsTheRemainingClientAndKeepsMissingResultsVisible() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var h = new NativeClientTest.Harness();
             var zero = new ClientNode("client-0", topology, directory.resolve("zero"));
             var one = new ClientNode("client-1", topology, directory.resolve("one"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "client", "control", zero.url()), Map.of("role", "client", "control", one.url())));
            zero.start(InputFixture.plan(directory, List.of(master(h)), 20, 2, 2, 4, 3000, 1000));
            one.close();
            eventually(() -> zero.state(new Sequence(1)).done());
            var result = zero.stop(new Empty());
            assertFalse(result.errors().isEmpty());
            assertTrue(zero.state(new Sequence(1)).stopped());
            assertTrue(zero.state(new Sequence(1)).issued() < 10);
        }
    }
    @Test void malformedFileFailsBeforeAnyWorkerStarts() throws Exception {
        Path topology = directory.resolve("topology.json"), input = directory.resolve("bad.binpb.zst");
        Files.writeString(input, "{\"token_ids\":[1],\"max_new_tokens\":2}\nnot-json\n");
        try (var h = new NativeClientTest.Harness(); var zero = new ClientNode("client-0", topology, directory.resolve("zero"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(), Map.of("role", "client", "control", zero.url())));
            assertThrows(java.io.IOException.class, () -> zero.start(new Start(null, input.toString(), 10, 1, 4, 3000, 1000)));
            assertFalse(zero.stop(new Empty()).errors().isEmpty());
            assertEquals(0, h.schedules.get());
            assertFalse(Files.exists(directory.resolve("zero/playback/1/plan.json")));
        }
    }
    @Test void truncatedPreparedShardStopsTheWholeGroup() throws Exception {
        Path topology = directory.resolve("topology.json"), input = directory.resolve("input.binpb.zst");
        InputFixture.write(input, Collections.nCopies(20, input().toBuilder().clearTokenIds()
            .addAllTokenIds(java.util.stream.IntStream.range(0, 100000).boxed().toList()).build()));
        try (var h = new NativeClientTest.Harness();
             var zero = new ClientNode("client-0", topology, directory.resolve("zero"));
             var one = new ClientNode("client-1", topology, directory.resolve("one"))) {
            new ObjectMapper().writeValue(topology.toFile(), List.of(h.endpoint(),
                Map.of("role", "client", "control", zero.url()), Map.of("role", "client", "control", one.url())));
            zero.start(new Start(null, input.toString(), 2, 2, 4, 3000, 1000));
            // Each shard spans multiple batches, so truncation fails before all requests are issued.
            Files.writeString(directory.resolve("zero/coordinator/1/shards/1.binpb.zst"), "");
            eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done());
            var result = zero.stop(new Empty());
            assertFalse(result.errors().isEmpty());
            assertTrue(result.clients().values().stream().allMatch(State::stopped));
            assertTrue(result.clients().values().stream().mapToLong(State::issued).sum() < 20);
        }
    }
    @Test void impossiblePlansAreRejectedBeforeThreadsStart() {
        assertThrows(IllegalArgumentException.class, () -> new Start(List.of("m:1"), "/input.binpb.zst", Double.NaN, 1, 1, 100, 100));
        assertThrows(IllegalArgumentException.class, () -> new Start(List.of("m:1"), "/input.binpb.zst", 0, 1, 1, 100, 100));
        assertThrows(IllegalArgumentException.class, () -> new Start(null, null, 1, 1, 1, 100, 100));
    }
}
