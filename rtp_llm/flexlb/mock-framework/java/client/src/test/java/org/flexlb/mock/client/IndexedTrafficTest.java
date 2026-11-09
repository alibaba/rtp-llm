package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.flexlb.mock.client.ClientProtocol.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class IndexedTrafficTest {
    @TempDir Path directory;

    private Path fixture() throws Exception { return fixture(true); }

    private Path fixture(boolean image) throws Exception {
        var rows = new ArrayList<ByteBuffer>();
        for (int i = 0; i < 5; i++) {
            long leaf = i == 2 ? 2 : i % 2 == 0 ? 4 : 6;
            var row = ByteBuffer.allocate(i == 0 && image ? 42 : 26);
            row.putLong(i * 10).putLong(leaf).putInt(-1).putInt(50).putShort((short) 1);
            if (i == 0 && image) row.putLong(1).putInt(2).putInt(924);
            rows.add(row);
        }
        var chains = ByteBuffer.allocate(40).putLong(1).putLong(0).putInt(4).putLong(5).putLong(2).putInt(2);
        return InputFixture.indexed(directory.resolve("traffic.mckidx"), rows, chains, 40, List.of(new int[]{1024, 768}), List.of(3), 0);
    }

    @Test void reconstructsSharedTextAcrossAnInteriorBranchAndInsertsMedia() throws Exception {
        try (var dataset = new IndexedTraffic(fixture())) {
            var image = dataset.entry(0, 0);
            var branch = dataset.entry(1, 0).input();
            var shortPrefix = dataset.entry(2, 0).input();
            assertEquals(List.of(2, 1, 0, 1, 1), image.input().getTokenIdsList());
            assertEquals(List.of(2, 1, 3, 1), branch.getTokenIdsList());
            assertEquals(List.of(2, 1), shortPrefix.getTokenIdsList());
            assertEquals("mock-image:1", image.input().getMultimodalInputs(0).getMultimodalUrl());
            assertEquals(List.of(924), image.expectedImageTokens());
        }
    }

    @Test void playerTouchesOnlyItsOrdinalsAndUsesTheirOriginalTimes() throws Exception {
        var path = fixture();
        String identity;
        try (var dataset = new IndexedTraffic(path)) { identity = dataset.identity; }
        byte[] data = Files.readAllBytes(path);
        // Corrupt only ordinal 0. A player assigned to 1 and 3 must still finish.
        data[Math.toIntExact(ByteBuffer.wrap(data).getLong(56))] ^= 1;
        Files.write(path, data);
        var odd = new WorkerPlan(1, null, path.toString(), 2, 1, 2, 100, identity, null, 2.0, 1, 4, 1000, 1000);
        var items = new ArrayList<LoadClient.Item>();
        long past = System.nanoTime() - 1_000_000_000;
        LoadClient.play(odd, past, items::add);
        assertEquals(List.of(101L, 103L), items.stream().map(LoadClient.Item::requestId).toList());
        assertEquals(List.of(5_000_000L, 15_000_000L), items.stream().map(item -> {
            try { return LoadClient.delayNanos(item.requestId() - 100, item.entry().arrivalMs(), null, 2.0); }
            catch (IOException error) { throw new AssertionError(error); }
        }).toList());
        assertEquals(10_000_000L, LoadClient.delayNanos(1, items.getFirst().entry().arrivalMs(), 100.0, null));
        var even = new WorkerPlan(1, null, path.toString(), 3, 0, 2, 100, identity, null, 2.0, 1, 4, 1000, 1000);
        assertThrows(IOException.class, () -> LoadClient.play(even, past, ignored -> fail()));
    }

    @Test void fixedTotalQpsInterleavesWorkerDeadlines() throws Exception {
        var path = fixture();
        try (var input = new IndexedTraffic(path)) {
            long past = System.nanoTime() - 1_000_000_000;
            var expected = List.of(List.of(0L, 20_000_000L, 40_000_000L), List.of(10_000_000L, 30_000_000L));
            for (int worker = 0; worker < 2; worker++) {
                var plan = new WorkerPlan(1, null, path.toString(), expected.get(worker).size(), worker, 2,
                    100, input.identity, 100.0, null, 1, 4, 1000, 1000);
                var times = new ArrayList<Long>();
                LoadClient.play(plan, past, item -> times.add(assertDoesNotThrow(() ->
                    LoadClient.delayNanos(item.requestId() - 100, item.entry().arrivalMs(), plan.qps(), null))));
                assertEquals(expected.get(worker), times);
            }
        }
    }

    @Test void outputLengthsStayIndependentAndFallbackIsOnlyForMissingValues() throws Exception {
        var rows = new ArrayList<ByteBuffer>();
        rows.add(ByteBuffer.allocate(26).putLong(0).putLong(1).putInt(37).putInt(50).putShort((short) 1));
        rows.add(ByteBuffer.allocate(26).putLong(1).putLong(1).putInt(-1).putInt(50).putShort((short) 1));
        rows.add(ByteBuffer.allocate(26).putLong(2).putLong(1).putInt(9).putInt(50).putShort((short) 0));
        rows.add(ByteBuffer.allocate(26).putLong(3).putLong(1).putInt(-1).putInt(50).putShort((short) 0));
        rows.add(ByteBuffer.allocate(26).putLong(4).putLong(1).putInt(0).putInt(50).putShort((short) 0));
        var chain = ByteBuffer.allocate(20).putLong(1).putLong(0).putInt(1);
        var path = InputFixture.indexed(directory.resolve("lengths.mckidx"), rows, chain, 4, List.of(), List.of(8192), 1);
        try (var dataset = new IndexedTraffic(path)) {
            assertEquals(1, dataset.missingOutput);
            var observed = dataset.entry(0, 1000);
            assertEquals(8192, observed.maxOutputTokens());
            assertEquals(37, observed.playbackLimit(6));
            assertEquals(8192, dataset.entry(1, 1000).playbackLimit(6));
            assertEquals(9, dataset.entry(2, 1000).playbackLimit(null));
            var unknown = dataset.entry(3, 1000);
            assertThrows(IOException.class, () -> unknown.playbackLimit(null));
            assertEquals(6, unknown.playbackLimit(6));
            var zero = dataset.entry(4, 1000);
            assertEquals(0, zero.observedOutputTokens());
            assertThrows(IOException.class, () -> zero.playbackLimit(6));
        }
        try (var client = new ClientNode("client-0", directory.resolve("topology.json"), directory.resolve("worker"))) {
            new ObjectMapper().writeValue(directory.resolve("topology.json").toFile(),
                List.of(Map.of("role", "client", "control", client.url())));
            var error = assertThrows(IOException.class, () ->
                client.start(new Start(null, path.toString(), 1.0, null, 1, 4, 1000, 1000)));
            assertTrue(error.getMessage().contains("fallback_max_output_tokens"));
            assertFalse(client.stop(new Empty()).errors().isEmpty());
            assertFalse(Files.exists(directory.resolve("worker/playback")));
        }
    }

    @Test void publicStartUsesOneSharedFileWithoutWritingShards() throws Exception {
        var source = fixture(false);
        var topology = directory.resolve("topology.json");
        var zeroDirectory = directory.resolve("zero");
        var oneDirectory = directory.resolve("one");
        var json = new ObjectMapper();
        try (var backend = new NativeClientTest.Harness();
             var zero = new ClientNode("client-0", topology, zeroDirectory);
             var one = new ClientNode("client-1", topology, oneDirectory)) {
            json.writeValue(topology.toFile(), List.of(backend.endpoint(),
                Map.of("role", "client", "control", zero.url()), Map.of("role", "client", "control", one.url())));
            zero.start(new Start(List.of(PlaybackTest.master(backend)), source.toString(), null, 2.0, 4, 8, 3000, 1000));
            PlaybackTest.eventually(() -> zero.state(new Sequence(1)).done() && one.state(new Sequence(1)).done());
            var summary = zero.stop(new Empty());
            assertTrue(summary.errors().isEmpty(), summary.toString());
            assertEquals(3, zero.state(new Sequence(1)).issued());
            assertEquals(2, one.state(new Sequence(1)).issued());
            assertEquals(5, backend.schedules.get());
            assertFalse(Files.exists(zeroDirectory.resolve("coordinator/1/shards")));
            for (var directory : List.of(zeroDirectory, oneDirectory)) {
                var plan = json.readTree(directory.resolve("playback/1/plan.json").toFile());
                assertEquals(source.toString(), plan.get("requests_file").asText());
            }
        }
    }
}
