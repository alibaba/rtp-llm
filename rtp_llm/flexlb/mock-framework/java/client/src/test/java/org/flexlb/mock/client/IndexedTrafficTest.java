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
            var row = ByteBuffer.allocate(i == 0 && image ? 40 : 24);
            row.putLong(i * 10).putLong(leaf).putInt(3).putInt(50);
            if (i == 0 && image) row.putLong(1).putInt(2).putInt(924);
            rows.add(row);
        }
        var chains = ByteBuffer.allocate(40).putLong(1).putLong(0).putInt(4).putLong(5).putLong(2).putInt(2);
        return InputFixture.indexed(directory.resolve("traffic.mckidx"), rows, chains, 40, List.of(new int[]{1024, 768}));
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
        data[96 + 5 * 16 + 28 + 12] ^= 1;
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
