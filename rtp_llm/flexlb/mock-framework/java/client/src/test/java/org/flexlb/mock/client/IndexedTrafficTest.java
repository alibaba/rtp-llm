package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.luben.zstd.ZstdCompressCtx;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.zip.CRC32;
import org.flexlb.mock.client.ClientProtocol.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class IndexedTrafficTest {
    @TempDir Path directory;

    private Path fixture() throws Exception { return fixture(true); }

    private Path fixture(boolean image) throws Exception {
        var rows = new ByteArrayOutputStream();
        var index = new ByteArrayOutputStream();
        for (int i = 0; i < 5; i++) {
            long leaf = i == 2 ? 2 : i % 2 == 0 ? 4 : 6;
            var row = ByteBuffer.allocate(i == 0 && image ? 40 : 24);
            row.putLong(i * 10).putLong(leaf).putInt(3).putInt(50);
            if (i == 0 && image) row.putLong(1).putInt(2).putInt(924);
            record(index, ByteBuffer.allocate(12).putLong(rows.size()).putInt(row.capacity() + 4));
            record(rows, row);
        }
        var tables = new ByteArrayOutputStream();
        index.writeTo(tables);
        var chains = ByteBuffer.allocate(40).putLong(1).putLong(0).putInt(4).putLong(5).putLong(2).putInt(2);
        byte[] page;
        try (var compressor = new ZstdCompressCtx()) {
            page = compressor.setChecksum(true).compress(chains.array());
        }
        record(tables, ByteBuffer.allocate(24).putLong(1).putLong(rows.size()).putInt(page.length).putInt(2));
        record(tables, ByteBuffer.allocate(8).putInt(1024).putInt(768));
        rows.writeTo(tables);
        tables.write(page);
        var path = directory.resolve("traffic.mckidx");
        try (var out = new DataOutputStream(Files.newOutputStream(path))) {
            out.writeBytes("MCKIDX01");
            out.writeLong(5);
            out.writeLong(96 + 5 * 16);
            out.writeLong(1);
            out.writeLong(96 + 5 * 16 + 28);
            out.writeLong(1);
            out.writeLong(96 + 5 * 16 + 28 + 12);
            out.writeLong(40);
            out.write(MessageDigest.getInstance("SHA-256").digest(tables.toByteArray()));
            tables.writeTo(out);
        }
        return path;
    }

    private static void record(ByteArrayOutputStream out, ByteBuffer row) throws IOException {
        out.write(row.array());
        var checksum = new CRC32();
        checksum.update(row.array());
        new DataOutputStream(out).writeInt((int) checksum.getValue());
    }

    @Test void reconstructsSharedTextAcrossAnInteriorBranchAndInsertsMedia() throws Exception {
        try (var dataset = new IndexedTraffic(fixture())) {
            var image = dataset.entry(0, 0);
            var branch = dataset.entry(1, 0).input();
            var shortPrefix = dataset.entry(2, 0).input();
            assertEquals(List.of(2, 1, 0, 1, 1), image.input().getTokenIdsList());
            assertEquals(List.of(2, 1, 3, 1), branch.getTokenIdsList());
            assertEquals(List.of(2, 1), shortPrefix.getTokenIdsList());
            assertEquals("mock-image:1", image.input().getImages(0).getUrl());
            assertEquals(2, image.input().getImages(0).getOffset());
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
        var odd = new Shard(1, null, path.toString(), 2, 1, 2, 100, identity, null, 2.0, 1, 4, 1000, 1000);
        var items = new ArrayList<Item>();
        long past = System.nanoTime() - 1_000_000_000;
        LoadClient.play(odd, past, items::add);
        assertEquals(List.of(1L, 3L), items.stream().map(Item::index).toList());
        assertEquals(List.of(101L, 103L), items.stream().map(Item::request_id).toList());
        assertEquals(List.of(5_000_000L, 15_000_000L), items.stream().map(item -> {
            try { return ReplayInput.delayNanos(item.index(), item.input(), null, 2.0); }
            catch (IOException error) { throw new AssertionError(error); }
        }).toList());
        assertEquals(10_000_000L, ReplayInput.delayNanos(1, items.getFirst().input(), 100.0, null));
        var even = new Shard(1, null, path.toString(), 3, 0, 2, 100, identity, null, 2.0, 1, 4, 1000, 1000);
        assertThrows(IOException.class, () -> LoadClient.play(even, past, ignored -> fail()));
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
