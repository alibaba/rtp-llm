package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.flexlb.mock.client.ClientProtocol.*;
import org.flexlb.mock.client.input.ClientInput.*;
import com.github.luben.zstd.ZstdCompressCtx;

class RequestFilesTest {
    @TempDir Path directory;
    Start filePlan(Path file) { return new Start(null, file.toString(), 1000000000.0, null, 3, 4, 1000, 1000); }
    static Shard shard(RequestFiles.Part part, int index, int stride) {
        return new Shard(1, null, part.path(), part.count(), index, stride, 100, part.sha256(), 1000000000.0, null, 1, 4, 1000, 1000);
    }
    static byte[] frame(Batch batch) throws Exception {
        try (var compressor = new ZstdCompressCtx().setLevel(3).setContentSize(true).setChecksum(true);
             var bytes = new ByteArrayOutputStream(); var out = new DataOutputStream(bytes)) {
            var encoded = compressor.compress(batch.toByteArray());
            out.writeInt(encoded.length); out.write(encoded); return bytes.toByteArray();
        }
    }
    @Test void sourceAndShardsUseSameCodecAndGlobalIds() throws Exception {
        var file = InputFixture.write(directory.resolve("input.binpb.zst"), java.util.stream.IntStream.range(0, 8)
            .mapToObj(i -> Request.newBuilder().addTokenIds(i).setMaxNewTokens(3).build()).toList());
        var parts = RequestFiles.split(filePlan(file), directory.resolve("shards"), 3, () -> false);
        assertEquals(List.of(3L, 3L, 2L), parts.stream().map(RequestFiles.Part::count).toList());
        var ids = new HashSet<Long>();
        for (int i = 0; i < 3; i++) {
            int offset = i;
            Player.play(shard(parts.get(i), i, 3), System.nanoTime(), item -> {
                assertEquals(offset, item.index() % 3);
                assertEquals(item.index(), item.input().getTokenIds(0));
                assertEquals(item.index() + 100, item.request_id());
                assertFalse(item.input().hasPriority());
                assertTrue(ids.add(item.request_id()));
            });
        }
        assertEquals(8, ids.size());
        assertThrows(IOException.class, () -> RequestFiles.split(filePlan(file), directory.resolve("shards"), 3, () -> false));
    }
    @Test void replayKeepsGlobalArrivalOffsetsAcrossShardsAndAppliesSpeedOnce() throws Exception {
        var file = InputFixture.write(directory.resolve("timed.binpb.zst"), java.util.stream.IntStream.range(0, 6)
            .mapToObj(i -> InputFixture.request().toBuilder().setArrivalOffsetMs(i * 100L).build()).toList());
        var plan = new Start(null, file.toString(), null, 5.0, 3, 4, 1000, 1000);
        var parts = RequestFiles.split(plan, directory.resolve("timed-shards"), 3, () -> false);
        for (int i = 0; i < 3; i++) {
            var part = parts.get(i);
            var replay = new Shard(1, null, part.path(), part.count(), i, 3, 100, part.sha256(), null, 5.0, 1, 4, 1000, 1000);
            Player.play(replay, System.nanoTime() - TimeUnit.SECONDS.toNanos(1), item -> {
                assertEquals(item.index() * 100, item.input().getArrivalOffsetMs());
                assertEquals(item.index() * 20_000_000L, assertDoesNotThrow(() -> Player.delayNanos(item.index(), item.input(), null, 5.0)));
            });
        }
    }
    @Test void replayRejectsMissingNegativeDecreasingAndOverlongTimestampsBeforePublication() throws Exception {
        var base = InputFixture.request();
        for (var inputs : List.of(List.of(base), List.of(base.toBuilder().setArrivalOffsetMs(-1).build()),
                List.of(base.toBuilder().setArrivalOffsetMs(2).build(), base.toBuilder().setArrivalOffsetMs(1).build()),
                List.of(base.toBuilder().setArrivalOffsetMs(86_400_001).build()))) {
            var file = directory.resolve(UUID.randomUUID() + ".binpb.zst");
            if (inputs.getFirst().getArrivalOffsetMs() < 0) {
                try (var writer = new InputFile.Writer(new ByteArrayOutputStream())) { assertThrows(IOException.class, () -> writer.write(inputs.getFirst())); }
                continue;
            }
            InputFixture.write(file, inputs);
            var target = directory.resolve(UUID.randomUUID().toString());
            assertThrows(IOException.class, () -> RequestFiles.split(new Start(null, file.toString(), null, 1.0, 1, 1, 1000, 1000), target, 1, () -> false));
            assertFalse(Files.exists(target.resolve("0.binpb.zst")));
        }
        assertThrows(IllegalArgumentException.class, () -> new Start(null, "/input", null, null, 1, 1, 1000, 1000));
        assertThrows(IllegalArgumentException.class, () -> new Start(null, "/input", 1.0, 1.0, 1, 1, 1000, 1000));
        assertThrows(IllegalArgumentException.class, () -> new Start(null, "/input", null, Double.NaN, 1, 1, 1000, 1000));
    }
    @Test void invalidRequestsAndJsonInputAreRejected() throws Exception {
        var valid = InputFixture.request();
        for (var invalid : List.of(valid.toBuilder().clearTokenIds().build(), valid.toBuilder().setTokenIds(0, -1).build(),
                valid.toBuilder().setMaxNewTokens(0).build(), valid.toBuilder().setPriority(-1).build(),
                valid.toBuilder().setUnknownFields(com.google.protobuf.UnknownFieldSet.newBuilder()
                    .addField(100, com.google.protobuf.UnknownFieldSet.Field.newBuilder().addVarint(1).build()).build()).build())) {
            try (var writer = new InputFile.Writer(new ByteArrayOutputStream())) {
                assertThrows(IOException.class, () -> writer.write(invalid));
            }
        }
        var file = directory.resolve("legacy.jsonl");
        Files.writeString(file, "{\"token_ids\":[1],\"max_new_tokens\":2}\n");
        assertThrows(IOException.class, () -> RequestFiles.split(filePlan(file), directory.resolve("legacy"), 1, () -> false));
        assertFalse(Files.exists(directory.resolve("legacy/0.binpb.zst")));
    }
    @Test void malformedFramesNeverPublishShards() throws Exception {
        var request = InputFixture.request();
        byte[] data = frame(Batch.newBuilder().setSequence(1).addRequests(request).build());
        byte[] seal = frame(Batch.newBuilder().setSequence(2).setTotalRequests(1).build());
        var cases = new ArrayList<byte[]>();
        cases.add(data); // Missing final seal.
        cases.add(Arrays.copyOf(data, data.length - 1));
        cases.add(frame(Batch.newBuilder().setSequence(2).addRequests(request).build()));
        cases.add(frame(Batch.newBuilder().setSequence(1).addRequests(Request.getDefaultInstance()).build()));
        var corrupt = data.clone(); corrupt[corrupt.length - 1] ^= 1; cases.add(corrupt);
        try (var bytes = new ByteArrayOutputStream()) {
            bytes.write(data); bytes.write(frame(Batch.newBuilder().setSequence(2).setTotalRequests(2).build())); cases.add(bytes.toByteArray());
        }
        try (var bytes = new ByteArrayOutputStream()) {
            bytes.write(data); bytes.write(seal); bytes.write(1); cases.add(bytes.toByteArray());
        }
        try (var compressor = new ZstdCompressCtx().setLevel(3).setChecksum(true).setContentSize(true);
             var bytes = new ByteArrayOutputStream(); var out = new DataOutputStream(bytes)) {
            byte[] first = compressor.compress(Batch.newBuilder().setSequence(1).addRequests(request).build().toByteArray());
            byte[] extra = compressor.compress(new byte[0]);
            out.writeInt(first.length + extra.length); out.write(first); out.write(extra); out.write(seal);
            cases.add(bytes.toByteArray());
        }
        for (var content : cases) {
            var file = directory.resolve(UUID.randomUUID() + ".binpb.zst");
            try (var out = Files.newOutputStream(file)) { out.write(InputFile.MAGIC); out.write(content); }
            var destination = directory.resolve(UUID.randomUUID().toString());
            assertThrows(IOException.class, () -> RequestFiles.split(filePlan(file), destination, 1, () -> false));
            assertFalse(Files.exists(destination.resolve("0.binpb.zst")));
        }
    }
    @Test void stoppedPreparationAndOverlongPlaybackNeverPublish() throws Exception {
        var file = InputFixture.write(directory.resolve("input.binpb.zst"), List.of(InputFixture.request()));
        assertThrows(InterruptedIOException.class, () -> RequestFiles.split(filePlan(file), directory.resolve("stop"), 1, () -> true));
        var plan = new Start(null, file.toString(), 0.000001, null, 1, 1, 1000, 1000);
        assertThrows(IOException.class, () -> RequestFiles.split(plan, directory.resolve("slow"), 1, () -> false));
        assertFalse(Files.exists(directory.resolve("slow/0.binpb.zst")));
    }
    @Test void truncationExtraRecordsAndChangedContentCannotPassPlayback() throws Exception {
        var file = InputFixture.write(directory.resolve("input.binpb.zst"), Collections.nCopies(3, InputFixture.request()));
        var part = RequestFiles.split(filePlan(file), directory.resolve("shards"), 1, () -> false).getFirst();
        for (int count : List.of(1, 4, 3)) {
            Files.delete(Path.of(part.path()));
            InputFixture.write(Path.of(part.path()), Collections.nCopies(count, InputFixture.request().toBuilder().setPriority(30).build()));
            assertThrows(IOException.class, () -> Player.play(shard(part, 0, 1), System.nanoTime(), item -> {}));
        }
    }
    @Test void expandedInputLargerThanHeapIsStreamedInASeparateJvm() throws Exception {
        Path file = directory.resolve("large.binpb.zst"), log = directory.resolve("small-heap.log");
        var request = InputFixture.request().toBuilder().clearTokenIds()
            .addAllTokenIds(java.util.stream.IntStream.range(1000000, 1001024).boxed().toList()).build();
        assertTrue((long) request.getSerializedSize() * 40000 > 64L * 1024 * 1024);
        InputFixture.write(file, Collections.nCopies(40000, request));
        var process = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin/java").toString(),
            "-Xmx48m", "-XX:ActiveProcessorCount=2", "-cp", System.getProperty("java.class.path"),
            getClass().getName(), file.toString(), directory.resolve("bounded").toString())
            .redirectErrorStream(true).redirectOutput(log.toFile()).start();
        try {
            assertTrue(process.waitFor(90, TimeUnit.SECONDS), "Small-heap playback timed out");
            assertEquals(0, process.exitValue(), Files.readString(log));
        } finally { process.destroyForcibly(); process.waitFor(); }
    }
    public static void main(String[] args) throws Exception {
        var plan = new Start(null, args[0], 1000000000.0, null, 1, 1, 1000, 1000);
        var parts = RequestFiles.split(plan, Path.of(args[1]), 1, () -> false);
        var count = new AtomicLong();
        Player.play(shard(parts.getFirst(), 0, 1), System.nanoTime(), item -> count.incrementAndGet());
        if (count.get() != 40000) throw new AssertionError("Missing requests: " + count.get());
    }
}
