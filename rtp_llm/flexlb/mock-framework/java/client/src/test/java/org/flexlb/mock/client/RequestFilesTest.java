package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.flexlb.mock.client.ClientProtocol.*;

class RequestFilesTest {
    @TempDir Path directory;
    Start filePlan(Path file) {
        return new Start(null, null, file.toString(), 1000000000, 3, 4, 1000, 1000);
    }
    static Shard shard(RequestFiles.Part part, int index, int stride) {
        return new Shard(1, null, part.path(), part.count(), index, stride, 100, part.sha256(), 1000000000, 1, 4, 1000, 1000);
    }
    @Test void shardsHaveUniqueGlobalIdsAndKeepDefaultPriority() throws Exception {
        var file = directory.resolve("input.jsonl");
        var lines = new ArrayList<String>();
        for (int i = 0; i < 8; i++) lines.add("{\"token_ids\":[" + i + "],\"max_new_tokens\":3}");
        Files.writeString(file, String.join("\r\n", lines));
        var parts = RequestFiles.split(filePlan(file), directory.resolve("shards"), 3, () -> false);
        assertEquals(List.of(3L, 3L, 2L), parts.stream().map(RequestFiles.Part::count).toList());
        var ids = new HashSet<Long>();
        for (int i = 0; i < 3; i++) {
            int offset = i;
            Player.play(shard(parts.get(i), i, 3), System.nanoTime(), item -> {
                assertEquals(offset, item.index() % 3);
                assertEquals(item.index(), item.input().token_ids().getFirst().longValue());
                assertEquals(item.index() + 100, item.request_id());
                assertEquals(50, item.input().priority());
                assertTrue(ids.add(item.request_id()));
            });
        }
        assertEquals(8, ids.size());
        assertThrows(java.io.IOException.class, () -> RequestFiles.split(filePlan(file), directory.resolve("shards"), 3, () -> false));
    }
    @Test void badRecordsAreRejectedBeforeAnyShardIsPublished() throws Exception {
        var file = directory.resolve("bad.jsonl");
        var valid = "{\"token_ids\":[1],\"max_new_tokens\":2}";
        for (String bad : List.of("", " ", "null", "{}", valid + " {}", valid.replace("[1]", "[true]"),
                valid.replace("[1]", "[1.5]"), valid.replace("[1]", "[2147483648]"), valid.replace("[1]", "[-1]"),
                valid.replace("[1]", "[]"), valid.replace(":2", ":0"), valid.replace("}", ",\"priority\":null}"),
                valid.replace("}", ",\"extra\":1}"), valid.replace("}", ",\"max_new_tokens\":3}"))) {
            Files.writeString(file, valid + "\n" + bad + "\n");
            Path destination = directory.resolve(UUID.randomUUID().toString());
            var error = assertThrows(java.io.IOException.class, () -> RequestFiles.split(filePlan(file), destination, 1, () -> false));
            assertTrue(error.getMessage().contains("record 2"), error.toString());
            assertFalse(Files.exists(destination.resolve("0.jsonl")));
        }
        Files.writeString(file, " ".repeat(RequestFiles.MAX_RECORD_BYTES + 1));
        assertThrows(java.io.IOException.class, () -> RequestFiles.split(filePlan(file), directory.resolve("large"), 1, () -> false));
        Files.writeString(file, "");
        assertThrows(java.io.IOException.class, () -> RequestFiles.split(filePlan(file), directory.resolve("empty"), 1, () -> false));
    }
    @Test void stoppedPreparationAndOverlongPlaybackNeverPublish() throws Exception {
        var file = directory.resolve("input.jsonl");
        Files.writeString(file, "{\"token_ids\":[1],\"max_new_tokens\":2}\n");
        assertThrows(java.io.InterruptedIOException.class, () -> RequestFiles.split(filePlan(file), directory.resolve("stop"), 1, () -> true));
        var plan = new Start(null, null, file.toString(), 0.000001, 1, 1, 1000, 1000);
        assertThrows(java.io.IOException.class, () -> RequestFiles.split(plan, directory.resolve("slow"), 1, () -> false));
        assertFalse(Files.exists(directory.resolve("slow/0.jsonl")));
    }
    @Test void truncationExtraRecordsAndChangedContentCannotPassPlayback() throws Exception {
        var file = directory.resolve("input.jsonl");
        var record = "{\"token_ids\":[1],\"max_new_tokens\":2,\"priority\":50}\n";
        Files.writeString(file, record.repeat(3));
        var part = RequestFiles.split(filePlan(file), directory.resolve("shards"), 1, () -> false).getFirst();
        for (String changed : List.of(record, record.repeat(4), record.replace("[1]", "[2]").repeat(3))) {
            Files.writeString(Path.of(part.path()), changed);
            assertThrows(java.io.IOException.class, () -> Player.play(shard(part, 0, 1), System.nanoTime(), item -> {}));
        }
    }
    @Test void inputAndShardsLargerThanHeapAreStreamedInASeparateJvm() throws Exception {
        Path file = directory.resolve("large.jsonl"), log = directory.resolve("small-heap.log");
        String record = "{\"token_ids\":[" + String.join(",", Collections.nCopies(256, "1234567")) + "],\"max_new_tokens\":3}\n";
        try (var writer = Files.newBufferedWriter(file)) { for (int i = 0; i < 40000; i++) writer.write(record); }
        assertTrue(Files.size(file) > 64L * 1024 * 1024);
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
        var plan = new Start(null, null, args[0], 1000000000, 1, 1, 1000, 1000);
        var parts = RequestFiles.split(plan, Path.of(args[1]), 1, () -> false);
        var count = new AtomicLong();
        Player.play(shard(parts.getFirst(), 0, 1), System.nanoTime(), item -> count.incrementAndGet());
        if (count.get() != 40000) throw new AssertionError("Missing requests: " + count.get());
    }
}
