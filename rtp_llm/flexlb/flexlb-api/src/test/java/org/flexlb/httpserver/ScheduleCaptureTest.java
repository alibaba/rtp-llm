package org.flexlb.httpserver;

import com.github.luben.zstd.ZstdInputStream;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;

class ScheduleCaptureTest {
    @TempDir Path directory;

    @Test
    void capturesRejectedAndCancelledAttemptsWithoutForwardedDuplicatesOrRawFields() throws Exception {
        var clock = new AtomicLong(1_000);
        var capture = new ScheduleCapture(directory, 2_000, 1, new byte[32], clock::get);
        var request = FlexlbScheduleRequestPB.newBuilder()
                .setRequestId(124987632)
                .setApiKey("not-for-capture")
                .setCacheKeyBlockSize(512).setSeqLen(1024).setMaxNewTokens(128).setPriority(50)
                .addAllBlockCacheKeys(List.of(811123909L, 961728432L)).build();

        assertEquals(0, capture.arrive(request));
        clock.set(2_050);
        assertEquals(0, capture.arrive(request.toBuilder().setForwardHop(1).build()));
        long first = capture.arrive(request);
        long retry = capture.arrive(request);
        long empty = capture.arrive(request.toBuilder().clearBlockCacheKeys().setRequestId(17).build());
        capture.complete(first, "rejected", 400);
        capture.complete(retry, "cancelled", 0);
        capture.complete(empty, "routed", 200);
        clock.set(93_001);

        Path output = directory.resolve("prefix-2000-" + ProcessHandle.current().pid() + ".jsonl.zst");
        for (int tries = 0; !Files.isRegularFile(output) && tries < 100; tries++) Thread.sleep(20);
        assertTrue(Files.isRegularFile(output));
        try (var in = new ZstdInputStream(Files.newInputStream(output))) {
            var content = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            var rows = content.lines().toList();
            assertEquals(2, rows.stream().filter(line -> line.startsWith("[\"node\",")).count());
            var attempts = rows.stream().filter(line -> line.startsWith("[\"request\",")).toList();
            assertEquals(3, attempts.size());
            assertEquals(attempts.get(0).split(",")[2], attempts.get(1).split(",")[2]);
            assertTrue(attempts.get(2).endsWith(",false,0]"));
            assertTrue(rows.contains("[\"result\",1,\"rejected\",400]"));
            assertTrue(rows.contains("[\"result\",2,\"cancelled\",0]"));
            assertEquals("[\"end\",3,0,0]", rows.getLast());
            for (String raw : List.of("124987632", "811123909", "961728432", "not-for-capture")) {
                assertFalse(content.contains(raw));
            }
        }
    }
}
