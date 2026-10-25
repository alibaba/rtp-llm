package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import static org.junit.jupiter.api.Assertions.*;

class ClientEventJournalTest {
    @TempDir Path directory;
    private static final ObjectMapper JSON = new ObjectMapper();

    @Test
    void flushesBeforeClientExitAndDoesNotMutateRequest() throws Exception {
        Path path = directory.resolve("events.jsonl");
        ObjectNode request = JSON.createObjectNode().put("rid", "one");
        try (ClientEventJournal journal = new ClientEventJournal(path)) {
            journal.record("issued", request);
            List<String> lines = Files.readAllLines(path);
            assertEquals(1, lines.size());
            assertEquals("issued", JSON.readTree(lines.get(0)).get("event").asText());
            assertFalse(request.has("sequence"));
            journal.record("terminal", request);
            assertEquals(2, Files.readAllLines(path).size());
        }
    }

    @Test
    void concurrentRequestsHaveCompleteOrderedLifecyclePairs() throws Exception {
        Path path = directory.resolve("events.jsonl");
        try (ClientEventJournal journal = new ClientEventJournal(path);
             var pool = Executors.newFixedThreadPool(8)) {
            List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
            for (int i = 0; i < 100; i++) {
                final int id = i;
                futures.add(pool.submit(() -> {
                    ObjectNode request = JSON.createObjectNode().put("rid", "r" + id);
                    journal.record("issued", request);
                    journal.record("terminal", request);
                }));
            }
            for (var future : futures) future.get();
        }
        List<String> lines = Files.readAllLines(path);
        assertEquals(200, lines.size());
        Map<String, String> previous = new HashMap<>();
        for (int i = 0; i < lines.size(); i++) {
            JsonNode row = JSON.readTree(lines.get(i));
            assertEquals(i + 1, row.get("sequence").asInt());
            String event = row.get("event").asText();
            assertEquals(event.equals("issued") ? null : "issued", previous.put(row.get("rid").asText(), event));
            assertTrue(row.get("recorded_epoch_ms").asLong() > 0);
        }
        assertEquals(100, previous.size());
        assertTrue(previous.values().stream().allMatch("terminal"::equals));
    }
}
