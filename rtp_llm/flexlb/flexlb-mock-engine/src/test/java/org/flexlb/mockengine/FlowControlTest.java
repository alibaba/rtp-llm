package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class FlowControlTest {
    @TempDir Path directory;
    private static final ObjectMapper JSON = new ObjectMapper();

    @Test void stopRequiresAppliedBoundaryAndPreservesOutstanding() throws Exception {
        FlowControl flow = new FlowControl(directory, "run", "background", "prepare");
        flow.publish("SENDING", 4, 3, 1);
        Files.writeString(directory.resolve("stop.json"), """
                {"run_id":"run","group_id":"background","operation":"stop_sending","command_id":"s1"}
                """);
        assertTrue(flow.stopRequested());
        assertFalse(JSON.readTree(directory.resolve("status.json").toFile()).has("applied_command_id"));
        flow.publish("DRAINING", 4, 3, 1);
        var status = JSON.readTree(directory.resolve("status.json").toFile());
        assertEquals("s1", status.path("applied_command_id").asText());
        assertEquals(3, status.path("outstanding").asInt());
        assertFalse(flow.awaitDue(System.nanoTime() + 60_000_000_000L));
        flow.publish("DRAINED", 4, 4, 4);
        assertEquals(0, JSON.readTree(directory.resolve("status.json").toFile()).path("outstanding").asInt());
    }

    @Test void controlledTraceDoesNotSilentlyRepairShapeOrKeys() throws Exception {
        var row = JSON.createObjectNode().put("rid", "one").put("il", 2).put("ol", 1)
                .put("ts", 0).put("priority", 50).put("cache_key_block_size", 1024);
        row.putArray("input_ids").add(7).add(8);
        JavaLoadClient.validateControlledTrace(row);
        row.put("il", 3);
        assertThrows(IllegalArgumentException.class, () -> JavaLoadClient.validateControlledTrace(row));
        row.put("il", 2);
        row.putArray("bh").add(123);
        assertThrows(IllegalArgumentException.class, () -> JavaLoadClient.validateControlledTrace(row));
        row.remove("bh");
        row.put("priority", 101);
        assertThrows(IllegalArgumentException.class, () -> JavaLoadClient.validateControlledTrace(row));
    }

    @Test void foreignCommandAndReusedDirectoryAreRejected() throws Exception {
        FlowControl flow = new FlowControl(directory, "run", "group", "formal");
        flow.publish("STARTING", 0, 0, 0);
        assertThrows(IllegalArgumentException.class, () -> new FlowControl(directory, "run", "group", "formal"));
        Files.writeString(directory.resolve("stop.json"), """
                {"run_id":"other","group_id":"group","operation":"stop_sending","command_id":"s1"}
                """);
        assertThrows(java.io.IOException.class, flow::stopRequested);
    }
}
