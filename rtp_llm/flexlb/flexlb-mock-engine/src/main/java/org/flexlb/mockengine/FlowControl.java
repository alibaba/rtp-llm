package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Optional per-process flow-group control. Business requests never cross this channel. */
final class FlowControl {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final Path directory;
    private final String runId;
    private final String groupId;
    private final String phaseId;
    private String stopCommand;
    private String state = "STARTING";
    private long revision;
    private long lastProgressNanos;

    FlowControl(Path directory, String runId, String groupId, String phaseId) throws IOException {
        if (runId.isBlank() || groupId.isBlank() || phaseId.isBlank()) {
            throw new IllegalArgumentException("flow control requires run, group and phase identities");
        }
        this.directory = directory;
        this.runId = runId;
        this.groupId = groupId;
        this.phaseId = phaseId;
        Files.createDirectories(directory);
        if (Files.exists(directory.resolve("status.json"))) {
            throw new IllegalArgumentException("flow control directory has already been used");
        }
    }

    synchronized boolean stopRequested() throws IOException {
        Path path = directory.resolve("stop.json");
        if (!Files.exists(path)) return false;
        JsonNode command = JSON.readTree(path.toFile());
        if (!groupId.equals(command.path("group_id").asText())
                || !runId.equals(command.path("run_id").asText())
                || !"stop_sending".equals(command.path("operation").asText())
                || command.path("command_id").asText().isBlank()) {
            throw new IOException("invalid flow stop command identity or operation");
        }
        stopCommand = command.path("command_id").asText();
        return true;
    }

    boolean awaitDue(long dueNanos) throws IOException, InterruptedException {
        while (System.nanoTime() < dueNanos) {
            if (stopRequested()) return false;
            long remaining = dueNanos - System.nanoTime();
            if (remaining > 0) Thread.sleep(Math.min(50, Math.max(1, remaining / 1_000_000)));
        }
        return !stopRequested();
    }

    synchronized void publish(String nextState, int submitted, int started, int terminal) throws IOException {
        state = nextState;
        ObjectNode row = JSON.createObjectNode();
        row.put("schema_version", 1);
        row.put("run_id", runId);
        row.put("group_id", groupId);
        row.put("phase_id", phaseId);
        row.put("state", state);
        row.put("revision", ++revision);
        row.put("recorded_epoch_ms", System.currentTimeMillis());
        row.put("submitted", submitted);
        row.put("started", started);
        row.put("terminal", terminal);
        row.put("outstanding", Math.max(0, submitted - terminal));
        if (!"SENDING".equals(state) && !"STARTING".equals(state) && stopCommand != null) {
            row.put("applied_command_id", stopCommand);
        }
        Path temporary = directory.resolve("status.json.tmp");
        Files.writeString(temporary, row.toString());
        Files.move(temporary, directory.resolve("status.json"),
                StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    }

    synchronized void progress(int submitted, int started, int terminal) throws IOException {
        long now = System.nanoTime();
        if (now - lastProgressNanos >= 100_000_000L) {
            publish(state, submitted, started, terminal);
            lastProgressNanos = now;
        }
    }

    void identify(ObjectNode row) {
        row.put("run_id", runId);
        row.put("group_id", groupId);
        row.put("phase_id", phaseId);
    }
}
