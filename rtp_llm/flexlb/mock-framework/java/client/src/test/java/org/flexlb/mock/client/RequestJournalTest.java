package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.protocol.MultimodalProtocol;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Verifies request evidence, final seals, and propagation of writer failures.
 */
class RequestJournalTest {
    @TempDir Path directory;

    @Test void backupAttemptsAndCancellationAreRecordedIncrementally() throws Exception {
        var path = directory.resolve("requests.jsonl");
        var journal = new RequestJournal(path, "client-0", plan());
        try (var first = new BackendRpcClientTest.Harness(); var backup = new BackendRpcClientTest.Harness();
             var client = new BackendRpcClient(() -> List.of(first.endpoint().get("rpc"), backup.endpoint().get("rpc")), journal::record, MultimodalProtocol.IMAGE_TOKEN, 0L, 1)) {
            first.scheduleFailure = io.grpc.Status.UNAVAILABLE;
            backup.output = (id, response) -> response.onNext(BackendRpcClientTest.token(id, 1, false));
            var call = client.send(BackendRpcClientTest.request(100), Duration.ofSeconds(8));
            call.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS);
            PlaybackTest.eventually(() -> {
                try { return Files.readString(path).contains("\"FIRST_TOKEN\""); } catch (IOException e) { return false; }
            });
            assertFalse(read(path).getLast().has("seal"), "Evidence must exist before the request ends");
            call.cancel(); call.completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            journal.finish(1, 1);
            var rows = read(path);
            var events = rows.subList(1, rows.size() - 1);
            assertEquals(List.of("ISSUED", "SCHEDULE_START", "SCHEDULE_RESULT",
                "SCHEDULE_START", "SCHEDULE_RESULT", "FIRST_TOKEN", "TERMINAL"),
                events.stream().map(row -> row.path("kind").asText()).toList());
            for (int i = 0; i < events.size(); i++) {
                assertEquals(Integer.toString(i + 1), events.get(i).get("sequence").textValue());
                assertEquals("100", events.get(i).get("request_id").textValue());
            }
            assertEquals("UNAVAILABLE", events.get(2).path("grpc_code").asText());
            assertEquals(2, events.get(4).path("attempt").asInt()); assertEquals("OK", events.get(4).path("grpc_code").asText());
            assertEquals("cancelled", events.getLast().path("outcome").asText()); assertTrue(events.getLast().has("ttft_nanos"));
            assertEquals("1", rows.getLast().path("seal").path("finished").textValue());
            assertThrows(FileAlreadyExistsException.class, () -> new RequestJournal(path, "client-0", plan()));
        }
    }

    static ClientProtocol.ClientReplayPlan plan() {
        return new ClientProtocol.ClientReplayPlan(1, null, "/unused.mckidx", 1, 0, 1, 100, "0".repeat(64), 1.0, null, 1, 1, 3000, 1000);
    }

    static List<JsonNode> read(Path path) throws IOException {
        var mapper = new ObjectMapper();
        var rows = new ArrayList<JsonNode>();
        for (var line : Files.readAllLines(path)) rows.add(mapper.readTree(line));
        return rows;
    }

    @Test void queueOverflowCannotSealOrSilentlyDrop() throws Exception {
        var path = directory.resolve("requests.jsonl");
        var journal = new RequestJournal(path, "client-0", plan(), 1);
        journal.record(new ObjectMapper().createObjectNode().put("kind", "ISSUED").put("request_id", "100"));
        assertNotNull(journal.failure());
        assertThrows(IOException.class, () -> journal.finish(1, 1));
        assertTrue(read(path).stream().noneMatch(row -> row.has("seal")));
    }

    @Test void ioFailureCannotReportSuccessfulClose() throws Exception {
        var journal = new RequestJournal(directory.resolve("requests.jsonl"), "client-0", plan());
        var field = RequestJournal.class.getDeclaredField("file"); field.setAccessible(true);
        ((FileChannel) field.get(journal)).close();
        assertThrows(IOException.class, () -> journal.finish(0, 0));
        assertTrue(journal.failure().contains("write failed"));
    }

    @Test void unfinishedRequestCannotSeal() throws Exception {
        var path = directory.resolve("requests.jsonl");
        var journal = new RequestJournal(path, "client-0", plan());
        assertThrows(IOException.class, () -> journal.finish(1, 0));
        assertTrue(read(path).stream().noneMatch(row -> row.has("seal")));
    }

    @Test void writerFailureStopsAllClientsAndKeepsPlaybackFailed() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var master = new BackendRpcClientTest.Harness();
             var zero = new ClientNode("client-0", topology, directory.resolve("client-0"));
             var one = new ClientNode("client-1", topology, directory.resolve("client-1"))) {
            new com.fasterxml.jackson.databind.ObjectMapper().writeValue(topology.toFile(), List.of(master.endpoint(),
                Map.of("role", "client", "control", zero.url()), Map.of("role", "client", "control", one.url())));
            zero.startReplay(InputFixture.plan(directory, null, 100, 10, 4, 16, 3000, 1000));
            PlaybackTest.eventually(() -> one.state(new ClientProtocol.Sequence(1)).issued() > 0);
            var workerField = ClientNode.class.getDeclaredField("worker"); workerField.setAccessible(true);
            var journalField = ClientNode.ReplayWorker.class.getDeclaredField("journal"); journalField.setAccessible(true);
            var channelField = RequestJournal.class.getDeclaredField("file"); channelField.setAccessible(true);
            var journal = (RequestJournal) journalField.get(workerField.get(one));
            ((FileChannel) channelField.get(journal)).close();
            PlaybackTest.eventually(() -> zero.state(new ClientProtocol.Sequence(1)).done());
            var result = zero.stopReplay(new ClientProtocol.Empty());
            assertFalse(result.errors().isEmpty()); assertEquals(2, result.clients().size());
            assertTrue(result.clients().values().stream().allMatch(ClientProtocol.State::done));
            assertTrue(result.clients().values().stream().allMatch(s -> s.unsent() > 0));
            assertFalse(result.clients().get(one.url()).errors().isEmpty());
        }
    }

    @Test void businessRejectionAndDeadlineKeepDistinctEvidence() throws Exception {
        for (boolean timeout : List.of(false, true)) {
            var path = directory.resolve(timeout ? "timeout.jsonl" : "rejection.jsonl");
            var journal = new RequestJournal(path, "client-0", plan());
            try (var master = new BackendRpcClientTest.Harness();
                 var client = new BackendRpcClient(() -> List.of(master.endpoint().get("rpc")), journal::record, MultimodalProtocol.IMAGE_TOKEN, 0L, 1)) {
                if (timeout) master.scheduleDelay = 1000;
                else master.route = r -> r.toBuilder().setSuccess(false).setCode(429).clearLifecycle().build();
                var call = client.send(BackendRpcClientTest.request(100), Duration.ofMillis(timeout ? 100 : 3000));
                call.completion().toCompletableFuture().get(4, TimeUnit.SECONDS);
                journal.finish(1, 1);
                var events = read(path).stream().filter(row -> row.has("kind")).toList();
                var attempt = events.stream().filter(e -> "SCHEDULE_RESULT".equals(e.path("kind").asText())).findFirst().orElseThrow();
                assertEquals(timeout ? "DEADLINE_EXCEEDED" : "OK", attempt.path("grpc_code").asText());
                assertEquals(!timeout, attempt.has("response_success")); assertFalse(attempt.path("response_success").asBoolean());
                assertEquals(timeout ? 0 : 429, attempt.path("error_code").asLong());
                assertEquals(timeout ? "timeout" : "error", events.getLast().path("outcome").asText());
                assertFalse(events.getLast().has("ttft_nanos"));
                assertTrue(events.stream().noneMatch(e -> "FIRST_TOKEN".equals(e.path("kind").asText())));
            }
        }
    }

}
