package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import com.google.protobuf.util.JsonFormat;
import java.io.*;
import java.nio.channels.FileChannel;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.client.evidence.ClientEvidence.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RequestJournalTest {
    @TempDir Path directory;
    static ClientProtocol.Shard plan() {
        return new ClientProtocol.Shard(1, null, "/unused", 1, 0, 1, 100, "0".repeat(64), 1.0, null, 1, 1, 3000, 1000);
    }
    static List<EventBatch> read(Path path) throws Exception {
        var rows = Files.readAllLines(path);
        if (rows.isEmpty()) return List.of();
        var batch = EventBatch.newBuilder();
        JsonFormat.parser().merge(rows.getFirst(), batch);
        for (var row : rows.subList(1, rows.size())) {
            if (row.startsWith("{\"seal\"")) JsonFormat.parser().merge(row, batch);
            else JsonFormat.parser().merge(row, batch.addEventsBuilder());
        }
        return List.of(batch.build());
    }
    @Test void backupAttemptsAndCancellationAreRecordedIncrementally() throws Exception {
        var path = directory.resolve("requests.jsonl");
        var journal = new RequestJournal(path, "client-0", plan());
        try (var first = new NativeClientTest.Harness(); var backup = new NativeClientTest.Harness();
             var client = new NativeClient(() -> List.of(first.endpoint().get("rpc"), backup.endpoint().get("rpc")), journal::record)) {
            first.scheduleFailure = io.grpc.Status.UNAVAILABLE;
            backup.output = (id, response) -> response.onNext(NativeClientTest.token(id, 1, false));
            var call = client.send(NativeClientTest.request(100), Duration.ofSeconds(8));
            call.firstToken().toCompletableFuture().get(2, TimeUnit.SECONDS);
            PlaybackTest.eventually(() -> {
                try { return Files.readString(path).contains("\"FIRST_TOKEN\""); } catch (IOException e) { return false; }
            });
            assertFalse(read(path).getLast().hasSeal(), "Evidence must exist before the request ends");
            call.cancel(); call.completion().toCompletableFuture().get(2, TimeUnit.SECONDS);
            journal.finish(1, 1);
            var frames = read(path);
            var events = frames.stream().flatMap(b -> b.getEventsList().stream()).toList();
            assertEquals(List.of(Event.Kind.ISSUED, Event.Kind.SCHEDULE_START, Event.Kind.SCHEDULE_RESULT,
                Event.Kind.SCHEDULE_START, Event.Kind.SCHEDULE_RESULT, Event.Kind.FIRST_TOKEN, Event.Kind.TERMINAL),
                events.stream().map(Event::getKind).toList());
            for (int i = 0; i < events.size(); i++) {
                assertEquals(i + 1, events.get(i).getSequence()); assertEquals(100, events.get(i).getRequestId());
            }
            assertEquals("UNAVAILABLE", events.get(2).getGrpcCode());
            assertEquals(2, events.get(4).getAttempt()); assertEquals("OK", events.get(4).getGrpcCode());
            assertEquals("cancelled", events.getLast().getOutcome()); assertTrue(events.getLast().hasTtftNanos());
            assertEquals(1, frames.getLast().getSeal().getFinished());
            assertThrows(FileAlreadyExistsException.class, () -> new RequestJournal(path, "client-0", plan()));
        }
    }
    @Test void queueOverflowCannotSealOrSilentlyDrop() throws Exception {
        var path = directory.resolve("requests.jsonl");
        var journal = new RequestJournal(path, "client-0", plan(), 1);
        journal.record(Event.newBuilder().setKind(Event.Kind.ISSUED).setRequestId(100));
        assertNotNull(journal.failure());
        assertThrows(IOException.class, () -> journal.finish(1, 1));
        assertTrue(read(path).stream().noneMatch(EventBatch::hasSeal));
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
        assertTrue(read(path).stream().noneMatch(EventBatch::hasSeal));
    }
    @Test void writerFailureStopsAllClientsAndKeepsPlaybackFailed() throws Exception {
        Path topology = directory.resolve("topology.json");
        try (var master = new NativeClientTest.Harness();
             var zero = new ClientNode("client-0", topology, directory.resolve("client-0"));
             var one = new ClientNode("client-1", topology, directory.resolve("client-1"))) {
            new com.fasterxml.jackson.databind.ObjectMapper().writeValue(topology.toFile(), List.of(master.endpoint(),
                Map.of("role", "client", "control", zero.url()), Map.of("role", "client", "control", one.url())));
            zero.start(InputFixture.plan(directory, null, 100, 10, 4, 16, 3000, 1000));
            PlaybackTest.eventually(() -> one.state(new ClientProtocol.Sequence(1)).issued() > 0);
            var workerField = ClientNode.class.getDeclaredField("worker"); workerField.setAccessible(true);
            var journalField = LoadClient.class.getDeclaredField("journal"); journalField.setAccessible(true);
            var channelField = RequestJournal.class.getDeclaredField("file"); channelField.setAccessible(true);
            var journal = (RequestJournal) journalField.get(workerField.get(one));
            ((FileChannel) channelField.get(journal)).close();
            PlaybackTest.eventually(() -> zero.state(new ClientProtocol.Sequence(1)).done());
            var result = zero.stop(new ClientProtocol.Empty());
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
            try (var master = new NativeClientTest.Harness();
                 var client = new NativeClient(() -> List.of(master.endpoint().get("rpc")), journal::record)) {
                if (timeout) master.scheduleDelay = 1000;
                else master.route = r -> r.toBuilder().setSuccess(false).setCode(429).clearLifecycle().build();
                var call = client.send(NativeClientTest.request(100), Duration.ofMillis(timeout ? 100 : 3000));
                call.completion().toCompletableFuture().get(4, TimeUnit.SECONDS);
                journal.finish(1, 1);
                var events = read(path).stream().flatMap(b -> b.getEventsList().stream()).toList();
                var attempt = events.stream().filter(e -> e.getKind() == Event.Kind.SCHEDULE_RESULT).findFirst().orElseThrow();
                assertEquals(timeout ? "DEADLINE_EXCEEDED" : "OK", attempt.getGrpcCode());
                assertEquals(!timeout, attempt.hasResponseSuccess()); assertFalse(attempt.getResponseSuccess());
                assertEquals(timeout ? 0 : 429, attempt.getErrorCode());
                assertEquals(timeout ? "timeout" : "error", events.getLast().getOutcome());
                assertFalse(events.getLast().hasTtftNanos());
                assertTrue(events.stream().noneMatch(e -> e.getKind() == Event.Kind.FIRST_TOKEN));
            }
        }
    }

}
