package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Auto-TPM priority support in {@link JavaLoadClient}: per-record priority
 * parsing (trace field beats the PRIORITY env default), the FORCE_PRIORITY
 * single-level pin (overrides both), propagation onto
 * FlexlbScheduleRequestPB.priority (field 14; 0 keeps it off the wire),
 * validation with graceful fallback (warn + default, never a hard fail),
 * and raw per-request row serialization (synthesized rows omit the
 * priority key entirely rather than writing a misleading 0).
 */
class LoadClientPriorityTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    private JavaLoadClient dryRunClient(int priority) {
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:7003",
                0, 16, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
                120_000L, 500.0, false, false, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true,
                priority);
        return new JavaLoadClient(config);
    }

    private static JavaLoadClient.TraceRecord record(long requestId, int priority) {
        return new JavaLoadClient.TraceRecord(requestId, "rid-" + requestId,
                "trace-" + requestId, 1000L, 2048, 10,
                List.of(1L, 2L), List.of(1, 2, 3), priority);
    }

    private static EngineRpcService.GenerateInputPB input(long requestId) {
        return EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(10)
                        .build())
                .build();
    }

    // ---- priority propagation onto the schedule request ----

    @Test
    void scheduleRequestCarriesRecordPriority() {
        JavaLoadClient client = dryRunClient(0);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                client.buildScheduleRequest(record(1L, 70), input(1L));
        assertEquals(70, request.getPriority());
    }

    @Test
    void priorityZeroStaysOffTheWire() {
        JavaLoadClient client = dryRunClient(0);
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB request =
                client.buildScheduleRequest(record(2L, 0), input(2L));
        assertEquals(0, request.getPriority());
        // proto3 scalar default: value 0 must not be serialized.
        FlexlbScheduleProtocol.FlexlbScheduleRequestPB withPriority =
                client.buildScheduleRequest(record(2L, 30), input(2L));
        assertTrue(withPriority.getSerializedSize() > request.getSerializedSize());
    }

    // ---- trace parsing: record field beats env default ----

    @Test
    void traceRecordFieldOverridesConfigDefault() throws Exception {
        JavaLoadClient client = dryRunClient(40);

        ObjectNode withField = MAPPER.createObjectNode()
                .put("il", 100).put("ol", 10).put("ts", 1L)
                .put("request_id", "r1").put("priority", 70);
        JavaLoadClient.TraceRecord parsed = client.parseTraceRecord(withField);
        assertNotNull(parsed);
        assertEquals(70, parsed.priority);

        ObjectNode withoutField = MAPPER.createObjectNode()
                .put("il", 100).put("ol", 10).put("ts", 1L)
                .put("request_id", "r2");
        JavaLoadClient.TraceRecord defaulted = client.parseTraceRecord(withoutField);
        assertNotNull(defaulted);
        assertEquals(40, defaulted.priority);
    }

    // ---- trace parsing: FORCE_PRIORITY pins a single level ----

    @Test
    void forcePriorityPinsEveryRecord() throws Exception {
        // Bottom constructor carries the FORCE_PRIORITY knob (fromEnv reads
        // the env); the 35-param convenience overload forwards 0 = disabled.
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:7003",
                0, 16, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
                120_000L, 500.0, false, false, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true,
                40, 50, "replay", 0.0, true);
        JavaLoadClient client = new JavaLoadClient(config);

        ObjectNode withField = MAPPER.createObjectNode()
                .put("il", 100).put("ol", 10).put("ts", 1L)
                .put("request_id", "r1").put("priority", 70);
        JavaLoadClient.TraceRecord pinned = client.parseTraceRecord(withField);
        assertNotNull(pinned);
        assertEquals(50, pinned.priority);

        ObjectNode withoutField = MAPPER.createObjectNode()
                .put("il", 100).put("ol", 10).put("ts", 1L)
                .put("request_id", "r2");
        JavaLoadClient.TraceRecord pinnedDefault = client.parseTraceRecord(withoutField);
        assertNotNull(pinnedDefault);
        assertEquals(50, pinnedDefault.priority);
    }

    @Test
    void loopAndTruncationPreservePriority() {
        JavaLoadClient.TraceRecord original = new JavaLoadClient.TraceRecord(
                1L, "rid-1", "trace-1", 1000L, 4096, 500,
                List.of(1L), List.of(1, 2, 3), 60);
        List<JavaLoadClient.TraceRecord> truncated =
                JavaLoadClient.truncateRecords(List.of(original), 1024, 100);
        assertEquals(1, truncated.size());
        assertEquals(60, truncated.get(0).priority);
        assertEquals(1024, truncated.get(0).inputLen);
        assertEquals(100, truncated.get(0).outputLen);
    }

    // ---- raw per-request row serialization ----

    @Test
    void syntheticRowsOmitThePriorityKey() {
        ObjectNode real = JavaLoadClient.perRequestNode(result(70, "ok", 10.0));
        assertTrue(real.has("priority"));
        assertEquals(70, real.get("priority").asInt());

        // client_events.jsonl rows distinguish synthesized entries by key
        // absence — a literal 0 would pollute downstream priority stats.
        ObjectNode synthetic = JavaLoadClient.perRequestNode(syntheticResult("timeout"));
        assertFalse(synthetic.has("priority"),
                "synthetic rows must not write priority=0");
    }

    @Test
    void invalidTracePriorityFallsBackToConfigDefault() throws Exception {
        JavaLoadClient client = dryRunClient(40);

        ObjectNode invalidHigh = MAPPER.createObjectNode()
                .put("il", 100).put("ol", 10).put("ts", 1L)
                .put("request_id", "r1").put("priority", 200);
        assertEquals(40, client.parseTraceRecord(invalidHigh).priority);

        ObjectNode invalidNegative = MAPPER.createObjectNode()
                .put("il", 100).put("ol", 10).put("ts", 1L)
                .put("request_id", "r2").put("priority", -5);
        assertEquals(40, client.parseTraceRecord(invalidNegative).priority);

        // Explicit 0 stays unset (legacy wire behavior), not "invalid".
        ObjectNode explicitZero = MAPPER.createObjectNode()
                .put("il", 100).put("ol", 10).put("ts", 1L)
                .put("request_id", "r3").put("priority", 0);
        assertEquals(0, client.parseTraceRecord(explicitZero).priority);
    }

    @Test
    void envPriorityKnobsRejectInvalidLevels() {
        // Valid and legacy values pass through unchanged.
        assertEquals(70, JavaLoadClient.Config.sanitizePriority(70));
        assertEquals(0, JavaLoadClient.Config.sanitizePriority(0));
        assertEquals(60, JavaLoadClient.Config.sanitizeForcePriority(60));
        assertEquals(0, JavaLoadClient.Config.sanitizeForcePriority(0));
        // Out-of-range values warn and fall back instead of failing the run.
        assertEquals(50, JavaLoadClient.Config.sanitizePriority(200));
        assertEquals(50, JavaLoadClient.Config.sanitizePriority(-5));
        assertEquals(0, JavaLoadClient.Config.sanitizeForcePriority(101));
        assertEquals(0, JavaLoadClient.Config.sanitizeForcePriority(-1));
    }

    private static JavaLoadClient.RequestResult result(int priority, String status, double scheduleMs) {
        JavaLoadClient.RequestResult result = new JavaLoadClient.RequestResult();
        result.priority = priority;
        result.status = status;
        result.scheduleMs = scheduleMs;
        return result;
    }

    /** A collector-synthesized row (timeout/exception fallback): no real
     *  request ever carried a priority for it. */
    private static JavaLoadClient.RequestResult syntheticResult(String status) {
        JavaLoadClient.RequestResult result = result(0, status, 0.0);
        result.synthetic = true;
        return result;
    }
}
