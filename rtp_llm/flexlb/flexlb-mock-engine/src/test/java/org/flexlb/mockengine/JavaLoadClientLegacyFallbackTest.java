package org.flexlb.mockengine;

import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Semaphore;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.withSettings;

/**
 * Locks the legacy single-target (router == null) schedule-stage fallback
 * trigger of JavaLoadClient.handleRequest: ANY schedule failure (here a gRPC
 * UNAVAILABLE from the schedule stub) with ENABLE_FALLBACK + a non-empty
 * fallback prefill list must attempt the direct-to-engine fallback
 * (route_path="fallback"), while the escape hatch stays off unless explicitly
 * opted in.
 *
 * <p>Regression lock for the code-review finding where the mode-dependent
 * trigger {@code router != null && "transport".equals(...)} silently
 * disabled the legacy fallback: in the legacy mode router is null, so the
 * trigger was constantly false and attemptFallback was unreachable — the
 * opposite of the documented "legacy ANY-failure, byte-identical" contract.
 *
 * <p>The schedule leg is a Mockito RETURNS_SELF stub injected through the
 * package field (deterministic immediate UNAVAILABLE, no network); the
 * fallback engine leg points at a loopback port with no listener, so
 * runFallbackStream also fails fast and attemptFallback stamps its
 * route_path="fallback" catch contract.
 */
class JavaLoadClientLegacyFallbackTest {

    @TempDir
    Path tempDir;

    /** Loopback ports with no listener: connects fail fast (RST). */
    private static final String UNREACHABLE_MASTER = "127.0.0.1:1";
    private static final String UNREACHABLE_ENGINE = "127.0.0.1:2";

    private JavaLoadClient legacyClient(boolean enableFallback) {
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", UNREACHABLE_MASTER,
                0, 4, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
                2_000L, 500.0, false, false, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "",
                enableFallback, "", false);
        return new JavaLoadClient(config);
    }

    /**
     * Replaces scheduleStubs[0] (a real channel-backed stub to the
     * unreachable master) with a mock that throws UNAVAILABLE on schedule(),
     * so the legacy schedule leg fails deterministically and instantly.
     */
    private static void injectFailingScheduleStub(JavaLoadClient client) throws Exception {
        Field stubsField = JavaLoadClient.class.getDeclaredField("scheduleStubs");
        stubsField.setAccessible(true);
        FlexlbServiceGrpc.FlexlbServiceBlockingStub[] stubs =
                (FlexlbServiceGrpc.FlexlbServiceBlockingStub[]) stubsField.get(client);
        FlexlbServiceGrpc.FlexlbServiceBlockingStub failing =
                mock(FlexlbServiceGrpc.FlexlbServiceBlockingStub.class,
                        withSettings().defaultAnswer(org.mockito.Answers.RETURNS_SELF));
        doThrow(new StatusRuntimeException(Status.UNAVAILABLE))
                .when(failing).schedule(any(FlexlbScheduleProtocol.FlexlbScheduleRequestPB.class));
        stubs[0] = failing;
    }

    private static JavaLoadClient.RequestResult handle(
            JavaLoadClient client, JavaLoadClient.TraceRecord record) throws Exception {
        Method handle = JavaLoadClient.class.getDeclaredMethod(
                "handleRequest", JavaLoadClient.TraceRecord.class, Semaphore.class, double.class);
        handle.setAccessible(true);
        return (JavaLoadClient.RequestResult)
                handle.invoke(client, record, new Semaphore(4), 0.0);
    }

    private static void closeClient(JavaLoadClient client) throws Exception {
        Method close = JavaLoadClient.class.getDeclaredMethod("close");
        close.setAccessible(true);
        close.invoke(client);
    }

    private static JavaLoadClient.TraceRecord rec(int idx) {
        List<Integer> tokens = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            tokens.add(i);
        }
        return new JavaLoadClient.TraceRecord(idx, "rid-" + idx, "trace-" + idx, 0L,
                64, 8, List.of(), tokens);
    }

    @Test
    @Timeout(30)
    void legacyScheduleFailureTriggersFallbackWhenEnabled() throws Exception {
        JavaLoadClient client = legacyClient(true);
        try {
            injectFailingScheduleStub(client);
            client.fallbackPrefillAddrs.add(UNREACHABLE_ENGINE);

            JavaLoadClient.RequestResult result = handle(client, rec(0));

            // attemptFallback ran: the direct-to-engine row is stamped
            // route_path="fallback" even when the engine leg also fails
            // (both errors are preserved on the row).
            assertEquals("fallback", result.routePath);
            assertEquals("exception", result.status);
            assertTrue(result.error.startsWith("master="), result.error);
            assertTrue(result.error.contains("; fallback="), result.error);
            assertEquals("transport", result.errorKind);
        } finally {
            closeClient(client);
        }
    }

    @Test
    @Timeout(30)
    void legacyScheduleFailureStaysOnErrorRowWhenFallbackDisabled() throws Exception {
        JavaLoadClient client = legacyClient(false);
        try {
            injectFailingScheduleStub(client);
            client.fallbackPrefillAddrs.add(UNREACHABLE_ENGINE);

            JavaLoadClient.RequestResult result = handle(client, rec(1));

            // Opt-in semantics: without ENABLE_FALLBACK the schedule failure
            // surfaces as a plain master-path error row — the fallback list
            // being non-empty must not be enough to bypass the master.
            assertEquals("master", result.routePath);
            assertEquals("exception", result.status);
            assertFalse(result.error.contains("fallback="), result.error);
            assertEquals("transport", result.errorKind);
        } finally {
            closeClient(client);
        }
    }
}
