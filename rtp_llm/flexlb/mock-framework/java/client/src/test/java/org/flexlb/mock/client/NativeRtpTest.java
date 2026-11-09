package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.Map;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.flexlb.mock.engine.scheduler.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Checks generation through the native RTP Client and Engine RPCs.
 */
class NativeRtpTest {
    @TempDir Path directory;

    @Test void nativeClientConsumesGenerateAndEnqueueFetchFromLLMMockEngines() throws Exception {
        try (var p = node("prefill"); var d = node("decode")) {
            var result = ClientProbe.exercise(p.rpcAddress(), d.rpcAddress());
            assertEquals(2, result.get("schedule_calls"));
            assertEquals(3, ((NativeClient.Result) result.get("generate")).outputTokens());
            assertEquals(3, ((NativeClient.Result) result.get("fetch")).outputTokens());
        }
    }

    LLMMockEngine node(String role) throws Exception { return node(role, 4096); }

    LLMMockEngine node(String role, int maxSequence) throws Exception {
        var path = directory.resolve(role + ".json");
        new ObjectMapper().writeValue(path.toFile(), Map.of("id", role + "-0", "component_role", role,
            "component", Map.of("engine", "rtp", "config", Map.of("mock.scheduler.mode", "group", "seq_size_per_block", 16, "test_block_num", 9,
                "max_seq_len", maxSequence, "reserve_block_ratio", 0, "mock.compute", Map.of("forward", Map.of("expression", "5")))), "environment", Map.of()));
        return LLMMockEngine.start(RtpConfig.load(path));
    }

    @Test void contextBoundedGenerateAndFetchEndNormallyIncludingEmptyTerminalFrames() throws Exception {
        for (int sequence : new int[]{5, 6}) try (var p = node("prefill"); var d = node("decode", sequence)) {
            var result = ClientProbe.exercise(p.rpcAddress(), d.rpcAddress(), sequence - 4);
            assertEquals(sequence - 4, ((NativeClient.Result) result.get("generate")).outputTokens());
            assertEquals(sequence - 4, ((NativeClient.Result) result.get("fetch")).outputTokens());
        }
    }

}
