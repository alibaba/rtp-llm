package org.flexlb.mock.client;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.Map;
import org.flexlb.mock.engine.rtp.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class NativeRtpTest {
    @TempDir Path directory;
    RtpNode node(String role) throws Exception {
        var path = directory.resolve(role + ".json");
        new ObjectMapper().writeValue(path.toFile(), Map.of("id", role + "-0", "component_role", role,
            "component", Map.of("engine", "rtp", "config", Map.of("seq_size_per_block", 16, "test_block_num", 9,
                "max_seq_len", 4096, "reserve_block_ratio", 0, "compute", Map.of("expression", "5"))), "environment", Map.of()));
        return RtpNode.start(RtpConfig.load(path));
    }
    @Test void nativeClientConsumesBothGenerateAndEnqueueFetchFromActualRtpNodes() throws Exception {
        try (var p = node("prefill"); var d = node("decode")) {
            var result = ClientProbe.exercise(p.rpcAddress(), d.rpcAddress());
            assertEquals(2, result.get("schedule_calls"));
            assertEquals(3, ((NativeClient.Result) result.get("generate")).outputTokens());
            assertEquals(3, ((NativeClient.Result) result.get("fetch")).outputTokens());
        }
    }
}
