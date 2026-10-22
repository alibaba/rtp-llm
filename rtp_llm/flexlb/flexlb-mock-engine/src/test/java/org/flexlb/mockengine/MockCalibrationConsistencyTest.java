package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MockCalibrationConsistencyTest {
    @TempDir Path directory;

    @Test
    void standaloneFlatConfigUsesSameFormulaAsLegacyEnvelope() throws Exception {
        Path master = directory.resolve("master.json");
        Path performance = directory.resolve("performance.json");
        Files.writeString(performance, "{}");
        String expression = "max(170, 0.0045 * sum(computeTokens))";
        MockMasterConfig.writeWithPrefillExpression(master, expression);
        var mapper = new ObjectMapper();
        var envelope = mapper.readTree(master.toFile());
        var legacy = MockPerformanceModel.load(performance.toString(), master.toString());
        String flat = envelope.path("zone_process_setting").path("process_info")
                .path("envs").get(0).get(1).asText();
        Files.writeString(master, flat);
        var standalone = MockPerformanceModel.load(performance.toString(), master.toString());
        assertEquals(legacy.prefillExpressionState(), standalone.prefillExpressionState());
    }

    @Test
    void packagedCalibrationMustMatchMaterializedRun() throws Exception {
        byte[] resource;
        try (var stream = getClass().getClassLoader()
                .getResourceAsStream("dsv4_l20_mock_calibration.json")) {
            if (stream == null) throw new IllegalStateException("calibration resource missing");
            resource = stream.readAllBytes();
        }
        String sha = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(resource));
        var expected = new ObjectMapper().readTree(resource);
        Path master = directory.resolve("master.json");
        Path performance = directory.resolve("performance.json");
        MockMasterConfig.writeWithPrefillExpression(master,
                expected.path("prefill_expression").asText());
        var mapper = new ObjectMapper();
        mapper.writeValue(performance.toFile(), Map.of("calibration_sha256", sha));
        var model = MockPerformanceModel.load(performance.toString(), master.toString());
        assertEquals(expected.path("decode").path("tokens_per_step").asDouble(),
                model.tokensPerStep());

        Files.writeString(performance, "{\"calibration_sha256\":\"stale\"}");
        var error = assertThrows(IllegalStateException.class,
                () -> MockPerformanceModel.load(performance.toString(), master.toString()));
        assertTrue(error.getMessage().contains("calibration_sha256 disagrees"));
    }
}
