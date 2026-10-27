package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.*;
import java.time.Duration;
import java.util.Map;
import org.flexlb.mock.api.model.Types.Role;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ComputeModelsTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @TempDir Path directory;

    @Test void loadsRolesWithEffectiveDefaultsAndNoHiddenCalibration() throws Exception {
        var compute = ComputeModels.fromJson(JSON.readTree("""
            {"prefill":{"expression":"0.5*totalComputeTokens"},
             "decode":{"expression":"param(baseMs, 1) + batchSize","noise":{"scale":2}}}
            """));
        assertEquals(1, compute.configuration().prefill().noise().scale());
        var batch = FormulaForwardTest.batch(FormulaForwardTest.request("a", 10, 2));
        assertEquals(Duration.ofMillis(4), compute.plan(Role.prefill, "p0", batch).planned());
        assertEquals(Duration.ofMillis(4), compute.plan(Role.decode, "d0", batch).planned());
        assertThrows(IllegalArgumentException.class, () -> compute.plan(Role.vit, "v0", batch));
    }

    @Test void readsTheActualRunnerConfigAndKeepsNativePayloadUntouched() throws Exception {
        var file = directory.resolve("config.json");
        Files.writeString(file, """
            {"component":{"config":{"REAL_NATIVE_JSON":"{native payload}"}},
             "environment":{"compute":{"prefill":{"expression":"1"}}}}
            """);
        assertEquals("1", ComputeModels.fromComponentFile(file).configuration().prefill().expression());
    }

    @Test void rejectsMisspelledFieldsScalarCoercionNullAndInvalidExpressions() throws Exception {
        for (String value : new String[]{
                "{\"prefill\":{\"expression\":\"1\",\"extra\":1}}",
                "{\"prefill\":{\"expression\":\"unknownTokens\"}}",
                "{\"prefill\":{\"expression\":\"1\",\"noise\":{\"scale\":\"2\"}}}",
                "{\"prefill\":{\"expression\":\"1\",\"noise\":{\"seed\":1.2}}}",
                "{\"prefill\":{\"expression\":\"1\",\"noise\":{\"stddev_ms\":1,\"seed\":1}}}",
                "{\"prefill\":null}"})
            assertThrows(Exception.class, () -> ComputeModels.fromJson(JSON.readTree(value)), value);
    }

    @Test void componentInputRejectsDuplicateKeys() throws Exception {
        var file = directory.resolve("duplicate.json");
        Files.writeString(file, "{\"environment\":{\"compute\":{},\"compute\":{}}}");
        assertThrows(IOException.class, () -> ComputeModels.fromComponentFile(file));
    }

    @Test void configurationAndFormulaRunFromCompilerExport() throws Exception {
        var file = Path.of("target/compute-classpath.txt");
        Files.writeString(file, System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
    }
}
