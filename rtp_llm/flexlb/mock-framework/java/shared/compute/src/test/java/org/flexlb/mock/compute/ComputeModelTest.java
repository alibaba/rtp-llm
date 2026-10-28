package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.*;
import java.time.Duration;
import org.flexlb.mock.api.model.Types.Role;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ComputeModelTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @TempDir Path directory;
    @Test void oneComponentOwnsOneFormulaWithEffectiveDefaults() throws Exception {
        var compute = ComputeModel.fromJson(JSON.readTree("{\"expression\":\"0.5*totalComputeTokens\"}"));
        assertEquals(1, compute.configuration().noise().scale());
        var batch = FormulaForwardTest.batch(FormulaForwardTest.request("a", 10, 2));
        assertEquals(Duration.ofMillis(4), compute.plan(Role.prefill, "p0", batch).planned());
    }
    @Test void readsGroupConfigAndRejectsOldTopLevelEvenWhenBothExist() throws Exception {
        var file = directory.resolve("config.json");
        Files.writeString(file, "{\"component\":{\"config\":{\"compute\":{\"expression\":\"1\"}}}}");
        assertEquals("1", ComputeModel.fromComponentFile(file).configuration().expression());
        Files.writeString(file, "{\"component\":{\"config\":{\"compute\":{\"expression\":\"1\"}}},\"environment\":{\"compute\":{}}}");
        assertThrows(IllegalArgumentException.class, () -> ComputeModel.fromComponentFile(file));
    }
    @Test void rejectsMisspelledFieldsCoercionNullAndBadFormula() throws Exception {
        for (String value : new String[]{"{\"expression\":\"1\",\"extra\":1}", "{\"expression\":\"unknownTokens\"}",
                "{\"expression\":\"1\",\"noise\":{\"scale\":\"2\"}}", "{\"expression\":\"1\",\"noise\":{\"seed\":1.2}}",
                "{\"expression\":\"1\",\"noise\":{\"stddev_ms\":1,\"seed\":1}}", "{\"expression\":null}", "{\"prefill\":{\"expression\":\"1\"}}"})
            assertThrows(Exception.class, () -> ComputeModel.fromJson(JSON.readTree(value)), value);
    }
    @Test void componentInputRejectsDuplicateKeys() throws Exception {
        var file = directory.resolve("duplicate.json");
        Files.writeString(file, "{\"component\":{\"config\":{\"compute\":{},\"compute\":{}}}}");
        assertThrows(IOException.class, () -> ComputeModel.fromComponentFile(file));
    }
    @Test void exportsTestClasspath() throws Exception {
        Files.writeString(Path.of("target/compute-classpath.txt"), System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
    }
}
