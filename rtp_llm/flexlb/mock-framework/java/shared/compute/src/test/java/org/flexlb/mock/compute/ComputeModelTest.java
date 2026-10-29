package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.time.Duration;
import org.flexlb.mock.api.model.Types.Role;
import org.junit.jupiter.api.Test;

class ComputeModelTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @Test void oneComponentOwnsOneFormulaWithEffectiveDefaults() throws Exception {
        var compute = ComputeModel.fromJson(JSON.readTree("{\"expression\":\"0.5*totalComputeTokens\"}"));
        assertEquals(0, compute.configuration().noise().stddev_ms());
        var batch = FormulaForwardTest.batch(FormulaForwardTest.request("a", 10, 2));
        assertEquals(Duration.ofMillis(4), compute.plan(Role.prefill, "p0", batch).planned());
    }
    @Test void rejectsMisspelledFieldsCoercionNullAndBadFormula() throws Exception {
        for (String value : new String[]{"{\"expression\":\"1\",\"extra\":1}", "{\"expression\":\"unknownTokens\"}",
                "{\"expression\":\"1\",\"noise\":{\"scale\":2}}", "{\"expression\":\"1\",\"scale\":2}", "{\"expression\":\"1\",\"noise\":{\"seed\":1.2}}",
                "{\"expression\":\"1\",\"noise\":{\"stddev_ms\":1,\"seed\":1}}", "{\"expression\":null}", "{\"prefill\":{\"expression\":\"1\"}}"})
            assertThrows(Exception.class, () -> ComputeModel.fromJson(JSON.readTree(value)), value);
    }
    @Test void exportsTestClasspath() throws Exception {
        Files.writeString(Path.of("target/compute-classpath.txt"), System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
    }
}
