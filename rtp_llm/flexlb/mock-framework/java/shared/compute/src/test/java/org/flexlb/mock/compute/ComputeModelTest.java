package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.time.Duration;
import org.flexlb.mock.api.model.Types.Role;
import org.junit.jupiter.api.Test;

class ComputeModelTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private ComputeModel load(String value) throws Exception { return ComputeModel.fromJson(JSON.readTree(value)); }
    @Test void oneComponentOwnsForwardProgressAndEos() throws Exception {
        var compute = load("{\"forward\":{\"expression\":\"0.5*totalComputeTokens\"}}");
        assertEquals(0, compute.configuration().forward().noise().stddev_ms());
        var batch = FormulaForwardTest.batch(FormulaForwardTest.request("a", 10, 2));
        assertEquals(Duration.ofMillis(4), compute.plan(Role.prefill, "p0", batch).planned());
        assertEquals(new ComputeModel.Result(1, false), compute.completed(compute.prepare(Role.prefill,
            new ComputeModel.Request("r", 0, 8, 0, 0, 4, 100, false))));
    }
    @Test void eosRunsAfterAcceptanceAndDoesNotRefundVerification() throws Exception {
        var compute = load("{\"forward\":{\"expression\":\"1\"},\"decode\":{\"draft_tokens\":4,\"acceptance_rate\":1},\"eos\":{\"probability\":1}}");
        var plan = compute.prepare(Role.decode, new ComputeModel.Request("r", 0, 8, 1, 3, 20, 100, false));
        assertEquals(5, plan.verifyTokens()); assertEquals(5, plan.additionalKvTokens());
        assertEquals(new ComputeModel.Result(2, true), compute.completed(plan));
        assertEquals(5, plan.verifyTokens());
    }
    @Test void maximumSequenceAndOutputLimitsWinEvenWithEosIgnored() throws Exception {
        var compute = load("{\"forward\":{\"expression\":\"1\"},\"decode\":{\"tokens_per_step\":5},\"eos\":{\"probability\":1}}");
        assertEquals(new ComputeModel.Result(4, true), compute.completed(compute.prepare(Role.decode,
            new ComputeModel.Request("r", 0, 8, 1, 0, 5, 100, true))));
        assertEquals(new ComputeModel.Result(3, true), compute.completed(compute.prepare(Role.decode,
            new ComputeModel.Request("r", 0, 8, 1, 0, 20, 12, true))));
        assertThrows(IllegalArgumentException.class, () -> new ComputeModel.Request("r", 0, 8, 0, 0, 1, 8, false));
    }
    @Test void rejectsLegacyLocationsMisspellingsAndInvalidNestedModels() throws Exception {
        for (String value : new String[]{"{}", "{\"expression\":\"1\"}", "{\"forward\":{\"expression\":\"1\",\"scale\":2}}",
                "{\"forward\":{\"expression\":\"unknownTokens\"}}", "{\"forward\":{\"expression\":\"1\",\"noise\":{\"seed\":1.2}}}",
                "{\"forward\":{\"expression\":\"1\",\"unexpected\":1}}",
                "{\"forward\":{\"expression\":\"1\",\"noise\":{\"stddev_ms\":-1,\"max_deviation_ms\":1,\"seed\":42}}}",
                "{\"forward\":{\"expression\":\"1\",\"noise\":{\"scale\":2}}}",
                "{\"forward\":{\"expression\":\"1\",\"noise\":{\"seed\":9223372036854775808}}}",
                "{\"forward\":{\"expression\":\"1\",\"noise\":null}}",
                "{\"forward\":{\"expression\":\"1\"},\"decode\":{\"draft_tokens\":-1,\"acceptance_rate\":1,\"seed\":42}}",
                "{\"forward\":{\"expression\":\"1\"},\"decode\":{\"draft_tokens\":4,\"acceptance_rate\":1.01,\"seed\":42}}",
                "{\"forward\":{\"expression\":\"1\"},\"decode\":{\"draft_tokens\":4,\"acceptance_rate\":1,\"seed\":1.5}}",
                "{\"forward\":{\"expression\":\"1\"},\"eos\":{\"probability\":-0.1,\"seed\":42}}",
                "{\"forward\":{\"expression\":\"1\"},\"eos\":{\"probability\":1.1,\"seed\":42}}",
                "{\"forward\":{\"expression\":\"1\"},\"eos\":{\"unknown\":1}}",
                "{\"forward\":{\"expression\":\"1\"},\"eos\":{\"probability\":0.1}}", "{\"forward\":null}",
                "{\"forward\":{\"expression\":\"1\"},\"decode\":{\"draft_tokens\":3,\"tokens_per_step\":2.6}}"})
            assertThrows(Exception.class, () -> load(value), value);
    }
    @Test void exportsTestClasspath() throws Exception {
        Files.writeString(Path.of("target/compute-classpath.txt"), System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
    }
}
