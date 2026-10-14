package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.Set;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MockFeatureContractsTest {
    @Test void everyImplementationHasAnExplicitScopeAndRealChangesRequireReconciliation() throws Exception {
        Path root = Path.of("").toAbsolutePath();
        while (root != null && !Files.isDirectory(root.resolve("rtp_llm/cpp"))) root = root.getParent();
        assertNotNull(root, "feature contracts require the full source checkout");
        Path module = root.resolve("rtp_llm/flexlb/flexlb-mock-engine");
        var document = new ObjectMapper().readTree(getClass().getResourceAsStream("/mock-feature-contracts.json"));
        assertEquals(1, document.path("schema_version").asInt());
        Set<String> declared = new HashSet<>();
        for (var feature : document.path("features")) {
            assertTrue(Set.of("contract", "model", "harness").contains(feature.path("fidelity").asText()));
            assertFalse(feature.path("scope").asText().isBlank());
            assertFalse(feature.path("tests").isEmpty());
            for (var implementation : feature.path("implementation")) {
                assertTrue(declared.add(implementation.asText()), "duplicate implementation owner");
            }
            for (var test : feature.path("tests")) {
                String source = Files.readString(module.resolve("src/test/java/org/flexlb/mockengine").resolve(test.asText()));
                assertTrue(source.contains("@Test") || source.contains("@ParameterizedTest"));
            }
            var real = feature.path("real").fields();
            while (real.hasNext()) {
                var entry = real.next();
                String actual = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                        .digest(Files.readAllBytes(root.resolve(entry.getKey()))));
                assertEquals(entry.getValue().asText(), actual,
                        "real source changed: reconcile mock contract and regression vectors for " + entry.getKey());
            }
        }
        Set<String> actual = new HashSet<>();
        try (var files = Files.list(module.resolve("src/main/java/org/flexlb/mockengine"))) {
            files.filter(path -> path.toString().endsWith(".java")).forEach(path -> actual.add(path.getFileName().toString()));
        }
        assertEquals(actual, declared, "new/removed implementation needs an explicit feature contract");
    }
}
