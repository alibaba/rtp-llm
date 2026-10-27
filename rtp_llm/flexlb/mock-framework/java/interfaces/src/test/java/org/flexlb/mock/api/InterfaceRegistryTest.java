package org.flexlb.mock.api;

import org.junit.jupiter.api.Test;
import org.flexlb.mock.api.model.Types;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.IOException;
import java.util.TreeMap;
import static org.junit.jupiter.api.Assertions.*;

class InterfaceRegistryTest {
    interface Extension {
        record Args(int threshold) {}
        @Endpoint(value="extension/queue", kind=Endpoint.Kind.read)
        Types.MetricValue queue(Args args);
    }
    interface Duplicate {
        @Endpoint(value="extension/queue", kind=Endpoint.Kind.action)
        Types.None queue(Types.None args);
    }
    interface Malformed { Types.None missing(Types.None args); }

    @Test void exportsComponentInterfacesWithoutOldTtftAliases() {
        var catalog = InterfaceRegistry.standard();
        assertEquals(18, catalog.size()); // Python owns runtime/wait.
        assertTrue(catalog.containsKey("client/requests/ttft"));
        assertFalse(catalog.containsKey("client/requests/ttft_max"));
        assertTrue(catalog.containsKey("prometheus/query"));
        assertFalse(catalog.containsKey("generator/generate"));
    }
    @Test void newMethodAutomaticallyExportsItsArgumentSchema() throws IOException {
        var extra = InterfaceRegistry.describe(Extension.class);
        var spec = (Map<?, ?>) extra.get("extension/queue");
        var args = (Map<?, ?>) spec.get("args");
        var fields = (Map<?, ?>) args.get("properties");
        assertTrue(fields.containsKey("threshold"));
        assertTrue(CatalogMain.json(extra).contains("extension/queue"));
        var combined = new TreeMap<>(InterfaceRegistry.standard());
        combined.putAll(extra);
        Files.writeString(Path.of("target/test-extension-catalog.json"),
                CatalogMain.json(Map.of("interfaces", combined, "configuration", InterfaceRegistry.configuration())));
    }
    @Test void rejectsDuplicateAndUnannotatedMethods() {
        assertThrows(IllegalArgumentException.class, () -> InterfaceRegistry.describe(Extension.class, Duplicate.class));
        assertThrows(IllegalArgumentException.class, () -> InterfaceRegistry.describe(Malformed.class));
    }
    @Test void serializesControlCharactersInCatalog() {
        assertEquals("\"a\\\"\\n\\u0001\"", CatalogMain.json("a\"\n\u0001"));
    }

    @Test void exportsOutputDistributionVariantsAndTheirConstraints() {
        String catalog = CatalogMain.json(InterfaceRegistry.standard().get("client/traffic/run"));
        for (String value : new String[]{"uniform", "normal", "ordered"})
            assertTrue(catalog.contains("\"" + value + "\""), value);
        assertTrue(catalog.contains("\"bits\":64")); // Seed is an int64, not a float.
    }

    @Test void computeConfigurationIsExportedWithoutInventingBusinessEndpoints() {
        String config = CatalogMain.json(InterfaceRegistry.configuration());
        assertTrue(config.contains("\"prefill\""));
        assertTrue(config.contains("\"stddev_ratio\""));
        assertTrue(config.contains("\"max_deviation_ms\""));
        assertEquals(18, InterfaceRegistry.standard().size());
    }
}
