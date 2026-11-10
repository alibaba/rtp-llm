package org.flexlb.mock.protocol;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.core.type.TypeReference;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Checks JSON encoding of values sent over the Mock control API.
 */
class JsonWireTest {
    @Test void numericConstraintsMatchTheSharedPythonFixtures() throws Exception {
        var rows = JsonWire.JSON.readTree(Path.of("../../python/mock_framework/execution/test/fixtures/wire_numbers.json").toFile());
        for (var row : rows) {
            Map<String, Object> schema = JsonWire.JSON.convertValue(row.get("schema"), new TypeReference<>() {});
            if (row.get("valid").booleanValue())
                assertDoesNotThrow(() -> JsonWire.validate(row.get("value"), schema), row.toString());
            else assertThrows(IllegalArgumentException.class, () -> JsonWire.validate(row.get("value"), schema), row.toString());
        }
    }
}
