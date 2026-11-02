package org.flexlb.mock.runtime;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.core.type.TypeReference;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;

class JsonWireTest {
    @Test void numericConstraintsMatchTheSharedPythonFixtures() throws Exception {
        var rows = JsonWire.JSON.readTree(Path.of("../../tests/fixtures/wire_numbers.json").toFile());
        for (var row : rows) {
            Map<String, Object> schema = JsonWire.JSON.convertValue(row.get("schema"), new TypeReference<>() {});
            if (row.get("valid").booleanValue())
                assertDoesNotThrow(() -> JsonWire.validate(row.get("value"), schema), row.toString());
            else assertThrows(IllegalArgumentException.class, () -> JsonWire.validate(row.get("value"), schema), row.toString());
        }
    }
}
