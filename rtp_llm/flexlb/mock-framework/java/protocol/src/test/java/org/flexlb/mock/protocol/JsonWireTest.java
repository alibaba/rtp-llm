package org.flexlb.mock.protocol;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.core.type.TypeReference;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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

    @Test void failedWriteLeavesPublishedJsonAndRemovesTemporaryFile(@TempDir Path directory) throws Exception {
        Path target = directory.resolve("result.json");
        JsonWire.writeAtomically(target, Map.of("count", 1));
        assertThrows(IOException.class, () -> JsonWire.writeAtomically(target, Collections.singletonMap(null, 2),
            JsonWire.JSON.writerWithDefaultPrettyPrinter(), true));
        assertEquals(1, JsonWire.JSON.readTree(target.toFile()).get("count").intValue());
        try (var files = Files.list(directory)) {
            assertEquals(List.of(target), files.toList());
        }
    }

}
