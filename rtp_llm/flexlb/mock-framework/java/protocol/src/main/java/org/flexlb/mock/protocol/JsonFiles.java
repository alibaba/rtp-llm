package org.flexlb.mock.protocol;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Publishes JSON files atomically so readers never see a partial write.
 */
public final class JsonFiles {
    private static final ObjectWriter WRITER = new ObjectMapper().writer();

    private JsonFiles() {}

    public static void writeAtomically(Path target, Object value) throws IOException {
        writeAtomically(target, value, WRITER, false);
    }

    public static void writeAtomically(Path target, Object value, ObjectWriter writer, boolean replaceExisting) throws IOException {
        Path temporary = Files.createTempFile(target.toAbsolutePath().getParent(), ".json-", ".tmp");
        try {
            writer.writeValue(temporary.toFile(), value);
            if (replaceExisting) {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } else {
                Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }
}
