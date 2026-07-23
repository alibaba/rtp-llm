package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Every {@code .java} file across the flexlb modules must be plain text. A raw control byte (this
 * actually shipped once: NUL separators written as literal 0x00 inside char literals in
 * {@link DispatcherMetricsReporter}) makes Git classify the file as binary — diffs collapse to
 * "Binary files differ", numstat shows {@code -/-}, text search skips it — so the file silently
 * drops out of line-by-line review and static tooling. Control characters belong in escape
 * sequences ({@code '\u0000'}), never as raw bytes.
 */
class SourceTreeIsTextTest {

    @Test
    void noJavaSourceInAnyModuleContainsRawControlBytes() throws IOException {
        // Surefire runs with the module directory as the working directory; sibling modules'
        // src/ dirs (flexlb-common, flexlb-sync, flexlb-grpc, flexlb-cache carry no equivalent
        // guard of their own) are discovered one level up. Only src/ is walked, never a
        // module's target/ or other build output.
        assertTrue(Files.isDirectory(Path.of("src")),
                "expected to run from the module root (src/ not found)");
        List<Path> sourceRoots = discoverModuleSourceRoots();
        assertTrue(sourceRoots.size() >= 2,
                "expected to discover sibling modules' src/ dirs, found only: " + sourceRoots);

        List<String> offenders = new ArrayList<>();
        for (Path root : sourceRoots) {
            try (Stream<Path> files = Files.walk(root)) {
                files.filter(p -> p.toString().endsWith(".java")).forEach(p -> {
                    try {
                        byte[] bytes = Files.readAllBytes(p);
                        for (int i = 0; i < bytes.length; i++) {
                            int b = bytes[i] & 0xFF;
                            boolean allowed = b >= 0x20 || b == '\t' || b == '\n' || b == '\r';
                            if (!allowed) {
                                offenders.add(String.format("%s: raw byte 0x%02X at offset %d", p, b, i));
                                break;
                            }
                        }
                    } catch (IOException e) {
                        offenders.add(p + ": unreadable (" + e.getMessage() + ")");
                    }
                });
            }
        }
        assertTrue(offenders.isEmpty(),
                "Java sources must be plain text; write control characters as escapes, not raw bytes:\n"
                        + String.join("\n", offenders));
    }

    /**
     * Every {@code ../<module>/src} that exists — this module's own {@code src} included, since
     * the module itself is a child of {@code ..}. Sorted so walk order (and any failure message)
     * is deterministic across filesystems.
     */
    private static List<Path> discoverModuleSourceRoots() throws IOException {
        try (Stream<Path> siblings = Files.list(Path.of(".."))) {
            return siblings
                    .filter(Files::isDirectory)
                    .map(module -> module.resolve("src"))
                    .filter(Files::isDirectory)
                    .sorted()
                    .toList();
        }
    }
}
