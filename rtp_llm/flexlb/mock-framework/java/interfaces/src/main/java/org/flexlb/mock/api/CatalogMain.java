package org.flexlb.mock.api;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/** Export command; running it does not start any component. */
public final class CatalogMain {
    private CatalogMain() {}
    public static void main(String[] args) throws IOException {
        if (args.length != 1) throw new IllegalArgumentException("Usage: CatalogMain OUTPUT.json");
        Path output = Path.of(args[0]);
        if (output.toAbsolutePath().getParent() != null) Files.createDirectories(output.toAbsolutePath().getParent());
        Files.writeString(output, json(Map.of("interfaces", InterfaceRegistry.standard())) + "\n", StandardCharsets.UTF_8);
    }

    static String json(Object value) {
        if (value instanceof String s) {
            StringBuilder out = new StringBuilder("\"");
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '"' -> out.append("\\\"");
                    case '\\' -> out.append("\\\\");
                    case '\n' -> out.append("\\n");
                    case '\r' -> out.append("\\r");
                    case '\t' -> out.append("\\t");
                    default -> { if (c < 32) out.append(String.format("\\u%04x", (int)c)); else out.append(c); }
                }
            }
            return out.append('"').toString();
        }
        if (value instanceof Map<?, ?> map) {
            List<String> entries = new ArrayList<>();
            map.entrySet().stream().sorted(Comparator.comparing(e -> e.getKey().toString()))
                    .forEach(e -> entries.add(json(e.getKey().toString()) + ":" + json(e.getValue())));
            return "{" + String.join(",", entries) + "}";
        }
        if (value instanceof Collection<?> list) return "[" + String.join(",", list.stream().map(CatalogMain::json).toList()) + "]";
        if (value instanceof Boolean || value instanceof Number) return value.toString();
        throw new IllegalArgumentException("Unsupported catalog value: " + value);
    }
}
