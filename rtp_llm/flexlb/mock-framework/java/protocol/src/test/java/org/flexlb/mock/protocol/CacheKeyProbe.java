package org.flexlb.mock.protocol;

import java.util.*;

/** Cross-language test entry point; uses the same immutable input as Client and Engine. */
public final class CacheKeyProbe {
    private static List<Integer> ints(String csv) {
        return csv.isEmpty() ? List.of() : Arrays.stream(csv.split(",")).map(Integer::valueOf).toList();
    }
    public static void main(String[] args) {
        var spans = new ArrayList<CacheSequence.Span>();
        if (!args[2].isEmpty()) for (String span : args[2].split(";")) {
            String[] pair = span.split(":", 2);
            spans.add(new CacheSequence.Span(Integer.parseInt(pair[0]), ints(pair[1])));
        }
        var sequence = new CacheSequence(Integer.parseInt(args[0]), ints(args[1]), spans, Long.parseLong(args[3]));
        System.out.println(sequence.keys().stream().map(String::valueOf).collect(java.util.stream.Collectors.joining(",")));
    }
}
