package org.flexlb.dispatcher;

import org.springframework.http.HttpHeaders;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Relay end-to-end headers, excluding framing, Connection-nominated fields and caller routing
 * credentials.
 */
final class DispatcherHeaders {

    static final String TRUSTED_ROUTING_HEADER = "X-Rtp-Llm-Dispatcher-Routing-Token";

    private DispatcherHeaders() {
    }

    /** RFC 7230 hop-by-hop headers and framing computed by the outbound connection. */
    static final Set<String> HOP_BY_HOP = caseInsensitiveSet(Set.of(),
            "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
            "transfer-encoding", "upgrade", "host", "content-length");

    /** Caller-controlled copies of the internal trust header must never cross into FE. */
    static final Set<String> TO_FE_SKIP = caseInsensitiveSet(
            HOP_BY_HOP, TRUSTED_ROUTING_HEADER);

    /** Fanout rebuilds JSON bodies; raw response bytes must not be compressed by the FE. */
    static final Set<String> FANOUT_SKIP = caseInsensitiveSet(
            TO_FE_SKIP, "content-type", "accept-encoding");

    /**
     * Copy end-to-end headers while also honoring fields dynamically nominated by {@code Connection},
     * which are hop-by-hop even when absent from the fixed standard list.
     */
    static void copyEndToEnd(HttpHeaders source, HttpHeaders sink, Set<String> skip) {
        Set<String> effectiveSkip = skip;
        List<String> connectionValues = source.get(HttpHeaders.CONNECTION);
        if (connectionValues != null) {
            Set<String> withConnectionTokens = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
            withConnectionTokens.addAll(skip);
            for (String value : connectionValues) {
                for (String token : value.split(",")) {
                    String name = token.trim();
                    if (!name.isEmpty()) {
                        withConnectionTokens.add(name);
                    }
                }
            }
            effectiveSkip = withConnectionTokens;
        }
        Set<String> namesToSkip = effectiveSkip;
        source.forEach((name, values) -> {
            if (!namesToSkip.contains(name)) {
                sink.addAll(name, values);
            }
        });
    }

    /** Case-insensitive membership avoids per-header string allocation. */
    private static Set<String> caseInsensitiveSet(Set<String> base, String... extra) {
        Set<String> set = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
        set.addAll(base);
        set.addAll(Arrays.asList(extra));
        return Collections.unmodifiableSet(set);
    }
}
