package org.flexlb.dispatcher;

import org.springframework.http.HttpHeaders;

import java.util.Arrays;
import java.util.Collections;
import java.util.Set;
import java.util.TreeSet;

/**
 * Which inbound headers the dispatcher relays to FE, shared by the passthrough and fanout paths so
 * the two cannot drift: a caller must not lose its {@code Authorization}, tenant or tracing headers
 * merely because its request happened to be batch-shaped and took the split path.
 */
final class DispatcherHeaders {

    private DispatcherHeaders() {
    }

    /**
     * Hop-by-hop headers from RFC 7230 §6.1 plus framing headers WebClient must compute itself for
     * the outbound connection. Forwarding any of these from the inbound request — or back on the
     * response — corrupts the new connection: an inbound {@code Transfer-Encoding: chunked}
     * double-frames the body WebClient is already about to chunk-encode; an inbound {@code Host}
     * routes to whatever the original client put there; {@code Proxy-Authorization} would be
     * relayed downstream against the original intent. Comparison is case-insensitive.
     */
    static final Set<String> HOP_BY_HOP = caseInsensitiveSet(
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
            "host",
            "content-length");

    /**
     * Fanout drops everything hop-by-hop plus two more, because unlike passthrough it does not
     * stream bytes through — it parses each FE response and re-serializes a merged one:
     * <ul>
     *   <li>{@code accept-encoding} — the dispatcher reads the FE body as raw bytes and hands them
     *       to {@code JSON.parseObject}; letting FE gzip the response would break that parse.</li>
     *   <li>{@code content-length} / {@code content-type} — each chunk body is re-serialized here,
     *       so the inbound values describe the wrong entity ({@code content-type} is set explicitly
     *       by {@link FeClient}; {@code content-length} is already hop-by-hop).</li>
     * </ul>
     */
    static final Set<String> FANOUT_SKIP = caseInsensitiveSet(
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
            "host",
            "content-length",
            "content-type",
            "accept-encoding");

    /** Copy every header from {@code source} into {@code sink} except the names in {@code skip}. */
    static void copyEndToEnd(HttpHeaders source, HttpHeaders sink, Set<String> skip) {
        source.forEach((name, values) -> {
            if (!skip.contains(name)) {
                sink.addAll(name, values);
            }
        });
    }

    /** Case-insensitive membership without the per-header {@code toLowerCase} allocation. */
    private static Set<String> caseInsensitiveSet(String... names) {
        Set<String> set = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
        set.addAll(Arrays.asList(names));
        return Collections.unmodifiableSet(set);
    }
}
