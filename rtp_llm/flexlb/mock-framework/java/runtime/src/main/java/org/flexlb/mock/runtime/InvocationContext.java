package org.flexlb.mock.runtime;

import java.time.Instant;
import java.util.Map;

/** Invocation metadata, separate from business arguments and never a second API definition. */
public record InvocationContext(String run, Instant queryTime, Map<String, Object> observation, long timeoutMs) {
    private static final ThreadLocal<InvocationContext> CURRENT = new ThreadLocal<>();

    public static InvocationContext current() {
        var value = CURRENT.get();
        if (value == null) throw new IllegalStateException("No active API invocation");
        return value;
    }

    static void enter(InvocationContext value) { CURRENT.set(value); }
    static void leave() { CURRENT.remove(); }
}
