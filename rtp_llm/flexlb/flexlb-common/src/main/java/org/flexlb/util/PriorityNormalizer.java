package org.flexlb.util;

/**
 * Resolves priority from a valid protocol value, then the QoS metadata header, then the default
 * level 50. Valid levels are 1–100; zero and invalid protocol values fall through to the header.
 */
public final class PriorityNormalizer {
    /** gRPC metadata header carrying the caller QoS level (fallback channel). */
    public static final String QOS_HEADER_NAME = "x-dashscope-inner-qos-level";

    public static final int MIN_PRIORITY = 1;
    public static final int MAX_PRIORITY = 100;
    public static final int DEFAULT_PRIORITY = 50;

    /**
     * Sentinel for "request carried no priority". Never returned by
     * {@link #normalize(int, String, int)} — which always yields 1-100 —
     * so every normalized request participates in Auto-TPM. Kept as a
     * defensive guard and for unit-test assertions.
     */
    public static final int NO_PRIORITY = 0;

    private PriorityNormalizer() {}

    /**
     * Returns true iff the value denotes an explicit priority (i.e. is not
     * the {@link #NO_PRIORITY} sentinel). Always true for values returned
     * by {@link #normalize(int, String, int)}.
     *
     * <p><b>This is the single home of the "has a priority" concept.</b>
     * The only place a 0 priority can legitimately appear downstream is on
     * items admitted by the legacy (non-Auto-TPM) path before the switch was
     * flipped; such items never participate in any priority mechanism.
     * Downstream code MUST call this method instead of hand-rolling
     * {@code priority > 0} / {@code priority == 0} checks.
     */
    public static boolean hasPriority(int normalizedPriority) {
        return normalizedPriority != NO_PRIORITY;
    }

    /**
     * Normalize with the built-in default (50).
     */
    public static int normalize(int protoPriority, String headerValue) {
        return normalize(protoPriority, headerValue, DEFAULT_PRIORITY);
    }

    /**
     * Normalize the request priority.
     *
     * @param protoPriority   value from the proto field; 0 means unset
     * @param headerValue     raw {@code x-dashscope-inner-qos-level} header value, may be null/blank
     * @param defaultPriority configured default; itself normalized to 50 when invalid
     * @return the default priority when neither channel carries a value,
     *         otherwise a valid level in 1-100
     */
    public static int normalize(int protoPriority, String headerValue, int defaultPriority) {
        if (isValid(protoPriority)) {
            return protoPriority;
        }
        int headerPriority = parseHeader(headerValue);
        if (isValid(headerPriority)) {
            return headerPriority;
        }
        return isValid(defaultPriority) ? defaultPriority : DEFAULT_PRIORITY;
    }

    /**
     * Returns true iff the value is a valid priority level.
     * Valid range is 1-100; 0 is the NO_PRIORITY sentinel, negatives and
     * values above 100 are invalid.
     */
    public static boolean isValid(int priority) {
        return priority >= MIN_PRIORITY && priority <= MAX_PRIORITY;
    }

    /** Parse caller QoS without assigning a default scheduling priority. */
    public static int parseHeader(String headerValue) {
        if (headerValue != null) {
            try {
                int headerPriority = Integer.parseInt(headerValue.trim());
                if (isValid(headerPriority)) {
                    return headerPriority;
                }
            } catch (NumberFormatException ignored) {
            }
        }
        return NO_PRIORITY;
    }
}
