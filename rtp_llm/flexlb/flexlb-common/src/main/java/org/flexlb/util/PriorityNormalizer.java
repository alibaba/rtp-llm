package org.flexlb.util;

/**
 * Normalizes the Auto-TPM QoS priority carried by a request.
 *
 * <p>Valid priority levels are 1-100 (higher = more important).
 * Resolution order: proto field (when it carries a valid value) &gt; gRPC
 * metadata header {@code x-dashscope-inner-qos-level} &gt; default (50).
 * A proto value of 0 means "unset" (proto3 default) and falls through to the
 * header; any other invalid value also falls through.
 *
 * <p>When the request carries no priority at all (proto unset AND
 * header absent/blank) the result is the default priority (50), so the
 * request participates in Auto-TPM scheduling at the normal level. Only
 * explicitly carried but invalid values also fall back to the default.
 */
public final class PriorityNormalizer {

    /** gRPC metadata header carrying the caller QoS level (fallback channel). */
    public static final String QOS_HEADER_NAME = "x-dashscope-inner-qos-level";

    public static final int DEFAULT_PRIORITY = 50;

    /**
     * Sentinel for "request carried no priority". Never returned by
     * {@link #normalize(int, String, int)} — which always yields 1-100 —
     * so every normalized request participates in Auto-TPM. Kept as a
     * defensive guard and for unit-test assertions.
     */
    public static final int NO_PRIORITY = 0;

    private PriorityNormalizer() {
    }

    /**
     * Returns true iff the value denotes an explicit priority (i.e. is not
     * the {@link #NO_PRIORITY} sentinel). Always true for values returned
     * by {@link #normalize(int, String, int)}.
     *
     * <p><b>This is the single home of the "has a priority" concept.</b>
     * A 0 priority may still appear in worker-reported task state because the
     * engine protocol does not require that field. Such entries cannot be used
     * as priority-preemption victims.
     * Downstream code MUST call this method instead of hand-rolling
     * {@code priority > 0} / {@code priority == 0} checks.
     */
    public static boolean hasPriority(int normalizedPriority) {
        return normalizedPriority != NO_PRIORITY;
    }

    /**
     * Returns true iff the value is a valid priority level.
     * Valid range is 1-100; 0 is the NO_PRIORITY sentinel, negatives and
     * values above 100 are invalid.
     */
    public static boolean isValid(int priority) {
        return priority > 0 && priority <= 100;
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
        boolean headerPresent = headerValue != null && !headerValue.isBlank();
        if (protoPriority == 0 && !headerPresent) {
            // Not carried at all: use default priority instead of opting out.
            return isValid(defaultPriority) ? defaultPriority : DEFAULT_PRIORITY;
        }
        if (isValid(protoPriority)) {
            return protoPriority;
        }
        if (headerPresent) {
            try {
                int headerPriority = Integer.parseInt(headerValue.trim());
                if (isValid(headerPriority)) {
                    return headerPriority;
                }
            } catch (NumberFormatException ignored) {
            }
        }
        return isValid(defaultPriority) ? defaultPriority : DEFAULT_PRIORITY;
    }
}
