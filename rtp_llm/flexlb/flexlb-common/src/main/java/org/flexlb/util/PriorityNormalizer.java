package org.flexlb.util;

/**
 * Normalizes the Auto-TPM QoS priority carried by a request.
 *
 * <p>Valid priority levels are {30, 40, 50, 60, 70} (higher = more important).
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

    /** Sentinel for "request carried no priority"; such requests bypass Auto-TPM. */
    public static final int NO_PRIORITY = 0;

    private static final int[] VALID_PRIORITIES = {30, 40, 50, 60, 70};

    private PriorityNormalizer() {
    }

    /** Returns true iff the normalized value denotes an explicit priority. */
    public static boolean hasPriority(int normalizedPriority) {
        return normalizedPriority != NO_PRIORITY;
    }

    /** Returns true iff the value is one of the defined priority levels. */
    public static boolean isValid(int priority) {
        for (int valid : VALID_PRIORITIES) {
            if (priority == valid) {
                return true;
            }
        }
        return false;
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
     *         otherwise one of {30, 40, 50, 60, 70}
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
        if (protoPriority != 0) {
            Logger.warn("Invalid proto priority {}, falling back to header/default", protoPriority);
        }
        if (headerPresent) {
            try {
                int headerPriority = Integer.parseInt(headerValue.trim());
                if (isValid(headerPriority)) {
                    return headerPriority;
                }
                Logger.warn("Invalid {} header value '{}', falling back to default",
                        QOS_HEADER_NAME, headerValue);
            } catch (NumberFormatException e) {
                Logger.warn("Non-numeric {} header value '{}', falling back to default",
                        QOS_HEADER_NAME, headerValue);
            }
        }
        return isValid(defaultPriority) ? defaultPriority : DEFAULT_PRIORITY;
    }
}
