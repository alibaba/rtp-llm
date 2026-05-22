package org.flexlb.dispatcher;

import java.util.Locale;

/**
 * Batch splitting strategy parsed from a DSL string. Two modes:
 * <ul>
 *   <li>{@code size:N} — each chunk holds at most {@code N} items; chunk count = ceil(total / N).</li>
 *   <li>{@code count:N} — split into exactly {@code N} chunks (or fewer if total &lt; N), distributing
 *       the remainder across the leading chunks.</li>
 * </ul>
 *
 * <p>The bare-integer shorthand {@code "5"} is equivalent to {@code "size:5"} — kept narrow so the
 * DSL stays one-line per config value.
 *
 * <p>Mode keywords are case-insensitive; surrounding whitespace is trimmed. Any other shape —
 * unknown mode, missing/non-integer value, value &lt; 1, null, blank — throws
 * {@link IllegalArgumentException} so config-load fails loudly rather than silently defaulting.
 */
public record SubBatchSpec(Mode mode, int value) {

    public enum Mode { SIZE, COUNT }

    public static SubBatchSpec parse(String raw) {
        if (raw == null) {
            throw new IllegalArgumentException("subBatch spec must not be null");
        }
        String trimmed = raw.trim();
        if (trimmed.isEmpty()) {
            throw new IllegalArgumentException("subBatch spec must not be blank");
        }
        int colon = trimmed.indexOf(':');
        Mode mode;
        String valueText;
        if (colon < 0) {
            mode = Mode.SIZE;
            valueText = trimmed;
        } else {
            String modeText = trimmed.substring(0, colon).trim().toUpperCase(Locale.ROOT);
            try {
                mode = Mode.valueOf(modeText);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        "subBatch mode must be 'size' or 'count', got: " + modeText);
            }
            valueText = trimmed.substring(colon + 1).trim();
            if (valueText.isEmpty()) {
                throw new IllegalArgumentException("subBatch spec missing value after ':' in: " + raw);
            }
        }
        int parsed;
        try {
            parsed = Integer.parseInt(valueText);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("subBatch value must be an integer, got: " + valueText);
        }
        if (parsed < 1) {
            throw new IllegalArgumentException("subBatch value must be >= 1, got: " + parsed);
        }
        return new SubBatchSpec(mode, parsed);
    }
}
