package org.flexlb.dispatcher;

import java.util.Locale;

/** size:N caps chunk size; count:N balances the input across at most N nonempty chunks. */
public record SubBatchSpec(Mode mode, int value) {
    public enum Mode { SIZE, COUNT }

    public SubBatchSpec {
        if (mode == null || value < 1) {
            throw new IllegalArgumentException("subBatch requires a mode and a positive integer");
        }
    }

    public static SubBatchSpec parse(String raw) {
        if (raw == null) {
            throw new IllegalArgumentException("subBatch must not be null");
        }
        String[] parts = raw.trim().split(":", 2);
        try {
            Mode mode = parts.length == 1 ? Mode.SIZE : Mode.valueOf(parts[0].trim().toUpperCase(Locale.ROOT));
            return new SubBatchSpec(mode, Integer.parseInt(parts[parts.length - 1].trim()));
        } catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("subBatch must be size:N, count:N or N with N > 0: " + raw, error);
        }
    }
}
