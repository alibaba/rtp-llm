package org.flexlb.mock.engine.forward;

/** Mean output tokens per Decode round. */
public record DecodeSpec(Double tokens_per_step) {
    public DecodeSpec {
        tokens_per_step = tokens_per_step == null ? 1.0 : tokens_per_step;
        if (!Double.isFinite(tokens_per_step) || tokens_per_step < 1 || tokens_per_step > Integer.MAX_VALUE)
            throw new IllegalArgumentException("tokens_per_step requires a finite mean between 1 and Integer.MAX_VALUE");
    }
}
