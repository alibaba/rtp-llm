package org.flexlb.mock.api.model;

import org.flexlb.mock.api.Argument;

/** Mock progress parameters, shared across engine adapters. Not a native MTP algorithm selector. */
public record DecodeSpec(@Argument(optional=true, minimum=0, maximum=2147483646) Integer draft_tokens,
                         @Argument(optional=true, minimum=0, maximum=1) Double acceptance_rate,
                         @Argument(optional=true) Long seed,
                         @Argument(optional=true, minimum=1, maximum=2147483647) Double tokens_per_step) {
    public DecodeSpec {
        boolean speculative = draft_tokens != null && draft_tokens != 0
            || acceptance_rate != null && acceptance_rate != 0 || seed != null && seed != 0;
        if (tokens_per_step != null && (!Double.isFinite(tokens_per_step) || tokens_per_step < 1
                || tokens_per_step > Integer.MAX_VALUE || speculative))
            throw new IllegalArgumentException("tokens_per_step requires a finite mean >= 1 and cannot combine with speculative settings");
        draft_tokens = draft_tokens == null ? 0 : draft_tokens;
        if (draft_tokens < 0 || draft_tokens == Integer.MAX_VALUE) throw new IllegalArgumentException("Invalid draft_tokens");
        if (acceptance_rate == null && draft_tokens > 0) throw new IllegalArgumentException("draft_tokens requires acceptance_rate");
        acceptance_rate = acceptance_rate == null ? 0 : acceptance_rate;
        if (!Double.isFinite(acceptance_rate) || acceptance_rate < 0 || acceptance_rate > 1)
            throw new IllegalArgumentException("acceptance_rate must be within [0,1]");
        if (draft_tokens > 0 && acceptance_rate > 0 && acceptance_rate < 1 && seed == null)
            throw new IllegalArgumentException("Random Decode progress requires seed");
        seed = seed == null ? 0 : seed;
    }
}
