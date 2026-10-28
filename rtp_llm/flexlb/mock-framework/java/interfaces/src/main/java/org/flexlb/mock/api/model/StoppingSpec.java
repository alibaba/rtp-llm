package org.flexlb.mock.api.model;

import org.flexlb.mock.api.Argument;

public record StoppingSpec(@Argument(optional=true, minimum=0, maximum=1) Double eos_probability,
                           @Argument(optional=true) Long seed) {
    public StoppingSpec {
        eos_probability = eos_probability == null ? 0 : eos_probability;
        if (!Double.isFinite(eos_probability) || eos_probability < 0 || eos_probability > 1)
            throw new IllegalArgumentException("eos_probability must be within [0,1]");
        if (eos_probability > 0 && eos_probability < 1 && seed == null)
            throw new IllegalArgumentException("Random EOS stopping requires seed");
        seed = seed == null ? 0 : seed;
    }
}
