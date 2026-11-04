package org.flexlb.mock.api.model;

import org.flexlb.mock.api.Argument;

public record StoppingSpec(@Argument(optional=true, minimum=0, maximum=1) Double probability,
                           @Argument(optional=true) Long seed) {
    public StoppingSpec {
        probability = probability == null ? 0 : probability;
        if (!Double.isFinite(probability) || probability < 0 || probability > 1)
            throw new IllegalArgumentException("probability must be within [0,1]");
        if (probability > 0 && probability < 1 && seed == null)
            throw new IllegalArgumentException("Random EOS stopping requires seed");
        seed = seed == null ? 0 : seed;
    }
}
