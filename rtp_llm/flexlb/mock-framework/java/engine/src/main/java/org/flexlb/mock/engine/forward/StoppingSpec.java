package org.flexlb.mock.engine.forward;

public record StoppingSpec(Double probability,
                           Long seed) {
    public StoppingSpec {
        probability = probability == null ? 0 : probability;
        if (!Double.isFinite(probability) || probability < 0 || probability > 1)
            throw new IllegalArgumentException("probability must be within [0,1]");
        if (probability > 0 && probability < 1 && seed == null)
            throw new IllegalArgumentException("Random EOS stopping requires seed");
        seed = seed == null ? 0 : seed;
    }
}
