package org.flexlb.mock.engine.forward;

/**
 * Advances a request by a deterministic number of output tokens each round.
 */
record DecodeStepModel(Config configuration) {
    DecodeStepModel { java.util.Objects.requireNonNull(configuration); }

    public int tokens(long round) {
        if (round < 0) throw new IllegalArgumentException("Invalid Decode round");
        double mean = configuration.tokens_per_step();
        long count = Math.round(Math.ceil((round + 1.0) * mean) - Math.ceil(round * mean));
        if (count < 1 || count > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Decode progress exceeds numeric precision");
        return (int) count;
    }

    /**
     * Validates the mean output tokens per Decode round.
     */
    record Config(Double tokens_per_step) {
        static final Config DEFAULT = new Config(null);

        Config {
            tokens_per_step = tokens_per_step == null ? 1.0 : tokens_per_step;
            if (!Double.isFinite(tokens_per_step) || tokens_per_step < 1 || tokens_per_step > Integer.MAX_VALUE)
                throw new IllegalArgumentException("tokens_per_step requires a finite mean between 1 and Integer.MAX_VALUE");
        }
    }
}
