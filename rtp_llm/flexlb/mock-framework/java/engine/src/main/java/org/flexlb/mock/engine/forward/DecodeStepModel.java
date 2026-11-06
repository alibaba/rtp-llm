package org.flexlb.mock.engine.forward;

/** Advance a request by a deterministic number of output tokens each round. */
public record DecodeStepModel(DecodeSpec configuration) {
    public DecodeStepModel { java.util.Objects.requireNonNull(configuration); }

    public int tokens(long round) {
        if (round < 0) throw new IllegalArgumentException("Invalid Decode round");
        double mean = configuration.tokens_per_step();
        long count = Math.round(Math.ceil((round + 1.0) * mean) - Math.ceil(round * mean));
        if (count < 1 || count > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Decode progress exceeds numeric precision");
        return (int) count;
    }
}
