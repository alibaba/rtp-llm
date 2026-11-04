package org.flexlb.mock.engine.service;

import java.util.Objects;
import org.flexlb.mock.protocol.Role;
import org.flexlb.mock.engine.forward.ComputeModel;

/** Values needed to bind the native protocol and RtpEngineLoop; contains no queue policy settings. */
public record RtpServiceConfig(String id, Role role, int blockSize, int tpSize, int dpSize, int maxSeqLen,
                                int maxBatchTokens, ComputeModel compute, int imageToken, long cacheKeySeed) {
    public RtpServiceConfig {
        if (id == null || id.isBlank() || (role != Role.prefill && role != Role.decode)
                || blockSize < 1 || tpSize < 1 || dpSize < 1 || maxSeqLen < 1 || maxBatchTokens < 0)
            throw new IllegalArgumentException("Invalid RTP endpoint configuration");
        Objects.requireNonNull(compute);
    }
}
