package org.flexlb.mock.admission;

import java.util.Objects;
import org.flexlb.mock.api.model.Types.Role;
import org.flexlb.mock.compute.ComputeModel;

/** Values needed to bind the native protocol and EngineLoop; contains no queue policy settings. */
public record AdmissionConfig(String id, Role role, int blockSize, int tpSize, int maxSeqLen,
                                int maxBatchTokens, ComputeModel compute) {
    public AdmissionConfig {
        if (id == null || id.isBlank() || (role != Role.prefill && role != Role.decode)
                || blockSize < 1 || tpSize < 1 || maxSeqLen < 1 || maxBatchTokens < 0)
            throw new IllegalArgumentException("Invalid RTP endpoint configuration");
        Objects.requireNonNull(compute);
    }
}
