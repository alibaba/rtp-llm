package org.flexlb.cache.domain;

/** Exact identity of one cache-owning Engine generation. */
public record EngineGeneration(String address, long generationId) {

    public EngineGeneration {
        if (address == null || address.isBlank()) {
            throw new IllegalArgumentException("Engine address must not be blank");
        }
        if (generationId <= 0L) {
            throw new IllegalArgumentException(
                    "Engine generation must be positive: " + generationId);
        }
    }
}
