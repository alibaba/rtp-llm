package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Delivery of preemption instructions for Engine-owned Decode requests. */
@Getter
@Setter
public final class EngineCancellationConfig {

    public enum Mode {
        RPC,
        RETURN
    }

    private Mode mode = Mode.RPC;
}
