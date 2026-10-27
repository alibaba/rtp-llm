package org.flexlb.mock.api.model;

import org.flexlb.mock.api.Argument;
import org.flexlb.mock.api.Checks;
import org.flexlb.mock.api.Variant;
import static org.flexlb.mock.api.Checks.Rule.*;

/** Playback overrides; no sampler or dataset synthesis is implemented here. */
public sealed interface OutputTokens permits OutputTokens.Fixed, OutputTokens.Uniform,
        OutputTokens.Normal {
    @Variant(scalar=true)
    record Fixed(@Argument(minimum=0) int value) implements OutputTokens {}

    @Variant("uniform")
    @Checks(@Checks.Check(rule=ordered, fields={"min", "max"}))
    record Uniform(@Argument(minimum=0) int min, @Argument(minimum=0) int max,
                   long seed) implements OutputTokens {}

    @Variant("normal")
    @Checks(@Checks.Check(rule=ordered, fields={"min", "max"}))
    record Normal(@Argument(minimum=0) double mean, @Argument(minimum=0) double stddev,
                  @Argument(minimum=0) int min, @Argument(minimum=0) int max,
                  long seed) implements OutputTokens {}
}
