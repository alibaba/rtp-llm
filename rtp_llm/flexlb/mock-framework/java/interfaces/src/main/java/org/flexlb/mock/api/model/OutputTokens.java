package org.flexlb.mock.api.model;

import java.util.List;
import org.flexlb.mock.api.Argument;
import org.flexlb.mock.api.Checks;
import org.flexlb.mock.api.Variant;
import static org.flexlb.mock.api.Checks.Rule.*;

/** Playback overrides; no sampler or dataset synthesis is implemented here. */
public sealed interface OutputTokens permits OutputTokens.Fixed, OutputTokens.Uniform,
        OutputTokens.Normal, OutputTokens.Lognormal, OutputTokens.Choice {
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

    @Variant("lognormal")
    @Checks(@Checks.Check(rule=ordered, fields={"min", "max"}))
    record Lognormal(double mu, @Argument(minimum=0) double sigma,
                     @Argument(minimum=0) int min, @Argument(minimum=0) int max,
                     long seed) implements OutputTokens {}

    @Variant("choice")
    @Checks({@Checks.Check(rule=same_length, fields={"values", "weights"}),
             @Checks.Check(rule=positive_sum, fields={"weights"})})
    record Choice(@Argument(minItems=1, itemMinimum=0) List<Integer> values,
                  @Argument(optional=true, minItems=1, itemMinimum=0) List<Double> weights,
                  long seed) implements OutputTokens {}
}
