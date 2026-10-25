package org.flexlb.balance.scheduler.decode;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.dao.BalanceContext;

/**
 * Conditions a Decode worker must satisfy for one request, frozen before selection.
 * inputKvTokens is max(0, seqLen). inputAndMaxOutputKvTokens is
 * inputKvTokens + max(0, maxNewTokens), saturated at Long.MAX_VALUE.
 * Neither value is an estimate of the actual generated output length.
 */
public record DecodeRequirement(String requestId,
                                int priority,
                                long inputKvTokens,
                                long inputAndMaxOutputKvTokens,
                                DecodeEndpoint.CapacityLimit capacity,
                                DecodeMode mode,
                                DecodeCostFormula costFormula) {

    public static DecodeRequirement capture(BalanceContext context) {
        var request = context.getRequest();
        var config = context.getConfig();
        var decode = config.getRouter().getRoles().getDecode();
        var availability = decode.getAvailability();
        long promptTokens = Math.max(0L, request.getSeqLen());
        long maxOutputTokens = Math.max(0L, request.getMaxNewTokens());
        long inputAndMaxOutputTokens =
                promptTokens > Long.MAX_VALUE - maxOutputTokens ? Long.MAX_VALUE : promptTokens + maxOutputTokens;
        return new DecodeRequirement(request.getRequestId(), context.getPriority(), promptTokens,
                inputAndMaxOutputTokens,
                new DecodeEndpoint.CapacityLimit(
                        availability.getMaxEngineRequests() == null ? 0L : availability.getMaxEngineRequests(),
                        availability.getMaxKvUsagePercent()),
                DecodeMode.from(config), decode.getCostEstimator().compiledFormula());
    }
}
