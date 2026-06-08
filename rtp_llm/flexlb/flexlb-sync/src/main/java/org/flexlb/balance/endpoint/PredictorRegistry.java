package org.flexlb.balance.endpoint;

import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.springframework.stereotype.Component;

import java.util.concurrent.ConcurrentHashMap;

@Component
public class PredictorRegistry {

    private final ConcurrentHashMap<String, PrefillTimePredictor> predictors = new ConcurrentHashMap<>();

    public PrefillTimePredictor getDefault(FlexlbConfig config) {
        return predictors.computeIfAbsent("default", k -> new PrefillTimePredictor(
                config.getCostAlpha0(), config.getCostAlpha1(), config.getCostAlpha2(),
                config.getCostAlpha3(), config.getCostAlpha4(), config.getCostAlpha5()));
    }
}
