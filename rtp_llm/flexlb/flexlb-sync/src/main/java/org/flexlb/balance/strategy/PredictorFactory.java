package org.flexlb.balance.strategy;

import org.flexlb.config.FlexlbConfig;

public class PredictorFactory {

    public static PrefillTimePredictor create(FlexlbConfig config) {
        PolynomialPredictor poly = new PolynomialPredictor(
                config.getCostAlpha0(),
                config.getCostAlpha1(),
                config.getCostAlpha2(),
                config.getCostAlpha3(),
                config.getCostAlpha4(),
                config.getCostAlpha5());

        String type = config.getPrefillPredictorType();
        String gridJson = config.getPrefillGridTable();

        if ("grid".equals(type) && gridJson != null && !gridJson.isBlank()) {
            return GridPredictor.fromJson(gridJson, null);
        }
        if ("composite".equals(type) && gridJson != null && !gridJson.isBlank()) {
            return GridPredictor.fromJson(gridJson, poly);
        }
        return poly;
    }
}
