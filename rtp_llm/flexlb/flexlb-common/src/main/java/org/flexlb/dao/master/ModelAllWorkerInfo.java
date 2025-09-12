package org.flexlb.dao.master;

import lombok.Data;

/**
 * 模型下所有Workers汇总信息
 */
@Data
public class ModelAllWorkerInfo {

    private String modelName;
    private int engineCount;
    private int maxConcurrency;
    private int currentConcurrency;
    private int availableConcurrency;
    private String errorMsg;

    public void addMaxConcurrency(int maxConcurrency) {
        this.maxConcurrency += maxConcurrency;
    }

    public void addCurrentConcurrency(int currentConcurrency) {
        this.currentConcurrency += currentConcurrency;
    }

    public void addAvailableConcurrency(int availableConcurrency) {
        this.availableConcurrency += availableConcurrency;
    }

    public static ModelAllWorkerInfo from(String modelName, int engineCount, int maxConcurrency, int currentConcurrency, int availableConcurrency) {
        ModelAllWorkerInfo modelAllWorkerInfo = new ModelAllWorkerInfo();
        modelAllWorkerInfo.setModelName(modelName);
        modelAllWorkerInfo.setEngineCount(engineCount);
        modelAllWorkerInfo.setMaxConcurrency(maxConcurrency);
        modelAllWorkerInfo.setCurrentConcurrency(currentConcurrency);
        modelAllWorkerInfo.setAvailableConcurrency(availableConcurrency);

        return modelAllWorkerInfo;
    }

}
