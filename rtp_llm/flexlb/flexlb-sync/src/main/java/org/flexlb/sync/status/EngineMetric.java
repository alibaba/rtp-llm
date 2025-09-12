package org.flexlb.sync.status;

import lombok.Data;

@Data
public class EngineMetric {
    private int prefill;
    private int decode;
    private int pdFusion;
    private int vit;
    private int total;

    public EngineMetric(int prefill, int decode, int pdFusion, int vit) {
        this.prefill = prefill;
        this.decode = decode;
        this.pdFusion = pdFusion;
        this.vit = vit;
        this.total = prefill + decode + pdFusion + vit;
    }
}