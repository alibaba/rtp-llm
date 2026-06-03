package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PolynomialPredictorTest {

    @Test
    void singleRequest() {
        PolynomialPredictor p = new PolynomialPredictor(100, 0.01, 1e-7, 0, 0, 50);

        long ms = p.estimateMs(8000, 0);

        // T = 100 + 0.01*8000 + 1e-7*8000^2 + 0 + 0 + 50*1
        //   = 100 + 80 + 6.4 + 50 = 236
        assertEquals(236, ms);
    }

    @Test
    void batchRequest() {
        PolynomialPredictor p = new PolynomialPredictor(100, 0.01, 1e-7, 0, 0, 50);

        List<RequestProfile> batch = List.of(
                new RequestProfile(4000, 0),
                new RequestProfile(4000, 0));

        long ms = p.predictBatchMs(batch);

        // sumC = 8000, sumQuadratic = 1e-7*(4000^2 + 4000^2) = 3.2, bs = 2
        // T = 100 + 0.01*8000 + 3.2 + 0 + 50*2 = 283
        assertEquals(283, ms);
    }

    @Test
    void cacheHitReducesCompute() {
        PolynomialPredictor p = new PolynomialPredictor(100, 0.01, 1e-7, 0, 0.005, 50);

        long noHit = p.estimateMs(8000, 0);
        long withHit = p.estimateMs(8000, 4000);

        // noHit:  c=8000, p=0   → 100 + 80 + 6.4 + 0 + 50 = 236
        // withHit: c=4000, p=4000 → 100 + 40 + 1.6 + 20 + 50 = 211
        assertEquals(236, noHit);
        assertEquals(211, withHit);
    }

    @Test
    void emptyBatchReturnsZero() {
        PolynomialPredictor p = new PolynomialPredictor(100, 1, 0, 0, 0, 0);
        assertEquals(0, p.predictBatchMs(List.of()));
    }

    @Test
    void crossTermA3() {
        PolynomialPredictor p = new PolynomialPredictor(0, 0, 0, 0.001, 0, 0);

        long ms = p.estimateMs(10000, 2000);

        // c=8000, p=2000, a3*c*p = 0.001*8000*2000 = 16000
        assertEquals(16000, ms);
    }
}
