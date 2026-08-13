package org.flexlb.cache.hash;

import org.flexlb.cache.domain.LocalStandbyHashResult;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;

class LocalStandbyHashServiceTest {

    @Test
    void calculatesAndPublishesStandbyHashAsynchronously() throws Exception {
        LocalStandbyHashService hashService =
                new LocalStandbyHashService(
                        new CacheMatchConfiguration(modelMetaConfig()),
                        mock(FlexMonitor.class),
                        new VllmBlockHashStrategy());
        Request request = new Request();
        request.setRequestId("request-1");
        request.setLocalStandbyBlockSize(4);

        try {
            LocalStandbyHashResult result =
                    hashService.submit(request, new int[]{1, 2, 3, 4}, 4, 0)
                            .get(5, TimeUnit.SECONDS);

            assertEquals(List.of(2164874634404590027L), result.blockCacheKeys());
            assertEquals(4, result.blockSize());
            assertSame(result.blockCacheKeys(), request.getLocalStandbyBlockCacheKeys());
            assertEquals(4, request.getLocalStandbyBlockSize());
        } finally {
            hashService.shutdown();
        }
    }

    @Test
    void usesConfiguredStrategyForStandbyHash() throws Exception {
        LocalStandbyHashService hashService =
                new LocalStandbyHashService(
                        new CacheMatchConfiguration(modelMetaConfig()),
                        mock(FlexMonitor.class),
                        new SglangBlockHashStrategy());
        Request request = new Request();
        request.setRequestId("request-sglang");

        try {
            LocalStandbyHashResult result =
                    hashService.submit(request, new int[]{1, 2, 3, 4, 5}, 4, 0)
                            .get(5, TimeUnit.SECONDS);

            assertEquals(
                    List.of(-3488128144981237669L),
                    result.blockCacheKeys());
            assertEquals(
                    List.of(-3488128144981237669L),
                    request.getLocalStandbyCacheableBlockCacheKeys());
        } finally {
            hashService.shutdown();
        }
    }

    @Test
    void publishesSglangEagleBigramHashesAndOnlyFullPages() throws Exception {
        LocalStandbyHashService hashService =
                new LocalStandbyHashService(
                        new CacheMatchConfiguration(modelMetaConfig()),
                        mock(FlexMonitor.class),
                        new SglangBlockHashStrategy());
        Request request = new Request();
        request.setRequestId("request-sglang-eagle");

        try {
            LocalStandbyHashResult result =
                    hashService.submit(request, new int[]{1, 2, 3, 4, 5, 6}, 4, 1)
                            .get(5, TimeUnit.SECONDS);

            assertEquals(
                    List.of(-638950109823820341L),
                    result.blockCacheKeys());
            assertEquals(
                    List.of(-638950109823820341L),
                    request.getLocalStandbyCacheableBlockCacheKeys());
        } finally {
            hashService.shutdown();
        }
    }

    private ModelMetaConfig modelMetaConfig() {
        LocalStandbyConfig standby = new LocalStandbyConfig();
        standby.setHashThreadCount(1);
        standby.setHashQueueCapacity(4);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
        kvcm.setLocalStandby(standby);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);

        ModelMetaConfig config = new ModelMetaConfig();
        config.putServiceRoute(route.getServiceId(), route);
        return config;
    }
}
