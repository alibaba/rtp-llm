package org.flexlb.service.grace;

import org.flexlb.config.NaviSchedPrefillSchedulerConfig;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class NaviSchedMasterRuntimeTest {
    private NaviSchedPrefillSchedulerConfig config() {
        var config = new NaviSchedPrefillSchedulerConfig();
        config.setMasterBizName("master");
        config.setStartupTimeoutMs(20);
        config.setStopTimeoutMs(20);
        return config;
    }

    @Test
    void disabledStartsNoChild() throws Exception {
        var launcher = mock(NaviSchedMasterRuntime.Launcher.class);
        var runtime = new NaviSchedMasterRuntime(null, launcher, () -> false, () -> {});
        runtime.start();
        assertTrue(runtime.isReady());
        verifyNoInteractions(launcher);
        runtime.close();
    }

    @Test
    void requiresServiceResponseBeforeReadyAndStopsChild() throws Exception {
        Process child = mock(Process.class);
        when(child.isAlive()).thenReturn(true);
        when(child.waitFor(anyLong(), any())).thenReturn(true);
        var runtime = new NaviSchedMasterRuntime(config(), () -> child, () -> true, () -> {});
        runtime.start();
        assertTrue(runtime.isReady());
        runtime.close();
        assertFalse(runtime.isReady());
        verify(child).destroy();
        verify(child).waitFor(20, TimeUnit.MILLISECONDS);
    }

    @Test
    void listeningProcessWithoutHealthyGraphFailsStartupAndIsReaped() throws Exception {
        Process child = mock(Process.class);
        when(child.isAlive()).thenReturn(true);
        when(child.waitFor(anyLong(), any())).thenReturn(true);
        var runtime = new NaviSchedMasterRuntime(config(), () -> child, () -> false, () -> {});
        assertThrows(IllegalStateException.class, runtime::start);
        assertFalse(runtime.isReady());
        verify(child).destroy();
        verify(child).waitFor(20, TimeUnit.MILLISECONDS);
    }

    @Test
    void childExitRemovesReadinessAndClosesApplication() throws Exception {
        Process child = mock(Process.class);
        when(child.isAlive()).thenReturn(true);
        when(child.waitFor(anyLong(), any())).thenReturn(true);
        Runnable onFailure = mock(Runnable.class);
        var runtime = new NaviSchedMasterRuntime(config(), () -> child, () -> true, onFailure);
        runtime.start();
        when(child.isAlive()).thenReturn(false);
        assertFalse(runtime.isReady());
        runtime.checkHealth();
        verify(onFailure).run();
        runtime.close();
    }

    @Test
    void graphFailureRemovesReadinessAndClosesApplication() throws Exception {
        Process child = mock(Process.class);
        when(child.isAlive()).thenReturn(true);
        when(child.waitFor(anyLong(), any())).thenReturn(true);
        Runnable onFailure = mock(Runnable.class);
        AtomicBoolean healthy = new AtomicBoolean(true);
        var runtime = new NaviSchedMasterRuntime(config(), () -> child, healthy::get, onFailure);
        runtime.start();
        healthy.set(false);
        runtime.checkHealth();
        assertFalse(runtime.isReady());
        verify(onFailure).run();
        runtime.close();
    }

    @Test
    void stopHasBoundedForcedReapWhenTermIsIgnored() throws Exception {
        Process child = mock(Process.class);
        when(child.isAlive()).thenReturn(true);
        when(child.waitFor(anyLong(), any())).thenReturn(false, true);
        var runtime = new NaviSchedMasterRuntime(config(), () -> child, () -> true, () -> {});
        runtime.start();
        runtime.close();
        verify(child).destroyForcibly();
        verify(child, times(2)).waitFor(20, TimeUnit.MILLISECONDS);
    }

    @Test
    void childConfigurationPreservesModeAndMasterBiz() throws Exception {
        var document =
                JsonUtils.readStrictTree(NaviSchedMasterRuntime.childConfiguration(config()));
        assertEquals(3, document.path("schemaVersion").asInt());
        var prefill = document.path("prefillScheduler");
        assertEquals("NAVI_SCHED", prefill.path("type").asText());
        assertFalse(prefill.has("masterAddress"));
        assertEquals("master", prefill.path("masterBizName").asText());
        assertEquals(20, prefill.path("startupTimeoutMs").asInt());
        assertEquals(20, prefill.path("stopTimeoutMs").asInt());
    }

    @Test
    void derivesNativePortsFromFlexlbGrpcPort() {
        var environment =
                new MockEnvironment()
                        .withProperty("server.port", "7001")
                        .withProperty("management.server.port", "7002");
        assertEquals(7006, NaviSchedMasterRuntime.masterGrpcPort(environment));
        environment.setProperty("server.port", "27001");
        environment.setProperty("management.server.port", "27002");
        assertEquals(27006, NaviSchedMasterRuntime.masterGrpcPort(environment));
    }

    @Test
    void rejectsDerivedPortOverflowAndManagementCollision() {
        var environment = new MockEnvironment().withProperty("server.port", "65531");
        assertThrows(
                IllegalArgumentException.class,
                () -> NaviSchedMasterRuntime.masterGrpcPort(environment));
        environment.setProperty("server.port", "7001");
        for (int management = 7004; management <= 7006; management++) {
            environment.setProperty("management.server.port", String.valueOf(management));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> NaviSchedMasterRuntime.masterGrpcPort(environment));
        }
    }
}
