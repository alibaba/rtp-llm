package org.flexlb.service.optimizer;

import org.flexlb.dao.master.WorkerHost;
import org.flexlb.discovery.ServiceDiscovery;
import org.flexlb.discovery.ServiceHostListener;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ServiceDiscoveryAddressResolverTest {

    private static final String DOMAIN = "optimizer.test.domain.com";

    @Test
    void should_return_empty_before_start() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);

        assertTrue(r.getAddresses().isEmpty());
    }

    @Test
    void should_seed_initial_hosts_and_register_listener_on_start() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(List.of(WorkerHost.of("1.1.1.1", 8000, "site-a")));

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.start();

        assertEquals(List.of("1.1.1.1:8000"), r.getAddresses());
        verify(sd, times(1)).getHosts(DOMAIN);
        verify(sd, times(1)).listen(eq(DOMAIN), any(ServiceHostListener.class));
    }

    @Test
    void should_update_addresses_on_listener_callback() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(Collections.emptyList());

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.start();

        ArgumentCaptor<ServiceHostListener> captor = ArgumentCaptor.forClass(ServiceHostListener.class);
        verify(sd).listen(eq(DOMAIN), captor.capture());
        ServiceHostListener listener = captor.getValue();

        listener.onHostsChanged(List.of(
                WorkerHost.of("2.2.2.2", 9000),
                WorkerHost.of("3.3.3.3", 9001)
        ));

        assertEquals(List.of("2.2.2.2:9000", "3.3.3.3:9001"), r.getAddresses());
    }

    @Test
    void should_clear_addresses_when_listener_pushes_empty_or_null() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(List.of(WorkerHost.of("1.1.1.1", 8000)));

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.start();
        assertEquals(List.of("1.1.1.1:8000"), r.getAddresses());

        ArgumentCaptor<ServiceHostListener> captor = ArgumentCaptor.forClass(ServiceHostListener.class);
        verify(sd).listen(eq(DOMAIN), captor.capture());

        // Empty hosts clear the snapshot (aligned with EngineAddressNameResolver)
        captor.getValue().onHostsChanged(Collections.emptyList());
        assertTrue(r.getAddresses().isEmpty());

        // null also clears
        captor.getValue().onHostsChanged(List.of(WorkerHost.of("2.2.2.2", 9000)));
        assertEquals(List.of("2.2.2.2:9000"), r.getAddresses());
        captor.getValue().onHostsChanged(null);
        assertTrue(r.getAddresses().isEmpty());
    }

    @Test
    void should_skip_duplicate_start() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(Collections.emptyList());

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.start();
        r.start();

        verify(sd, times(1)).getHosts(DOMAIN);
        verify(sd, times(1)).listen(eq(DOMAIN), any(ServiceHostListener.class));
    }

    @Test
    void shutdown_should_not_propagate_to_shared_service_discovery() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(Collections.emptyList());

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.start();
        r.shutdown();

        // ServiceDiscovery is a shared Spring bean; shutdown() must not propagate
        verify(sd, never()).shutdown();
    }

    @Test
    void should_tolerate_getHosts_failure_and_still_register_listener() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenThrow(new RuntimeException("boom"));

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.start();

        assertTrue(r.getAddresses().isEmpty());
        verify(sd, times(1)).listen(eq(DOMAIN), any(ServiceHostListener.class));
    }

    @Test
    void should_ignore_listener_callbacks_after_shutdown() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(List.of(WorkerHost.of("1.1.1.1", 8000)));

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.start();
        assertEquals(List.of("1.1.1.1:8000"), r.getAddresses());

        ArgumentCaptor<ServiceHostListener> captor = ArgumentCaptor.forClass(ServiceHostListener.class);
        verify(sd).listen(eq(DOMAIN), captor.capture());

        r.shutdown();
        // After shutdown, late listener callbacks must not mutate state
        captor.getValue().onHostsChanged(List.of(WorkerHost.of("9.9.9.9", 8888)));
        assertEquals(List.of("1.1.1.1:8000"), r.getAddresses());
    }

    @Test
    void should_rollback_started_and_allow_retry_when_listen_fails() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(Collections.emptyList());
        // First listen() throws -> started should roll back so a retry is possible
        doThrow(new RuntimeException("listen boom"))
                .when(sd).listen(eq(DOMAIN), any(ServiceHostListener.class));

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        // start() returns false on listen failure
        assertFalse(r.start());
        verify(sd, times(1)).listen(eq(DOMAIN), any(ServiceHostListener.class));

        // After fixing listen, start() again should re-invoke getHosts and listen
        reset(sd);
        when(sd.getHosts(DOMAIN)).thenReturn(List.of(WorkerHost.of("4.4.4.4", 7000)));
        // start() returns true after recovery
        assertTrue(r.start());

        verify(sd, times(1)).getHosts(DOMAIN);
        verify(sd, times(1)).listen(eq(DOMAIN), any(ServiceHostListener.class));
        assertEquals(List.of("4.4.4.4:7000"), r.getAddresses());
    }

    @Test
    void should_not_start_after_shutdown() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.shutdown();
        r.start();

        // After shutdown, no service-discovery calls should be issued
        verify(sd, never()).getHosts(any());
        verify(sd, never()).listen(any(), any());
    }

    // ===== start() return-value contract =====

    @Test
    void start_should_return_true_on_success() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(Collections.emptyList());

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        assertTrue(r.start());
    }

    @Test
    void start_should_return_true_when_already_started() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(Collections.emptyList());

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        assertTrue(r.start());
        // Idempotent: subsequent calls also return true without re-issuing IO
        assertTrue(r.start());
        verify(sd, times(1)).getHosts(DOMAIN);
        verify(sd, times(1)).listen(eq(DOMAIN), any(ServiceHostListener.class));
    }

    @Test
    void start_should_return_false_when_listen_fails() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);
        when(sd.getHosts(DOMAIN)).thenReturn(Collections.emptyList());
        doThrow(new RuntimeException("listen boom"))
                .when(sd).listen(eq(DOMAIN), any(ServiceHostListener.class));

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        assertFalse(r.start());
    }

    @Test
    void start_should_return_false_after_shutdown() {
        ServiceDiscovery sd = mock(ServiceDiscovery.class);

        ServiceDiscoveryAddressResolver r = new ServiceDiscoveryAddressResolver(sd, DOMAIN);
        r.shutdown();
        assertFalse(r.start());
    }
}
