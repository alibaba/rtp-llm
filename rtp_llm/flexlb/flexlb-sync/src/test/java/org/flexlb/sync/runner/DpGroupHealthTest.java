package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class DpGroupHealthTest {

    @ParameterizedTest
    @CsvSource({"PREFILL,2,1", "PREFILL,2,2", "DECODE,4,1", "DECODE,4,2"})
    void aliveFalseRetiresEveryRankAndRecoveryNeedsAllRanks(RoleType role, int size, long version) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        WorkerDirectory directory = new WorkerDirectory(RunnerTestSupport.endpointRegistry(config));
        List<WorkerStatus> ranks = publishGroup(directory, role, size, "dp-a", 9000);
        WorkerStatus other = publishGroup(directory, role, 1, "dp-b", 9100).getFirst();
        assertEquals(size + 1, candidates(directory, role).size());

        poll(directory, ranks.getLast(),
                CompletableFuture.completedFuture(response(ranks.getLast(), "dp-a", size - 1, size, false, version)));

        assertEquals(List.of(other.getLogicalIpPort()), candidates(directory, role));
        for (WorkerStatus rank : ranks) {
            assertFalse(rank.isActiveGeneration());
            assertNull(directory.exactEndpoint(role, rank.getLogicalIpPort(), rank));
            assertNull(directory.statusSnapshot(role).get(rank.getLogicalIpPort()));
            assertNull(rank.tryBeginStatusPoll());
        }
        assertTrue(other.isActiveGeneration());

        for (int rank = 0; rank < size; rank++) {
            WorkerStatus recovered = discover(directory, role, 9000 + rank);
            poll(directory, recovered,
                    CompletableFuture.completedFuture(response(recovered, "dp-a", rank, size, true, 1)));
            assertEquals(rank == size - 1 ? size + 1 : 1, candidates(directory, role).size());
        }
    }

    @ParameterizedTest
    @CsvSource({"PREFILL,2", "DECODE,4"})
    void thirdConsecutiveRpcFailureRetiresWholeGroupAndSuccessResetsCounter(RoleType role, int size) {
        ConfigService config = mock(ConfigService.class);
        when(config.loadBalanceConfig()).thenReturn(SchedulingTestConfig.newConfig());
        WorkerDirectory directory = new WorkerDirectory(RunnerTestSupport.endpointRegistry(config));
        List<WorkerStatus> ranks = publishGroup(directory, role, size, "dp-a", 9000);
        WorkerStatus other = publishGroup(directory, role, 1, "dp-b", 9100).getFirst();
        WorkerStatus failed = ranks.getLast();
        for (int attempt = 1; attempt <= 2; attempt++) {
            poll(directory, failed, CompletableFuture.failedFuture(io.grpc.Status.UNAVAILABLE.asRuntimeException()));
            assertEquals(size + 1, candidates(directory, role).size());
        }
        poll(directory, failed, CompletableFuture.completedFuture(response(failed, "dp-a", size - 1, size, true, 1)));
        assertEquals(0, failed.pollHealth().consecutiveTransportFailures());
        for (int attempt = 1; attempt <= 3; attempt++) {
            poll(directory, failed, CompletableFuture.failedFuture(io.grpc.Status.UNAVAILABLE.asRuntimeException()));
            assertEquals(attempt < 3 ? size + 1 : 1, candidates(directory, role).size());
        }
        assertEquals(List.of(other.getLogicalIpPort()), candidates(directory, role));
        for (WorkerStatus rank : ranks) {
            assertFalse(rank.isActiveGeneration());
            assertNull(directory.statusSnapshot(role).get(rank.getLogicalIpPort()));
        }
    }

    private List<WorkerStatus> publishGroup(WorkerDirectory directory, RoleType role, int size, String master, int port) {
        List<WorkerStatus> members = new ArrayList<>();
        for (int rank = 0; rank < size; rank++) {
            WorkerStatus status = discover(directory, role, port + rank);
            poll(directory, status, CompletableFuture.completedFuture(response(status, master, rank, size, true, 1)));
            members.add(status);
        }
        return members;
    }

    private WorkerStatus discover(WorkerDirectory directory, RoleType role, int port) {
        WorkerStatus status = RunnerTestSupport.discovered(role, "default", "127.0.0.1", port, port, "test");
        return directory.currentOrDiscover(role, status.getLogicalIpPort(), () -> status);
    }

    private EngineRpcService.WorkerStatusPB response(WorkerStatus status,
                                                     String master,
                                                     int rank,
                                                     int size,
                                                     boolean alive,
                                                     long version) {
        return EngineRpcService.WorkerStatusPB.newBuilder()
                .setRole(status.getRole().getCode())
                .setRoleType(status.getRole() == RoleType.DECODE
                        ? EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE
                        : EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .setStatusVersion(version)
                .setAlive(alive)
                .setDpMasterAddress(master)
                .setDpRank(rank)
                .setDpSize(size)
                .build();
    }

    private void poll(WorkerDirectory directory,
                      WorkerStatus status,
                      CompletableFuture<EngineRpcService.WorkerStatusPB> response) {
        EngineGrpcService grpc = mock(EngineGrpcService.class);
        when(grpc.getWorkerStatusAsync(anyString(), anyInt(), anyLong(), anyLong(), any())).thenReturn(response);
        new GrpcWorkerStatusRunner("test-model", status.getLogicalIpPort(), "test", status.getRole(), "default", status,
                status.tryBeginStatusPoll(), directory, mock(EngineHealthReporter.class), grpc, 5000L,
                mock(CacheAwareService.class), Runnable::run).run();
    }

    private List<String> candidates(WorkerDirectory directory, RoleType role) {
        if (role == RoleType.DECODE) {
            return directory.decodeRoutingSnapshot("default").stream().map(view -> view.address()).toList();
        }
        return directory.prefillRoutingSnapshot(role)
                .stream()
                .map(EndpointRegistry.PrefillRoutingEntry::address)
                .toList();
    }
}
