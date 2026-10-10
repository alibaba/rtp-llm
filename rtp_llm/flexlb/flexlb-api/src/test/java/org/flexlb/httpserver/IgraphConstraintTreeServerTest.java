package org.flexlb.httpserver;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.constraint.IgraphConstraintTreePoller;
import org.junit.jupiter.api.Test;
import org.springframework.test.web.reactive.server.WebTestClient;

import static org.mockito.Mockito.*;

class IgraphConstraintTreeServerTest {
    @Test
    void exposesStatusQueuesRefreshAndRejectsFollowerOrBusy() {
        var poller = mock(IgraphConstraintTreePoller.class);
        var leader = mock(LBStatusConsistencyService.class);
        when(leader.isMaster()).thenReturn(true);
        when(poller.trigger()).thenReturn(true, false);
        when(poller.getStatus()).thenReturn(new IgraphConstraintTreePoller.Status(
                "SUBMITTED", 1, 2, 100, 4096, 2000000, 1900000, "check build status"));
        var client = WebTestClient.bindToRouterFunction(new IgraphConstraintTreeServer(poller, leader)
                .igraphConstraintTreeRoutes()).build();
        client.get().uri("/rtp_llm/constraint_tree/source/status").exchange().expectStatus().isOk()
                .expectBody().jsonPath("$.state").isEqualTo("SUBMITTED");
        client.post().uri("/rtp_llm/constraint_tree/source/refresh").exchange().expectStatus().isAccepted()
                .expectBody().jsonPath("$.accepted").isEqualTo(true);
        client.post().uri("/rtp_llm/constraint_tree/source/refresh").exchange().expectStatus().isEqualTo(409);
        when(leader.isMaster()).thenReturn(false);
        client.post().uri("/rtp_llm/constraint_tree/source/refresh").exchange().expectStatus().isEqualTo(503);
        verify(poller, times(2)).trigger();
    }
}
