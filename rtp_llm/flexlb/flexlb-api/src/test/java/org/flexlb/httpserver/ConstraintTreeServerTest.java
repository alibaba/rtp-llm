package org.flexlb.httpserver;

import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.constraint.ConstraintTreeBuildService;
import org.flexlb.constraint.ConstraintTreeCsrCodec;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.flexlb.constraint.ConstraintTreeModels.BuildState;
import org.flexlb.constraint.ConstraintTreeModels.Submission;
import org.flexlb.constraint.ConstraintTreeModels.SubmissionState;
import org.flexlb.transport.GeneralHttpNettyService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.reactive.server.WebTestClient;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;

import static org.junit.jupiter.api.Assertions.fail;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class ConstraintTreeServerTest {

    private final ConstraintTreeBuildService buildService = new ConstraintTreeBuildService();
    private final LBStatusConsistencyService consistencyService = mock(LBStatusConsistencyService.class);
    private final GeneralHttpNettyService httpService = mock(GeneralHttpNettyService.class);
    private final WebTestClient client = WebTestClient
            .bindToRouterFunction(new ConstraintTreeServer(buildService, consistencyService, httpService)
                    .constraintTreeRoutes())
            .build();

    ConstraintTreeServerTest() {
        when(consistencyService.isMaster()).thenReturn(true);
    }

    @AfterEach
    void tearDown() {
        buildService.destroy();
    }

    @Test
    void exposesBuildStatusAndWorkerCompatibleArtifact() throws Exception {
        client.post()
                .uri("/rtp_llm/constraint_tree/build")
                .header("Content-Type", "application/json")
                .bodyValue("""
                        {
                          "version": 42,
                          "model": "gul_item",
                          "rq_token_ids": [[169967, 216546], [169967, 215835, 7]]
                        }
                        """)
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.state").isEqualTo("ACCEPTED")
                .jsonPath("$.requested_version").isEqualTo(42);

        awaitBuilt();

        client.get()
                .uri("/rtp_llm/constraint_tree/status")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.state").isEqualTo("PARTIALLY_PUBLISHED")
                .jsonPath("$.active_version").isEqualTo(42);

        client.get()
                .uri("/rtp_llm/constraint_tree/artifact")
                .exchange()
                .expectStatus().isOk()
                .expectHeader().contentType(MediaType.APPLICATION_OCTET_STREAM)
                .expectBody(byte[].class)
                .value(payload -> {
                    ConstraintTreeCsrCodec.DecodedArtifact artifact = ConstraintTreeCsrCodec.decode(payload);
                    org.junit.jupiter.api.Assertions.assertEquals(42, artifact.version());
                    org.junit.jupiter.api.Assertions.assertEquals(1699, artifact.startTokenId());
                    org.junit.jupiter.api.Assertions.assertEquals(151645, artifact.endTokenId());
                    org.junit.jupiter.api.Assertions.assertEquals(2, artifact.sidCount());
                });
    }

    @Test
    void rejectsInvalidMetadataAtTheHttpBoundary() {
        client.post()
                .uri("/rtp_llm/constraint_tree/build")
                .header("Content-Type", "application/json")
                .bodyValue("{\"version\":0,\"model\":\"gul_item\",\"sids\":[\"1_3\"]}")
                .exchange()
                .expectStatus().isBadRequest()
                .expectBody()
                .jsonPath("$.error").isEqualTo("version must be greater than zero");
    }

    @Test
    void rejectsMalformedJsonAtTheHttpBoundary() {
        client.post()
                .uri(ConstraintTreeServer.BUILD_PATH)
                .header("Content-Type", "application/json")
                .bodyValue("not-json")
                .exchange()
                .expectStatus().isBadRequest()
                .expectBody()
                .jsonPath("$.error").isEqualTo("request body is not valid JSON");
    }

    @Test
    void participantForwardsBuildToItsElectedLeader() {
        when(consistencyService.isMaster()).thenReturn(false);
        when(consistencyService.getMasterHostIpPort()).thenReturn("10.0.0.7:7001");
        Submission forwarded = new Submission(SubmissionState.ACCEPTED, 43, 43, "build accepted");
        when(httpService.request(
                any(BuildRequest.class),
                eq(URI.create("http://10.0.0.7:7001")),
                eq(ConstraintTreeServer.BUILD_PATH),
                eq(Submission.class))).thenReturn(Mono.just(forwarded));

        client.post()
                .uri(ConstraintTreeServer.BUILD_PATH)
                .header("Content-Type", "application/json")
                .bodyValue("{\"version\":43,\"model\":\"gul_item\",\"sids\":[\"1_3\"]}")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.state").isEqualTo("ACCEPTED")
                .jsonPath("$.requested_version").isEqualTo(43);

        verify(httpService).request(
                any(BuildRequest.class),
                eq(URI.create("http://10.0.0.7:7001")),
                eq(ConstraintTreeServer.BUILD_PATH),
                eq(Submission.class));
    }

    private void awaitBuilt() throws Exception {
        long deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
        while (System.nanoTime() < deadline) {
            BuildState state = buildService.getStatus().state();
            if (state == BuildState.READY || state == BuildState.PARTIALLY_PUBLISHED) {
                return;
            }
            Thread.sleep(5);
        }
        fail("timed out waiting for a built artifact, status=" + buildService.getStatus());
    }
}
