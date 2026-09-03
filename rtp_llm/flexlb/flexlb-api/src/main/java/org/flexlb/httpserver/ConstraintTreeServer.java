package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.constraint.ConstraintTreeBuildService;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.flexlb.constraint.ConstraintTreeModels.SerializedArtifact;
import org.flexlb.constraint.ConstraintTreeModels.Submission;
import org.flexlb.transport.GeneralHttpNettyService;
import org.springframework.context.annotation.Bean;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.server.RouterFunction;
import org.springframework.web.reactive.function.server.ServerRequest;
import org.springframework.web.reactive.function.server.ServerResponse;
import org.springframework.web.server.ServerWebInputException;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.time.Duration;
import java.util.Optional;

import static org.springframework.web.reactive.function.server.RequestPredicates.accept;
import static org.springframework.web.reactive.function.server.RouterFunctions.route;

@Slf4j
@Component
public class ConstraintTreeServer {

    static final String BUILD_PATH = "/rtp_llm/constraint_tree/build";
    static final String STATUS_PATH = "/rtp_llm/constraint_tree/status";
    static final String ARTIFACT_PATH = "/rtp_llm/constraint_tree/artifact";

    private final ConstraintTreeBuildService buildService;
    private final LBStatusConsistencyService consistencyService;
    private final GeneralHttpNettyService httpService;

    public ConstraintTreeServer(ConstraintTreeBuildService buildService,
                                LBStatusConsistencyService consistencyService,
                                GeneralHttpNettyService httpService) {
        this.buildService = buildService;
        this.consistencyService = consistencyService;
        this.httpService = httpService;
    }

    @Bean
    public RouterFunction<ServerResponse> constraintTreeRoutes() {
        return route()
                .POST(BUILD_PATH, accept(MediaType.APPLICATION_JSON), this::build)
                .GET(STATUS_PATH, this::status)
                .GET(ARTIFACT_PATH, this::artifact)
                .build();
    }

    Mono<ServerResponse> build(ServerRequest request) {
        return request.bodyToMono(BuildRequest.class)
                .flatMap(buildRequest -> {
                    buildService.validateRequest(buildRequest);
                    if (!consistencyService.isMaster()) {
                        return forwardToMaster(buildRequest);
                    }
                    return submissionResponse(buildService.submit(buildRequest));
                })
                .switchIfEmpty(ServerResponse.badRequest()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(new ErrorResponse("request body must not be empty")))
                .onErrorResume(IllegalArgumentException.class, e -> ServerResponse.badRequest()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(new ErrorResponse(e.getMessage())))
                .onErrorResume(ServerWebInputException.class, e -> ServerResponse.badRequest()
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(new ErrorResponse("request body is not valid JSON")))
                .onErrorResume(e -> {
                    log.error("constraint tree build request failed", e);
                    return ServerResponse.status(500)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(new ErrorResponse("constraint tree build request failed"));
                });
    }

    Mono<ServerResponse> status(ServerRequest request) {
        if (!consistencyService.isMaster()) {
            return redirectToMaster(STATUS_PATH);
        }
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(buildService.getStatus());
    }

    Mono<ServerResponse> artifact(ServerRequest request) {
        if (!consistencyService.isMaster()) {
            return redirectToMaster(ARTIFACT_PATH);
        }
        boolean backup = request.queryParam("slot").map("backup"::equalsIgnoreCase).orElse(false);
        Optional<SerializedArtifact> artifact = backup
                ? buildService.getBackupArtifact()
                : buildService.getCurrentArtifact();
        return artifact
                .map(this::artifactResponse)
                .orElseGet(() -> ServerResponse.notFound().build());
    }

    private Mono<ServerResponse> artifactResponse(SerializedArtifact artifact) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .header("X-Constraint-Tree-Version", Long.toString(artifact.version()))
                .bodyValue(artifact.payload());
    }

    private Mono<ServerResponse> forwardToMaster(BuildRequest request) {
        URI masterUri = masterUri();
        if (masterUri == null) {
            return ServerResponse.status(503)
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(new ErrorResponse("constraint-tree leader is unavailable"));
        }
        return httpService.request(request, masterUri, BUILD_PATH, Submission.class)
                .timeout(Duration.ofSeconds(10))
                .flatMap(this::submissionResponse)
                .onErrorResume(e -> {
                    log.error("forward constraint tree build request to {} failed", masterUri, e);
                    return ServerResponse.status(502)
                            .contentType(MediaType.APPLICATION_JSON)
                            .bodyValue(new ErrorResponse("failed to forward request to constraint-tree leader"));
                });
    }

    private Mono<ServerResponse> submissionResponse(Submission submission) {
        return ServerResponse.ok()
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(submission);
    }

    private Mono<ServerResponse> redirectToMaster(String path) {
        URI masterUri = masterUri();
        if (masterUri == null) {
            return ServerResponse.status(503).build();
        }
        return ServerResponse.temporaryRedirect(masterUri.resolve(path)).build();
    }

    private URI masterUri() {
        String host = consistencyService.getMasterHostIpPort();
        if (host == null || host.isBlank()) {
            return null;
        }
        return URI.create("http://" + host);
    }

    private record ErrorResponse(String error) {
    }
}
