package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import reactor.core.publisher.Mono;

public interface FeClient {

    Mono<JsonNode> post(String feBaseUrl, String fePath, ObjectNode body);
}
