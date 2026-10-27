package org.flexlb.mock.api.metrics;

import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.*;
import static org.flexlb.mock.api.Endpoint.Kind.read;

public interface PrometheusApi {
    @Endpoint(value="prometheus/query", kind=read) QueryResult query(Promql args);
}
