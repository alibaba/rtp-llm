package org.flexlb.mock.api.client;

import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.model.Types.*;
import static org.flexlb.mock.api.Endpoint.Kind.*;

public interface ClientApi {
    @Endpoint(value="client/traffic/run", kind=action, background=true) TrafficReceipt run(Traffic args);
    @Endpoint(value="client/traffic/stop", kind=action) StopReceipt stop(None args);
    @Endpoint(value="client/traffic/drain", kind=action) DrainReceipt drain(None args);
    @Endpoint(value="client/traffic/qps", kind=read) MetricValue qps(ClientRange args);
    @Endpoint(value="client/traffic/success_qps", kind=read) MetricValue successQps(ClientRange args);
    @Endpoint(value="client/traffic/failure_qps", kind=read) MetricValue failureQps(ClientRange args);
    @Endpoint(value="client/traffic/success_rate", kind=read) MetricValue successRate(ClientRange args);
    @Endpoint(value="client/requests/ttft", kind=read, unit="s") MetricValue ttft(Ttft args);
}
