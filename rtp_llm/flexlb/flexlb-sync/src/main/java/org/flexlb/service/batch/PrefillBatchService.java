package org.flexlb.service.batch;

import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.domain.batch.PrefillBatchRequest;
import org.flexlb.domain.batch.SubmitBatchResponse;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.HttpRequestUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.net.URI;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

@Component
public class PrefillBatchService implements BatchService {

    protected static final Logger log = LoggerFactory.getLogger("syncLogger");

    private final String path = "/v1/chat/completions/submit";
    private final GeneralHttpNettyService generalHttpNettyService;

    public PrefillBatchService(GeneralHttpNettyService generalHttpNettyService) {
        this.generalHttpNettyService = generalHttpNettyService;
    }

    @Override
    public SubmitBatchResponse submitBatch(PrefillBatchRequest prefillBatchRequest) {
        long start = System.currentTimeMillis();
        try {
            SubmitBatchResponse batchResponse = doSubmit(prefillBatchRequest).toFuture().get(500, TimeUnit.MILLISECONDS);
            if (batchResponse.getCode() != 200) {
                log.error("submit batch failed, batchId:{}, code:{}, message:{}", prefillBatchRequest.getBatchId(), batchResponse.getCode(), batchResponse.getMessage());
            }
            return batchResponse;
        } catch (TimeoutException timeoutException) {
            log.error("submit batch timeout 500ms, batchId:{}", prefillBatchRequest.getBatchId());
            return SubmitBatchResponse.error(StrategyErrorType.SUBMIT_BATCH_TO_ENGINE_TIMEOUT.getErrorMsg());
        } catch (Throwable e) {
            log.error("submit batch failed, batchId:{}", prefillBatchRequest.getBatchId(), e);
            return SubmitBatchResponse.error(StrategyErrorType.SUBMIT_BATCH_TO_ENGINE_ERROR.getErrorMsg());
        } finally {
            log.info("submit batch cost:{}, itemInBatch:{}", System.currentTimeMillis() - start,
                    prefillBatchRequest.getRequests() != null ? prefillBatchRequest.getRequests().size() : 0);
        }
    }

    private Mono<SubmitBatchResponse> doSubmit(PrefillBatchRequest req) {
        return Mono.just(req)
                .flatMap(r -> {
                    Endpoint endpoint = new Endpoint();
                    endpoint.setAddress(r.getIp() + ":" + r.getPort());
                    endpoint.setType(LoadBalanceStrategyEnum.SPECIFIED_IP_PORT.getName());
                    URI uri = HttpRequestUtils.createURI(endpoint);
                    if (uri == null) {
                        return Mono.error(new RuntimeException("prefill address is null."));
                    }
                    log.info("submit batch to: {}", uri);
                    return generalHttpNettyService.request(
                            r,
                            uri,
                            path,
                            SubmitBatchResponse.class);
                });
    }
}
