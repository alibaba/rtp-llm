package org.flexlb.service.batch;

import org.flexlb.domain.batch.PrefillBatchRequest;
import org.flexlb.domain.batch.SubmitBatchResponse;

/**
 * @author zjw
 * description:
 * date: 2025/3/16
 */
public interface BatchService {

    SubmitBatchResponse submitBatch(PrefillBatchRequest prefillBatchRequest);

}
