package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class OptimizerRemoveInstanceResponse {

    private CommonResponseHeader header;
}
