package org.flexlb.service.monitor;

import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.STATUS_FAILURE_DETAIL;

@Component
public class StatusFailureReporter {
    private final FlexMonitor monitor;

    public StatusFailureReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        monitor.register(STATUS_FAILURE_DETAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    public void report(
            RoleType role,
            String operation,
            Stage stage,
            String reason,
            GrpcFailure.Analysis failure) {
        try {
            FlexMetricTags tags =
                    FlexMetricTags.of(
                            "role",
                            role == null ? "" : role.getCode(),
                            "operation",
                            operation,
                            "stage",
                            stage.name(),
                            "reason",
                            reason,
                            "origin",
                            failure != null && failure.confirmedLocalFailure()
                                    ? "MASTER"
                                    : "UNCONFIRMED",
                            "grpc_code",
                            failure == null ? "NONE" : failure.grpcCode().name());
            monitor.report(STATUS_FAILURE_DETAIL, tags, 1.0);
        } catch (RuntimeException | Error ignoredDiagnosticFailure) {
            // Failure diagnostics must not interrupt status processing or cleanup.
        }
    }
}
