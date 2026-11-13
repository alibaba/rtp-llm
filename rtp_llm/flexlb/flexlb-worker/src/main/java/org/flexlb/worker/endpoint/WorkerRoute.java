package org.flexlb.worker.endpoint;

/**
 * Retains exact endpoint registrations shared with the scheduler that owns the request.
 */
public interface WorkerRoute {
    String requestId();

    WorkerRequest ctx();

    org.flexlb.worker.endpoint.PrefillEndpoint prefillEndpoint();

    org.flexlb.worker.endpoint.DecodeEndpoint decodeEndpoint();

    org.flexlb.worker.endpoint.PrefillEndpoint.RequestRegistration prefillRegistration();

    org.flexlb.worker.endpoint.DecodeEndpoint.RequestRegistration decodeRegistration();

    org.flexlb.worker.endpoint.EncoderEndpoint.RequestRegistration encoderRegistration();
}
