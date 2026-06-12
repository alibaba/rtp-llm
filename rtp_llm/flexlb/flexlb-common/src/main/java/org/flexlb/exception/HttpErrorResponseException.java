package org.flexlb.exception;

import lombok.Getter;

/**
 * The HTTP exchange completed but the upstream answered a non-2xx status. The raw
 * response body is preserved so callers whose upstream encodes typed business errors
 * in non-2xx bodies (e.g. {@code /rtp_llm/batch_schedule}) can recover them instead
 * of collapsing everything into a transport failure.
 */
@Getter
public class HttpErrorResponseException extends RuntimeException {

    private static final int MESSAGE_BODY_LIMIT = 256;

    private final int statusCode;
    private final String body;

    public HttpErrorResponseException(int statusCode, String body) {
        super("http error, httpStatusCode=" + statusCode + ", body=" + summarize(body));
        this.statusCode = statusCode;
        this.body = body;
    }

    /** Keeps the exception message bounded; the full body stays available via {@link #getBody()}. */
    private static String summarize(String body) {
        if (body == null || body.length() <= MESSAGE_BODY_LIMIT) {
            return String.valueOf(body);
        }
        return body.substring(0, MESSAGE_BODY_LIMIT) + "...(" + body.length() + " chars total)";
    }
}
