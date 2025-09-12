package org.flexlb.dao.pv;

import lombok.Data;

/**
 * PV日志数据
 */
@Data
public class PvLogData {
    private Object request;
    private Object response;
    private String error;
    private boolean success;
    
    public static PvLogData success(Object request, Object response) {
        PvLogData data = new PvLogData();
        data.setRequest(request);
        data.setResponse(response);
        data.setSuccess(true);
        return data;
    }
    
    public static PvLogData error(Object request, String error) {
        PvLogData data = new PvLogData();
        data.setRequest(request);
        data.setError(error);
        data.setSuccess(false);
        return data;
    }
}