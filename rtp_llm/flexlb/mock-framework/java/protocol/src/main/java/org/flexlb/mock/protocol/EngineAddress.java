package org.flexlb.mock.protocol;

import java.net.InetSocketAddress;

/** Process placement, shared by all instances in a Pod; never engine model configuration. */
public record EngineAddress(String bind, String advertise) {
    public static EngineAddress local() {
        String bind=System.getenv().getOrDefault("MOCK_BIND_HOST","127.0.0.1");
        String advertise=System.getenv("MOCK_ADVERTISE_HOST");
        if (advertise == null) {
            if (bind.equals("0.0.0.0") || bind.equals("::")) {
                try { advertise = java.net.InetAddress.getLocalHost().getHostAddress(); }
                catch (java.net.UnknownHostException error) { throw new IllegalStateException("Cannot determine Engine advertise address", error); }
            } else advertise = bind;
        }
        if(bind.isBlank() || advertise.isBlank() || advertise.equals("0.0.0.0") || advertise.equals("::") || advertise.contains("/"))
            throw new IllegalArgumentException("Invalid Engine bind/advertise host");
        return new EngineAddress(bind,advertise);
    }
    public InetSocketAddress listener(int port) { return new InetSocketAddress(bind,port); }
    public String rpc(int port) { return (advertise.contains(":")?"["+advertise+"]":advertise)+":"+port; }
    public String http(int port) { return "http://"+rpc(port); }
}
