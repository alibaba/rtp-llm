// SPDX-License-Identifier: Apache-2.0
/* Read-only libibverbs enumeration/open/query/close probe. No QP, MR or traffic. */
#include <infiniband/verbs.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void json_string(const char *value) {
    const unsigned char *p = (const unsigned char *)(value ? value : "");
    putchar('"');
    for (; *p; ++p) {
        if (*p == '"' || *p == '\\') { putchar('\\'); putchar(*p); }
        else if (*p < 32) printf("\\u%04x", (unsigned int)*p);
        else putchar(*p);
    }
    putchar('"');
}

static void begin(const char *status, const char *stage, const char *device) {
    printf("{\"status\":"); json_string(status);
    printf(",\"stage\":"); json_string(stage);
    if (device) { printf(",\"device\":"); json_string(device); }
}

static void error_event(const char *stage, const char *device, unsigned int port,
                        int return_code, int saved_errno) {
    /* Verbs query APIs return a positive errno value; open/get-list set errno. */
    int error_code = return_code > 0 ? return_code : saved_errno;
    begin("FAIL", stage, device);
    if (port) printf(",\"port\":%u", port);
    printf(",\"return_code\":%d,\"errno\":%d,\"error_code\":%d,\"message\":",
           return_code, saved_errno, error_code);
    json_string(error_code ? strerror(error_code) : "API failed without an errno value");
    puts("}");
}

static const char *state_name(enum ibv_port_state state) {
    switch (state) {
        case IBV_PORT_NOP: return "NOP";
        case IBV_PORT_DOWN: return "DOWN";
        case IBV_PORT_INIT: return "INIT";
        case IBV_PORT_ARMED: return "ARMED";
        case IBV_PORT_ACTIVE: return "ACTIVE";
        case IBV_PORT_ACTIVE_DEFER: return "ACTIVE_DEFER";
        default: return "UNKNOWN";
    }
}

static const char *link_name(unsigned int layer) {
    switch (layer) {
        case IBV_LINK_LAYER_INFINIBAND: return "INFINIBAND";
        case IBV_LINK_LAYER_ETHERNET: return "ETHERNET";
        default: return "UNSPECIFIED";
    }
}

int main(void) {
    int device_count = 0, opened = 0, queried_devices = 0, closed = 0, failures = 0;
    unsigned int queried_ports = 0, active_ports = 0;
    /* Keep completed JSON lines available even if an external timeout fires. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    errno = 0;
    struct ibv_device **devices = ibv_get_device_list(&device_count);
    if (!devices) { error_event("enumerate", NULL, 0, -1, errno); return 1; }
    begin(device_count ? "PASS" : "FAIL", "enumerate", NULL);
    printf(",\"device_count\":%d}", device_count); putchar('\n');
    if (!device_count) failures++;

    for (int i = 0; i < device_count; ++i) {
        const char *name = ibv_get_device_name(devices[i]);
        errno = 0;
        struct ibv_context *context = ibv_open_device(devices[i]);
        if (!context) { error_event("open_device", name, 0, -1, errno); failures++; continue; }
        opened++;
        begin("PASS", "open_device", name); puts("}");

        struct ibv_device_attr device_attr;
        memset(&device_attr, 0, sizeof(device_attr));
        errno = 0;
        int rc = ibv_query_device(context, &device_attr);
        if (rc) { error_event("query_device", name, 0, rc, errno); failures++; }
        else {
            queried_devices++;
            device_attr.fw_ver[sizeof(device_attr.fw_ver) - 1] = '\0';
            begin("PASS", "query_device", name);
            printf(",\"firmware_version\":"); json_string(device_attr.fw_ver);
            printf(",\"physical_ports\":%u,\"vendor_id\":%u,\"vendor_part_id\":%u,\"hardware_version\":%u}\n",
                   (unsigned int)device_attr.phys_port_cnt, device_attr.vendor_id,
                   device_attr.vendor_part_id, device_attr.hw_ver);
            for (unsigned int port = 1; port <= device_attr.phys_port_cnt; ++port) {
                struct ibv_port_attr port_attr;
                memset(&port_attr, 0, sizeof(port_attr));
                errno = 0;
                rc = ibv_query_port(context, (uint8_t)port, &port_attr);
                if (rc) { error_event("query_port", name, port, rc, errno); failures++; continue; }
                queried_ports++;
                if (port_attr.state == IBV_PORT_ACTIVE) active_ports++;
                begin("PASS", "query_port", name);
                printf(",\"port\":%u,\"state_code\":%u,\"state\":", port, (unsigned int)port_attr.state);
                json_string(state_name(port_attr.state));
                printf(",\"link_layer_code\":%u,\"link_layer\":", (unsigned int)port_attr.link_layer);
                json_string(link_name(port_attr.link_layer));
                printf(",\"active\":%s,\"physical_state_code\":%u,\"active_mtu_code\":%u,\"active_width_code\":%u,\"active_speed_code\":%u,\"lid\":%u}\n",
                       port_attr.state == IBV_PORT_ACTIVE ? "true" : "false",
                       (unsigned int)port_attr.phys_state, (unsigned int)port_attr.active_mtu,
                       (unsigned int)port_attr.active_width, (unsigned int)port_attr.active_speed,
                       (unsigned int)port_attr.lid);
            }
        }
        /* Query failures still reach close; no PD/CQ/QP/MR was allocated. */
        errno = 0;
        rc = ibv_close_device(context);
        if (rc) { error_event("close_device", name, 0, rc, errno); failures++; }
        else { closed++; begin("PASS", "close_device", name); puts("}"); }
    }
    ibv_free_device_list(devices);
    begin(failures ? "FAIL" : "PASS", "complete", NULL);
    printf(",\"device_count\":%d,\"opened\":%d,\"queried_devices\":%d,\"closed\":%d,\"queried_ports\":%u,\"active_ports\":%u,\"api_failures\":%d,\"traffic_sent\":false,\"qp_created\":false,\"gpu_accessed\":false}\n",
           device_count, opened, queried_devices, closed, queried_ports, active_ports, failures);
    return failures ? 1 : 0;
}
