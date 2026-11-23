#pragma once

extern "C" {
const char* qwen38_lsa_error();
int qwen38_lsa_uid_size();
int qwen38_lsa_unique_id(void* output);
int qwen38_lsa_device(int device, char* bus_id, int size);
int qwen38_lsa_peer(int device, const char* bus_id);
int qwen38_lsa_allocate(int device, int hidden, void** output);
int qwen38_lsa_set_timeout(void* handle, int timeout_ms);
int qwen38_lsa_connect(void* handle, int rank, const void* unique_id);
int qwen38_lsa_check_capability(void* handle);
int qwen38_lsa_activate(void* handle);
int qwen38_lsa_check_activation(void* handle);
int qwen38_lsa_abort(void* handle);
int qwen38_lsa_apply(
    void* handle, const void* input, const void* shared, const void* gate, void* output, int rows, void* stream);
int qwen38_lsa_reference(void* handle, const void* input, void* output, int rows, void* stream);
void qwen38_lsa_destroy(void* handle);
}
