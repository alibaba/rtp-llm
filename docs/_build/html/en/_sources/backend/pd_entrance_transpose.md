# PD Disaggregation Transpose

## Background

The PD disaggregation implementation defaults to using the Prefill node as the request entry point. After the Prefill instance computes the first token, it needs to transfer the KV cache to the Decode instance. Subsequent token generation is completed on the Decode instance, and the results are streamed back to the user through the Prefill instance. In fact, the Prefill inference work is completed after computing the first token and transferring the KV Cache to the Decode instance. However, in the scenario where Prefill is the front-end stream receiver, the instance still needs to act as a relay to return the tokens output by the Decode instance to the user.

To solve this problem, RTP-LLM provides the capability of PD disaggregation stream receiver inversion, that is, making the Decode instance the request entry point. In the Decode front-end stream receiving implementation, the Decode instance will send an async loadCache RPC request to the Prefill instance. After receiving the request, the Prefill instance will start the first token computation, and the generated KV cache will be transmitted in units of model layers. Each time a layer's KV cache computation is completed, the Prefill instance will call the transfer RPC interface of the Decode instance to let it use RDMA read to read the corresponding KV cache block from the Prefill instance. After the KV Cache is fully loaded, the Decode instance will compute subsequent tokens locally and stream the results back to the user.

## Configuration

Currently, PD inversion is disabled by default in production. If you need to enable the PD inversion stream receiving capability, you need to configure the following environment variables when starting the Decode/Prefill instances:

```
DECODE_ENTRANCE=1
```