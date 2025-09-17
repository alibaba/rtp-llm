RTP LLM supports using 3FS for persistent caching of KVCache, providing a secondary cache for KVCache.

When model computational complexity is high or GPU memory is insufficient, the already computed KVCache can be offloaded to 3FS and reused in subsequent inference from the KVCache stored in 3FS, reducing TTFT (Time To First Token); at the same time, GPU memory computing resources can also be freed up for other uses.

The framework integrates using the 3FS Native Client (USRBIO API) approach to improve KVCache read and write performance.



# Usage

Currently, 3FS is disabled by default in production. You can enable 3FS by following these steps:



1. 3FS is currently only used in CUDA environments. During compilation, 3FS needs to be actively compiled, and 3FS needs to be actively enabled at runtime.

```shell
bazelisk build --config=cuda12_6 --config=3fs xxx
```



2. Business containers need to add the following configuration:

```shell
    "container_configs": [
      # "ULIMITS=(core:-1:-1,memlock:-1:-1)",
      "JSON_VOLUME_MOUNTS=[{\"mountPath\":\"/dev/shm\",\"name\":\"shm-vol\"}, {\"mountPath\":\"/3fs/stage/\",\"name\":\"shared-mountpoint\",\"mountPropagation\":\"HostToContainer\"}]",
      "JSON_VOLUMES=[{\"name\":\"shm-vol\",\"emptyDir\":{\"medium\": \"Memory\",\"sizeLimit\":\"40Gi\"}},{\"name\":\"shared-mountpoint\",\"emptyDir\":{}}]",
    ],
    "meta_tag_list": [
      {
        "key": "alibabacloud.com/fuse-service",
        "value": "rtp-3fs-h20"
      }
    ]
```

> Note: The shm size needs to be calculated. One TP requires 8GB (4GB for reading and 4GB for writing). For 4TP, 3FS requires 32GB of shm. Other services also occupy some shm, so it's set to 40GB here. Additionally, the memory size applied for by the business cannot be lower than this shm size.



Add RDMA loopback prevention label:

```shell
{
  "name": "storage.3fs.release=rtp",
  "type": "PROHIBIT_TEXT",
  "amount": 0
}
```



3. Add environment variables or args at startup:

```shell
# Environment variables
REUSE_CACHE=1 ENABLE_3FS=1

# Args
--reuse_cache True --enable_3fs True
```



4. Perform some validation after the container starts:

After the container is up, verify that the 3fs directory exists:

```shell
/3fs/stage/3fs/rtp_llm/
```

Verify that the shm size is correct:

```shell
df -h
```

Check if /dev/shm is the requested size, such as 40GB. If not, confirm that the total memory requested is not less than 40GB, as shm cannot be smaller than the total memory requested by the pod.



# Deployment Method

Considering the group's RDMA network architecture, there will be RDMA performance loss when communicating between different network clusters, so currently one network cluster deploys a separate 3FS storage cluster.

3FS has already been deployed in HPAI NA61, NA175, Kingsoft Cloud, and Beijing Lingjun data centers. Deployment in other data centers is ongoing.