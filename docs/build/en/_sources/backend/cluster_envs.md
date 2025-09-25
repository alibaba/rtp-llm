# Cluster Environments

The following describes the additional environment variables and resource labels that need to be set in different cluster and GPU card type environments.

| cluster | gpu_card        | networker | tp_envs    | pd_envs | deepep_envs | deepep_extra_resource |
|:-       |:-               | :- | :-       | :- | :- |:-              |
| KS01    | L20Z/L20X/L20Y  | With IP| <pre>NCCL_SOCKET_IFNAME=eth0<br>NCCL_IB_GID_INDEX=3</pre> | <pre></pre> | <pre>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0<br>NVSHMEM_IB_GID_INDEX=3</pre> | TEXT:alibabacloud.com/gpu-gdrcopy-enable=true  |
| Kingsoft Cloud     | L20Z/L20X/L20Y  | Without IP| <pre>NCCL_SOCKET_IFNAME=bond0<br>NCCL_IB_TC=40</pre> | <pre></pre> | <pre>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0<br>NVSHMEM_IB_TRAFFIC_CLASS=40</pre> | TEXT:alibabacloud.com/gpu-gdrcopy-enable=true  |
| Qinghai     | L20Z/L20X/L20Y  | Without IP| <pre>NCCL_SOCKET_IFNAME=bond0</pre> | <pre></pre> | <pre>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0</pre> | TEXT:alibabacloud.com/gpu-gdrcopy-enable=true  |
| NM125      | L20X/H20 | With IP | <pre>NCCL_SOCKET_IFNAME=eth0<br>NCCL_IB_GID_INDEX=3<br>NCCL_IB_TC=136</pre> | <pre>ACCL_RX_DEPTH=32<br>ACCL_TX_DEPTH=512<br>ACCL_SOFT_TX_DEPTH=8192<br>ACCL_MAX_USER_MR_GB=2000</pre> | <pre>NVSHMEM_IB_GID_INDEX=3<br>NVSHMEM_IB_TRAFFIC_CLASS=136</pre> | TEXT:alibabacloud.com/gpu-gdrcopy-enable=true  |
| HHPAI     | H20 | With IP  | <pre>NCCL_SOCKET_IFNAME=eth0<br>NCCL_IB_GID_INDEX=3</pre> | <pre></pre> | <pre>NVSHMEM_IB_GID_INDEX=3</pre> | TEXT:alibabacloud.com/gpu-gdrcopy-enable=true  |
| Lingjun      | H20 | With IP | <pre></pre> | <pre></pre> | <pre></pre> | TEXT:alibabacloud.com/gpu-gdrcopy-enable=true  |
| Volcano      | H20 | With IP | <pre>NCCL_SOCKET_IFNAME=eth0<br>NCCL_IB_GID_INDEX=5<br>NCCL_IB_TC=236</pre> | <pre>TOPO_VISIBLE_NICS=mlx5_0,mlx5_1,mlx5_2,mlx5_3</pre> | <pre>NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0<br>NVSHMEM_IB_GID_INDEX=5<br>NVSHMEM_IB_TRAFFIC_CLASS=236</pre> | TEXT:alibabacloud.com/gpu-gdrcopy-enable=true  |

# Docker Container Arguments

| arg | value|
| :- | :-|
|VOLUME_MOUNTS | (/dev/shm:Memory:16Gi)|
|DOWNWARD_API | (metadata.labels,metadata.annotations:/etc/podinfo)|
|ULIMITS | (core:-1:-1,memlock:-1:-1)|