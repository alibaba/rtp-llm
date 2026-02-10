def set_trace_on_tty():
    """
    启动一个连接到当前终端的 PDB 会话。
    在 Unix-like 系统上工作。
    """
    try:
        import pdb

        tty_r = open("/dev/tty", "r")
        tty_w = open("/dev/tty", "w")
        pdb.Pdb(stdin=tty_r, stdout=tty_w).set_trace()
    except OSError as e:
        print(f"Warning: Could not open /dev/tty: {e}. Skipping pdb.")
        import traceback

        traceback.print_exc()


def remote_debug_breakpoint(host="localhost", port=4444):
    """
    启动一个远程 PDB 会话，监听指定的主机和端口。
    使用 telnet 连接到该主机和端口以进行调试。
    """
    import debugpy

    debugpy.listen((host, port))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    debugpy.breakpoint()


import torch


def cudagraph_debug_kernel(
    data: torch.Tensor | None,
    info_id: int = 1,
    m: int = 0,
    n: int = 0,
    row_len: int = 0,
    name: str = "cudagraph_debug_kernel",
):
    if data is None:
        return
    print(f"{name} shape is {data.shape}")
    if data.dim() == 1:
        data = data.unsqueeze(0)
    data = data.contiguous().to(torch.float32)
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    row_len = data.size(1) if row_len == 0 else row_len
    n = data.size(1) if (n == 0 or n > data.size(1)) else n
    m = data.size(0) if (m == 0 or m > data.size(0)) else m
    rtp_llm_ops.debug_kernel(
        data=data,
        start_row=0,
        start_col=0,
        m=m,
        n=n,
        row_len=row_len,  # 每行的长度
        info_id=info_id,
    )
