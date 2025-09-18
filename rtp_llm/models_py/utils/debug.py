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
