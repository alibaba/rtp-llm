"""Explicit N6 variant of the inherited one-request native RDMA role."""

from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_role import main


if __name__ == "__main__":
    main(sharded=True)
