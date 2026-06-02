import torch

TORCH_DTYPE_SIZE_BYTES = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
    torch.int8: 1,
    torch.float8_e4m3fn: 1,
}


class LinearCacheConverter:
    def __init__(
        self,
        *,
        local_num_v_heads: int,
        head_v_dim: int,
        head_k_dim: int,
        ssm_state_dtype: torch.dtype,
        linear_conv_kernel_dim: int,
        qkv_size: int,
        conv_state_dtype: torch.dtype,
    ) -> None:
        self.local_num_v_heads = local_num_v_heads
        self.head_v_dim = head_v_dim
        self.head_k_dim = head_k_dim
        self.ssm_state_dtype = ssm_state_dtype
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.qkv_size = qkv_size
        self.conv_state_dtype = conv_state_dtype

        self.ssm_state_item_size = self.dtype_size_bytes(self.ssm_state_dtype)
        self.conv_state_item_size = self.dtype_size_bytes(self.conv_state_dtype)
        self.ssm_state_size = self.local_num_v_heads * self.head_v_dim * self.head_k_dim
        self.conv_state_size = (self.linear_conv_kernel_dim - 1) * self.qkv_size
        self.ssm_state_size_bytes = self.ssm_state_size * self.ssm_state_item_size
        self.conv_state_size_bytes = self.conv_state_size * self.conv_state_item_size
        self.block_size_bytes = self.ssm_state_size_bytes + self.conv_state_size_bytes

    @staticmethod
    def dtype_size_bytes(dtype: torch.dtype) -> int:
        try:
            return TORCH_DTYPE_SIZE_BYTES[dtype]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported dtype for linear cache conversion: {dtype}"
            ) from exc

    @classmethod
    def _build_typed_storage_view(
        cls,
        base_tensor: torch.Tensor,
        dtype: torch.dtype,
        size: tuple[int, ...],
        stride_bytes: tuple[int, ...],
        storage_offset_bytes: int = 0,
    ) -> torch.Tensor:
        target_item_size = cls.dtype_size_bytes(dtype)
        base_item_size = cls.dtype_size_bytes(base_tensor.dtype)
        base_storage_offset_bytes = base_tensor.storage_offset() * base_item_size
        view_storage_offset_bytes = base_storage_offset_bytes + storage_offset_bytes

        if len(size) != len(stride_bytes):
            raise ValueError("size and stride_bytes must have the same rank")
        if view_storage_offset_bytes % target_item_size != 0:
            raise ValueError(
                f"storage_offset_bytes={view_storage_offset_bytes} is not aligned to {dtype}"
            )

        stride = []
        for stride_byte in stride_bytes:
            if stride_byte % target_item_size != 0:
                raise ValueError(
                    f"stride_bytes={stride_byte} is not aligned to target dtype {dtype}"
                )
            stride.append(stride_byte // target_item_size)

        view_tensor = torch.empty(0, device=base_tensor.device, dtype=dtype)
        view_tensor.set_(
            base_tensor.untyped_storage(),
            view_storage_offset_bytes // target_item_size,
            size,
            tuple(stride),
        )
        return view_tensor

    def get_block_size_bytes(self, kv_cache_tensor: torch.Tensor) -> int:
        return kv_cache_tensor.stride(0) * self.dtype_size_bytes(kv_cache_tensor.dtype)

    def get_conv_state_tensor(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        block_size_bytes = self.get_block_size_bytes(kv_cache_tensor)
        assert (
            block_size_bytes >= self.block_size_bytes
        ), "block_size is too small, please check seq_size_per_block"
        return self._build_typed_storage_view(
            kv_cache_tensor,
            self.conv_state_dtype,
            size=(
                kv_cache_tensor.shape[0],
                self.linear_conv_kernel_dim - 1,
                self.qkv_size,
            ),
            stride_bytes=(
                block_size_bytes,
                self.qkv_size * self.conv_state_item_size,
                self.conv_state_item_size,
            ),
            storage_offset_bytes=self.ssm_state_size_bytes,
        )

    def get_ssm_state_tensor(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        block_size_bytes = self.get_block_size_bytes(kv_cache_tensor)
        assert (
            block_size_bytes >= self.block_size_bytes
        ), "block_size is too small, please check seq_size_per_block"
        return self._build_typed_storage_view(
            kv_cache_tensor,
            self.ssm_state_dtype,
            size=(
                kv_cache_tensor.shape[0],
                self.local_num_v_heads,
                self.head_v_dim,
                self.head_k_dim,
            ),
            stride_bytes=(
                block_size_bytes,
                self.head_v_dim * self.head_k_dim * self.ssm_state_item_size,
                self.head_k_dim * self.ssm_state_item_size,
                self.ssm_state_item_size,
            ),
        )
