import logging
import re
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from time import time

import requests  # Import the requests library for making HTTP requests
import torch

from .core import COMMON_PREFIX, SharedMemIpcMeta, SharedMemoryIPCHelper


@dataclass
class NamedTensor:
    """
    A simple data container for associating a name (string) with a PyTorch tensor.

    Attributes:
        name (str): The name of the tensor (e.g., a weight name in a model).
        tensor (torch.Tensor): The PyTorch tensor data.
    """

    name: str
    tensor: torch.Tensor

    def __str__(self):
        return f"{self.name}[{self.tensor.shape}]"

    def __repr__(self):
        return self.__str__()


def get_expert_id(name: str) -> int:
    try:
        extracted = name[name.find("experts.") + len("experts.") :]
        extracted = extracted[: extracted.find(".")]
        return int(extracted)
    except Exception as e:
        raise Exception(f"fail to get experts id from given weight name: {name}")


def preprocess(nt: NamedTensor) -> NamedTensor:
    # convert nt.name(huggingface) to rtp-llm name
    REPLACEMENT = {
        "model.embed_tokens.weight": "model.embedding",
        "input_layernorm.weight": "pre_layernorm_weights.gamma",
        "input_layernorm.bias": "pre_layernorm_weights.beta",
        "post_attention_layernorm.weight": "post_layernorm_weights.gamma",
        "self_attn.o_proj.weight": "self_attention_weights.attention_output_weight.kernel",
        "self_attn.o_proj.bias": "self_attention_weights.attention_output_weight.bias",  # error.
        "self_attn.qkv_proj.weight": "self_attention_weights.query_weight.kernel",
        "self_attn.qkv_proj.bias": "self_attention_weights.query_weight.bias",
        "model.norm.weight": "model.final_layernorm.gamma",
        "self_attn.k_norm.weight": "self_attention_weights.k_layernorm.gamma",
        "self_attn.q_norm.weight": "self_attention_weights.q_layernorm.gamma",
        "self_attn.k_norm.bias": "self_attention_weights.k_layernorm.beta",
        "self_attn.q_norm.bias": "self_attention_weights.q_layernorm.beta",
    }
    for k, v in REPLACEMENT.items():
        if k in nt.name:
            nt.name = nt.name.replace(k, v)
            break

    if nt.name == "lm_head.weight":
        nt.name = "model.lm_head"

    # tensor preprocess function, convert hugging face layout to rtp-llm layout
    if "self_attention_weights.attention_output_weight.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "self_attention_weights.query_weight.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "__ffn_weights__.intermediate_weight13.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "__ffn_weights__.intermediate_weight2.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)

    """
    if "__moe_weights__.intermediate_weight.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    if "__moe_weights__.intermediate_weight2.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(1, 2)
    """
    if "__moe_weights__.gate.kernel" in nt.name:
        nt.tensor = nt.tensor.transpose(0, 1)
    return nt


class TensorBucketBuilder:
    """
    Groups and combines model weights, particularly those that are stored
    in a fragmented manner (like q_proj, k_proj, v_proj, or MoE expert weights)
    in the training system, before transferring them to the inference system.
    This helps in reducing the number of IPC calls and potentially makes the
    weights compatible with the inference system's expected format.
    """

    def __init__(self):
        """
        Initializes the TensorBucketBuilder with empty buffers for pending tensors
        and those ready to be sent, and sets the current layer ID to "Undefined".
        """
        self.pending_buffer: list[NamedTensor] = []
        self.bucket: list[NamedTensor] = []
        self.layer_id: str = "Undefined"

    def get_layer_id(self, name: str) -> str | None:
        """
        Extracts the layer ID from a tensor's name using a regular expression.

        Args:
            name (str): The name of the tensor.

        Returns:
            str | None: The extracted layer ID (e.g., 'model.layers.0.') or None if no match is found.
        """
        pattern = r"(model\.layers\.\d+\.)"
        match = re.search(pattern, name)

        if match:
            layer_id = match.group(1)
            return layer_id
        else:
            return None

    def flush(self) -> list[NamedTensor]:
        """
        Processes and combines the tensors in the pending buffer based on
        predefined rules (e.g., combining q_proj, k_proj, v_proj into qkv_proj,
        or grouping MoE expert weights), clears both buffers, and returns
        the list of combined and uncombined (but ready-to-send) NamedTensors.

        Returns:
            list[NamedTensor]: A list of NamedTensor objects ready for transport.
        """
        # Mainly combine three cases
        # One is q, k, v proj need to be combined into qkv_proj
        # One is feedforward a, b, c needs to be combined into feedforward a, c

        # One is expert.xx.down_proj needs to be combined into partial_moe_weights.intermediate_weight.kernel
        # One is expert.xx.up_proj needs to be combined into partial_moe_weights.intermediate_weight2.kernel
        # One is expert.xx.gate_proj needs to be combined into partial_moe_weights.gate.kernel

        qkv_weight: list[NamedTensor] = []
        qkv_bias: list[NamedTensor] = []
        expert_down_proj: list[NamedTensor] = []
        expert_up_proj: list[NamedTensor] = []
        expert_gate_proj: list[NamedTensor] = []
        mlp_weight: list[NamedTensor] = []

        for nt in self.pending_buffer:
            # qkv proj
            if ".q_proj.weight" in nt.name:
                qkv_weight.append(nt)
            if ".k_proj.weight" in nt.name:
                qkv_weight.append(nt)
            if ".v_proj.weight" in nt.name:
                qkv_weight.append(nt)

            if ".q_proj.bias" in nt.name:
                qkv_bias.append(nt)
            if ".k_proj.bias" in nt.name:
                qkv_bias.append(nt)
            if ".v_proj.bias" in nt.name:
                qkv_bias.append(nt)

            # expert
            if ".experts." in nt.name:
                if "down_proj" in nt.name:
                    expert_down_proj.append(nt)
                if "up_proj" in nt.name:
                    expert_up_proj.append(nt)
                if "gate_proj" in nt.name:
                    expert_gate_proj.append(nt)
            elif ".mlp." in nt.name:
                mlp_weight.append(nt)

        ret: list[NamedTensor] = self.bucket.copy()
        if len(qkv_weight) != 0:
            if len(qkv_weight) != 3:
                raise Exception(
                    f"qkv proj 的权重个数错误，应该有 3个 名为 q_proj, k_proj, v_proj 的权重，但接收到 {qkv_weight}"
                )
            q = [nt for nt in qkv_weight if "q_proj.weight" in nt.name]
            k = [nt for nt in qkv_weight if "k_proj.weight" in nt.name]
            v = [nt for nt in qkv_weight if "v_proj.weight" in nt.name]
            ret.append(
                NamedTensor(
                    name=self.layer_id + "self_attn.qkv_proj.weight",
                    tensor=torch.cat([nt.tensor for nt in q + k + v], dim=0),
                )
            )

        if len(qkv_bias) != 0:
            if len(qkv_bias) != 3:
                raise Exception(
                    f"qkv proj 的权重个数错误，应该有 3个 名为 q_proj, k_proj, v_proj 的权重，但接收到 {qkv_bias}"
                )
            q = [nt for nt in qkv_bias if "q_proj.bias" in nt.name]
            k = [nt for nt in qkv_bias if "k_proj.bias" in nt.name]
            v = [nt for nt in qkv_bias if "v_proj.bias" in nt.name]

            ret.append(
                NamedTensor(
                    name=self.layer_id + "self_attn.qkv_proj.bias",
                    tensor=torch.cat([nt.tensor for nt in q + k + v], dim=0),
                )
            )

        if len(expert_up_proj) != 0 and len(expert_gate_proj) != 0:
            up_projs = sorted(expert_up_proj, key=lambda x: get_expert_id(x.name))
            gt_projs = sorted(expert_gate_proj, key=lambda x: get_expert_id(x.name))

            tensors = []
            for up, gt in zip(up_projs, gt_projs):
                tensors.append(torch.cat([up.tensor, gt.tensor], dim=0).unsqueeze(0))

            ret.append(
                NamedTensor(
                    name=self.layer_id + "__moe_weights__.intermediate_weight.kernel",
                    tensor=torch.cat(tensors, dim=0),
                )
            )

        if len(expert_down_proj) != 0:
            ret.append(
                NamedTensor(
                    name=self.layer_id + "__moe_weights__.intermediate_weight2.kernel",
                    tensor=torch.cat(
                        [
                            nt.tensor.unsqueeze(0)
                            for nt in sorted(
                                expert_down_proj, key=lambda x: get_expert_id(x.name)
                            )
                        ],
                        dim=0,
                    ),
                )
            )

        if len(mlp_weight) > 0:
            if len(mlp_weight) == 1:
                # moe gate
                nt = mlp_weight[0]
                ret.append(
                    NamedTensor(
                        name=self.layer_id + "__moe_weights__.gate.kernel",
                        tensor=nt.tensor,
                    )
                )

            elif len(mlp_weight) == 3:
                up_proj = [nt for nt in mlp_weight if "up_proj.weight" in nt.name][
                    0
                ].tensor
                dn_proj = [nt for nt in mlp_weight if "down_proj.weight" in nt.name][
                    0
                ].tensor
                gt_proj = [nt for nt in mlp_weight if "gate_proj.weight" in nt.name][
                    0
                ].tensor

                ret.append(
                    NamedTensor(
                        name=self.layer_id
                        + "__ffn_weights__.intermediate_weight13.kernel",
                        tensor=torch.cat([gt_proj, up_proj], dim=0),
                    )
                )
                ret.append(
                    NamedTensor(
                        name=self.layer_id
                        + "__ffn_weights__.intermediate_weight2.kernel",
                        tensor=dn_proj,
                    )
                )
            else:
                raise Exception(f"mlp 的权重个数错误，接收到 {mlp_weight}")

        self.pending_buffer.clear()
        self.bucket.clear()
        return ret

    def combine(self, name: str, tensor: torch.Tensor) -> list[NamedTensor]:
        """
        Receives a tensor and either adds it to a pending buffer for later combination,
        adds it to a bucket for immediate sending, or flushes the pending tensors
        if a new layer's tensor is encountered.

        Args:
            name (str): The name of the tensor.
            tensor (torch.Tensor): The tensor data.

        Returns:
            list[NamedTensor]: A list of combined/ready-to-send tensors. This list is empty
                               unless a flush operation was triggered (by a layer ID change)
                               or if the tensor had no layer ID.
        """
        # Call this combine function, which will try to concatenate weights by layer id and send them,
        # which is usually faster and can solve some weight concatenation problems.
        INTRESTED_PREFIXS = [".q_proj.", ".k_proj.", ".v_proj.", ".experts.", ".mlp."]

        layer_id: str | None = self.get_layer_id(name)
        if layer_id is None:
            # If this is a weight without a layer id, we choose to send it individually
            return [NamedTensor(name, tensor)]

        ret: list[NamedTensor] = []
        if layer_id != self.layer_id:
            # If the layer ID changes, flush the current pending buffer and bucket
            ret = self.flush()
            self.layer_id = layer_id

        if any([prefix in name for prefix in INTRESTED_PREFIXS]):
            # If this is a tensor that needs to be concatenated, put it into the pending buffer
            self.pending_buffer.append(NamedTensor(name, tensor))

        else:
            # If this is a weight that does not need to be concatenated, put it directly into the bucket
            self.bucket.append(NamedTensor(name, tensor))

        return ret


class TensorTransportClient:
    def __init__(
        self,
        url: str = "locahost:26006/update_weight",
        shm_size: int = 8 * 1024 * 1024 * 1024,
    ):
        """
        Initializes the TensorTransportClient, creating a single, large
        shared memory block for efficient repeated transfers.

        Args:
            url (str): The target URL for the POST request to notify the server
                       about the shared memory data. Defaults to "locahost:26006/update_weight".
            shm_size (int): The maximum size (in bytes) of the shared memory
                            block to pre-allocate. This should be large enough
                            to accommodate the largest batch of combined tensors you plan to send.
                            Defaults to 8GB.
        """
        self.tensor_bucket = TensorBucketBuilder()
        self.cpu_ipc: SharedMemoryIPCHelper = SharedMemoryIPCHelper()
        self.shm_size = shm_size

        self.shm_name = f"{COMMON_PREFIX}_persistent_{uuid.uuid4().__str__()}"
        self.url = url
        try:
            # Create a single, persistent shared memory block
            self.shm = shared_memory.SharedMemory(
                create=True, size=self.shm_size, name=self.shm_name
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create persistent shared memory block: {e}")

    def _send(self, encoded_metas: list[SharedMemIpcMeta]) -> None:
        """
        Internal method to send the IPC metadata to the server via an HTTP POST request.

        Args:
            encoded_metas (list[SharedMemIpcMeta]): A list of metadata objects
                                                    describing the tensors' locations in SHM.

        Raises:
            requests.exceptions.RequestException: If there's an error during the HTTP request.
            IOError: If the server returns a non-200 status code.
            Exception: If the server returns a 'status': 'error' in its JSON response.
        """
        if not encoded_metas:
            return

        try:
            response = requests.post(
                self.url,
                json={
                    "time": time(),
                    "desc": [m.encode() for m in encoded_metas],
                    "method": "shm",
                },
            )

            if response.status_code == 200:
                response_ = response.json()

                if "status" in response_:
                    if response_["status"] == "error":
                        raise Exception(
                            f"IPC Transport failed, Server returns error: {response_}"
                        )
            else:
                raise IOError(
                    f"IPC Tranport failed, Server returns code: {response.status_code}"
                )

        except requests.exceptions.RequestException as e:
            raise e

        except Exception as e:
            raise e

    def write(self, name: str, t: torch.Tensor):
        """
        Processes a tensor, potentially combining it with others, writes the combined
        tensors' data into the pre-allocated shared memory block, and then sends
        the metadata to the remote host via an HTTP POST request.

        The tensor is sent layer by layer.

        The tensor is first converted to a contiguous CPU tensor before processing.

        Args:
            name (str): The name to associate with the tensor.
            t (torch.Tensor): The tensor to send.

        Raises:
            requests.exceptions.RequestException: If there's an error during the HTTP request.
            ValueError: If an unsupported IPC method is chosen or if the tensor is too large for SHM.
        """
        t = t.cpu()

        logging.info(
            f"tipc transporting tensor, name={name}, dtype={t.dtype}, shape={t.shape}, device={t.device}"
        )
        # Combine the current tensor with any pending tensors from the same layer
        named_tensors = self.tensor_bucket.combine(name, t)

        if len(named_tensors) > 0:
            encoded_bytes: int = 0
            encoded_metas: list[SharedMemIpcMeta] = []

            # Write the tensors to the shared memory block
            for nt in named_tensors:
                nt = preprocess(nt)
                m: SharedMemIpcMeta = self.cpu_ipc.build_tensor_meta(
                    name=nt.name, t=nt.tensor, offset=encoded_bytes, shm=self.shm
                )
                encoded_bytes += m.size_bytes
                encoded_metas.append(m)
            # Send the metadata to the server
            self._send(encoded_metas=encoded_metas)

    def flush(self):
        """
        Forces the TensorBucketBuilder to process and send any remaining
        tensors in its buffers, regardless of layer ID changes.
        """
        named_tensors = self.tensor_bucket.flush()

        if len(named_tensors) > 0:
            encoded_bytes: int = 0
            encoded_metas: list[SharedMemIpcMeta] = []

            for nt in named_tensors:
                nt = preprocess(nt)
                m: SharedMemIpcMeta = self.cpu_ipc.build_tensor_meta(
                    name=nt.name, t=nt.tensor, offset=encoded_bytes, shm=self.shm
                )
                encoded_bytes += m.size_bytes
                encoded_metas.append(m)
            self._send(encoded_metas=encoded_metas)

    def __del__(self):
        """
        Ensures the persistent shared memory block is closed and unlinked
        when the TensorTransportClient object is garbage collected.
        """
        if hasattr(self, "shm") and self.shm:
            try:
                self.shm.close()
                self.shm.unlink()  # Unlink the shared memory block from the system
                logging.info(
                    f"Persistent shared memory block '{self.shm_name}' closed and unlinked."
                )
            except FileNotFoundError:
                logging.info(
                    f"Warning: Persistent shared memory block '{self.shm_name}' already unlinked."
                )
            except Exception as e:
                logging.info(
                    f"Error during __del__ of shared memory block '{self.shm_name}': {e}"
                )
