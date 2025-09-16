import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image
from torch import nn

from rtp_llm.multimodal.multimodal_common import (
    AudioEmbeddingInterface,
    ImageEmbeddingInterface,
    ImageTransform,
)

try:
    import tensorrt as trt
except ImportError as e:
    pass


def torch_dtype_from_trt(dtype):
    # TODO(xyz): support quantization such as int8, int4
    if dtype == trt.bfloat16:
        return torch.bfloat16
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f"unsupported tensorrt data type {dtype}")


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError(f"unsupported tensorrt device type {device}")


def torch_type_to_path(dtype: torch.dtype):
    if dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float32:
        return "fp32"
    else:
        raise TypeError(f"unknown torch data type {dtype}")


# TODO(xyz): support handle video and audio case, not only image
class MultiModalTRTEngine(nn.Module, ImageEmbeddingInterface, AudioEmbeddingInterface):
    def __init__(
        self,
        model_name: str,
        image_size: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
        vit_config,
    ):
        super(MultiModalTRTEngine, self).__init__()
        self.image_size = image_size
        self.image_transform = ImageTransform(self.image_size)
        self.device = device
        self.dtype = dtype
        self.max_batch_size = 1
        self.cur_batch_size = 1
        self.input_names = ["input"]
        self.output_names = ["output"]
        output_dir = MultiModalTRTEngine.cache_path(
            model_name, self.dtype, vit_config.trt_cache_path
        )
        self.onnx_file_path = os.path.join(output_dir, "multimodal.onnx")
        self.engine_file_path = os.path.join(output_dir, "multimodal.trt")
        self.engine = None
        self.lock = threading.Lock()

    @staticmethod
    def trt_engine_cached(model_name: str, dtype: torch.dtype, trt_cache_path) -> bool:
        return MultiModalTRTEngine.completion_file_path(
            model_name, dtype, trt_cache_path
        ).exists()

    @staticmethod
    def cache_path(model_name: str, dtype: torch.dtype, trt_cache_path) -> str:
        """Get cache path for TRT engine.

        Args:
            model_name: Model name.
            dtype: Torch dtype."""
        if trt_cache_path is None:
            trt_cache_path = os.path.join(os.getcwd(), "trt_cache")
        return os.path.join(
            trt_cache_path,
            f"{model_name}_{torch_type_to_path(dtype)}",
        )

    @staticmethod
    def completion_file_path(
        model_name: str, dtype: torch.dtype, trt_cache_path
    ) -> Path:
        return Path(
            os.path.join(
                MultiModalTRTEngine.cache_path(model_name, dtype, trt_cache_path),
                "trt_engine.done",
            )
        )

    @staticmethod
    def _export_onnx_helper(
        network: torch.nn.Module,
        dummy_input: torch.Tensor,
        onnx_file_path: str,
        input_names: List[str],
        output_names: List[str],
    ):
        with torch.inference_mode():
            torch.onnx.export(
                network,
                dummy_input,
                onnx_file_path,
                opset_version=17,
                input_names=input_names,
                output_names=output_names,
            )

    async def _export_onnx_async_internal(
        self, network: torch.nn.Module, dummy_input: torch.Tensor
    ):
        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor() as executor:
            future = loop.run_in_executor(
                executor,
                self._export_onnx_helper,
                network,
                dummy_input,
                self.onnx_file_path,
                self.input_names,
                self.output_names,
            )
            await asyncio.wait([future])

    def export_onnx(self, network: torch.nn.Module, tp_size: int):
        logging.info("Start exporting torch to ONNX model")
        dummy_input = (
            torch.randn(self.cur_batch_size, 3, self.image_size, self.image_size)
            .to(self.device)
            .to(self.dtype)
        )

        if tp_size <= 1:
            self._export_onnx_helper(
                network,
                dummy_input,
                self.onnx_file_path,
                self.input_names,
                self.output_names,
            )
        else:
            # here we need to export onnx in another new thread, otherwise it will
            # block heartbeat of gang_server when TP > 1
            asyncio.run(self._export_onnx_async_internal(network, dummy_input))

        logging.info("Finish exporting ONNX model")

    def generate_trt_engine(self):
        logging.info("Start generating TRT engine!")

        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()

        if self.dtype == torch.float32:
            pass
        elif self.dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)
        else:
            raise ValueError(f"unsupported torch data type {self.dtype}")

        profile = builder.create_optimization_profile()

        # parse onnx model
        parser = trt.OnnxParser(network, logger)
        with open(self.onnx_file_path, "rb") as model:
            if not parser.parse(model.read(), "/".join(self.onnx_file_path.split("/"))):
                logging.info(f"Failed parsing {self.onnx_file_path}")
                for error in range(parser.num_errors):
                    logging.info(parser.get_error(error))
            logging.info(f"Succeeded parsing {self.onnx_file_path}")

        nBS = -1
        nMinBS = 1
        nOptBS = max(1, int(self.max_batch_size / 2))
        nMaxBS = self.max_batch_size
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, self.image_size, self.image_size]
        profile.set_shape(
            inputT.name,
            [nMinBS, 3, self.image_size, self.image_size],
            [nOptBS, 3, self.image_size, self.image_size],
            [nMaxBS, 3, self.image_size, self.image_size],
        )

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        if engineString is None:
            logging.info(f"Failed building {self.engine_file_path}")
        else:
            logging.info(f"Succeeded building {self.engine_file_path} in {t1 - t0} s")
        with open(self.engine_file_path, "wb") as f:
            f.write(engineString)

    def load_trt_engine(self):
        logging.info("Start loading TRT engine!")
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_file_path, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        logging.info("Start initializing TRT engine!")
        if self.engine is not None:
            self.bindings = [None] * (len(self.input_names) + len(self.output_names))
            self.context = self.engine.create_execution_context()

            # fix input shape
            input_shape = self.engine.get_tensor_shape(self.input_names[0])
            input_shape[0] = self.cur_batch_size
            self.context.set_input_shape(self.input_names[0], tuple(input_shape))

            self.output_dtype = torch_dtype_from_trt(
                self.engine.get_tensor_dtype(self.output_names[0])
            )
            self.output_device = torch_device_from_trt(
                self.engine.get_tensor_location(self.output_names[0])
            )
        else:
            raise ValueError(f"Failed loading {self.engine_file_path}")

    def forward(self, *inputs):
        # use lock to avoid concurrency conflict issue
        with self.lock:

            input = inputs[0]
            batch_size = input.shape[0]

            output_shape = tuple(self.engine.get_tensor_shape(self.output_names[0]))

            # update input shape
            if batch_size != self.cur_batch_size:
                self.cur_batch_size = batch_size
                input_shape = self.engine.get_tensor_shape(self.input_names[0])
                input_shape[0] = self.cur_batch_size
                self.context.set_input_shape(self.input_names[0], tuple(input_shape))
                output_shape[0] = self.cur_batch_size

            output = torch.empty(
                size=output_shape, dtype=self.output_dtype, device=self.output_device
            )

            # ensure the input tensor passed into trt engine is continous in memory,
            # if not, change the input tensor to be continous
            self.context.set_tensor_address(self.input_names[0], input.data_ptr())
            self.context.set_tensor_address(self.output_names[0], output.data_ptr())

            # execute the engine synchronously
            self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        return output

    def encode(self, images: List[Image.Image], device: Union[str, torch.device]):
        images = self.image_transform.encode(images, device, self.dtype)
        return self(images)

    def image_embedding(
        self, images: List[Image.Image], device: Union[str, torch.device]
    ) -> torch.Tensor:
        if len(images) != 0:
            images = self.encode(images, device)
        return images.to(device=device)

    def audio_embedding(
        self, audio: Tuple[torch.Tensor, int], device: Union[str, torch.device]
    ) -> torch.Tensor:
        raise NotImplementedError()
