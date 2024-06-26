import gc
import logging
import os
import time
from pathlib import Path
from typing import List, Union

import torch
from torch import nn

from maga_transformer.models.multimodel.multimodel_common import ImageTransform

try:
    import tensorrt as trt
except ImportError as e:
    pass


def torch_dtype_from_trt(dtype: trt.DataType):
    # TODO(xyz): support quantization such as int8, int4
    if dtype == trt.bfloat16:
        return torch.bfloat16
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f"unsupported tensorrt data type {dtype}")


def torch_device_from_trt(device: trt.TensorLocation):
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


class MultiModelTRTEngine(nn.Module):
    def __init__(
        self,
        model_name: str,
        network: nn.Module,
        image_size: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        super(MultiModelTRTEngine, self).__init__()
        self.image_size = image_size
        self.image_transform = ImageTransform(self.image_size)
        self.device = device
        self.dtype = dtype
        self.max_batch_size = 1
        self.cur_batch_size = 1
        self.input_names = ["input"]
        self.output_names = ["output"]

        output_dir = MultiModelTRTEngine.cache_path(model_name, self.dtype)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        onnx_file_path = os.path.join(output_dir, "multimodel.onnx")
        engine_file_path = os.path.join(output_dir, "multimodel.trt")

        if MultiModelTRTEngine.trt_engine_cached(model_name, self.dtype):
            # self.export_onnx(network, onnx_file_path, device, dtype)

            # TODO(xyz): gc torch network, free memory for generating trt engine
            del network
            gc.collect()
            torch.cuda.empty_cache()

            self.generate_trt_engine(onnx_file_path, engine_file_path, dtype)

            # finish generating trt engine, create a completion file
            MultiModelTRTEngine.completion_file_path(model_name, dtype).touch()

        self.engine = self.load_trt_engine(engine_file_path)

        if self.engine is not None:

            self.bindings = [None] * (len(self.input_names) + len(self.output_names))
            self.context = self.engine.create_execution_context()

            # fix input shape
            input_shape = self.engine.get_tensor_shape(self.input_names[0])
            input_shape[0] = self.cur_batch_size
            self.context.set_input_shape(self.input_names[0], tuple(input_shape))

            self.output_shape = tuple(
                self.engine.get_tensor_shape(self.output_names[0])
            )
            self.output_dtype = torch_dtype_from_trt(
                self.engine.get_tensor_dtype(self.output_names[0])
            )
            self.output_device = torch_device_from_trt(
                self.engine.get_tensor_location(self.output_names[0])
            )

    @staticmethod
    def trt_engine_cached(model_name: str, dtype: torch.dtype) -> bool:
        return not MultiModelTRTEngine.completion_file_path(model_name, dtype).exists()

    @staticmethod
    def cache_path(model_name: str, dtype: torch.dtype) -> str:
        return os.path.join(
            os.environ.get("TRT_CACHE_PATH", os.path.join(os.getcwd(), "trt_cache")),
            f"{model_name}_{torch_type_to_path(dtype)}",
        )

    @staticmethod
    def completion_file_path(model_name: str, dtype: torch.dtype) -> Path:
        return Path(
            os.path.join(
                MultiModelTRTEngine.cache_path(model_name, dtype), "vit_trt.done"
            )
        )

    def export_onnx(
        self,
        network: torch.nn.Module,
        onnx_file_path: str,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        logging.info("Start exporting torch to ONNX model")
        image = (
            torch.randn(self.batch_size, 3, self.image_size, self.image_size)
            .to(device)
            .to(dtype)
        )
        with torch.inference_mode():
            torch.onnx.export(
                network,
                image,
                onnx_file_path,
                opset_version=17,
                input_names=self.input_names,
                output_names=self.output_names,
            )
        logging.info("Finish exporting ONNX model")

    def generate_trt_engine(self, onnx_path: str, engine_path: str, dtype: torch.dtype):
        logging.info("Start generating TRT engine!")

        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()

        if dtype == torch.float32:
            pass
        elif dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)
        else:
            raise ValueError(f"unsupported torch data type {dtype}")

        profile = builder.create_optimization_profile()

        # parse onnx model
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnx_path.split("/"))):
                logging.info(f"Failed parsing {onnx_path}")
                for error in range(parser.num_errors):
                    logging.info(parser.get_error(error))
            logging.info(f"Succeeded parsing {onnx_path}")

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
            logging.info(f"Failed building {engine_path}")
        else:
            logging.info(f"Succeeded building {engine_path} in {t1 - t0} s")
        with open(engine_path, "wb") as f:
            f.write(engineString)

    def load_trt_engine(self, filepath: str):
        logging.info("Start loading TRT engine!")
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            logging.info("Finish loading TRT engine!")
            return engine

    def forward(self, *inputs):
        input = inputs[0]
        batch_size = input.shape[0]

        # update input shape
        if batch_size != self.cur_batch_size:
            input_shape = self.engine.get_tensor_shape(self.input_names[0])
            input_shape[0] = self.cur_batch_size
            self.context.set_input_shape(self.input_names[0], tuple(input_shape))
            self.output_shape = tuple(
                self.engine.get_tensor_shape(self.output_names[0])
            )
            self.output_shape[0] = self.cur_batch_size

        output = torch.empty(
            size=self.output_shape, dtype=self.output_dtype, device=self.output_device
        )

        # ensure the input tensor passed into trt engine is continous in memory,
        # if not, change the input tensor to be continous
        self.bindings[0] = input.data_ptr()
        self.bindings[1] = output.data_ptr()

        # execute the engine synchronously
        self.context.execute_v2(self.bindings)

        return output

    def encode(self, image_paths: List[str], device: Union[str, torch.device]):
        images = self.image_transform.encode(image_paths, self.device, self.dtype)
        return self(images)

    def image_embedding(
        self, images: List[str], device: Union[str, torch.device]
    ) -> torch.Tensor:
        if len(images) != 0:
            images = self.encode(images, device)
            assert images.shape[0] == len(images)
        return images.to(device=device)
