import os
import logging
from pathlib import Path
import time
from typing import List
from maga_transformer.models.qwen_vl_vit import Preprocess

import torch
try:
    import tensorrt as trt
except ImportError as e:
    pass

def torch_dtype_from_trt(dtype):
   if dtype == trt.int8:
       return torch.int8
   elif dtype == trt.bool:
       return torch.bool
   elif dtype == trt.int32:
       return torch.int32
   elif dtype == trt.float16:
       return torch.float16
   elif dtype == trt.float32:
       return torch.float32
   else:
       raise TypeError("%s is not supported by torch" % dtype)
   
def torch_device_from_trt(device):
   if device == trt.TensorLocation.DEVICE:
       return torch.device("cuda")
   elif device == trt.TensorLocation.HOST:
       return torch.device("cpu")
   else:
       return TypeError("%s is not supported by torch" % device)


class VITEngine(torch.nn.Module):
    @staticmethod
    def should_generate_engine():
        return not VITEngine.get_check_done_file().exists()
    
    @staticmethod
    def get_engine_filepath():
        return os.environ.get('QWEN_VL_VIT_TRT_ONNX_EXPORT_PATH', os.path.join(os.getcwd(), "qwen_vl_onnx"))
    
    @staticmethod
    def get_check_done_file() -> Path:
        return Path(os.path.join(VITEngine.get_engine_filepath(), 'vit_trt.done'))
    
    def __init__(self, vit, image_size):
        super(VITEngine, self).__init__()
        self.image_size = image_size
        self.image_pre_obj = Preprocess(self.image_size)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        
        output_dir = VITEngine.get_engine_filepath()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        onnx_file_path = os.path.join(output_dir, "vit.onnx")
        engine_file_path = os.path.join(output_dir, "vit.trt")
        
        if VITEngine.should_generate_engine():
            self.export_onnx(vit, onnx_file_path)
            self.generate_trt_engine(onnx_file_path, engine_file_path)
            VITEngine.get_check_done_file().touch()

        self.engine = self.load_trt_engine(engine_file_path)
        
        if self.engine is not None:
            self.input_names = ["input"]
            self.output_names = ["output"]
            self.bindings = [None] * (len(self.input_names) + len(self.output_names))
            self.outputs = [None] * len(self.output_names)
            
            self.context = self.engine.create_execution_context()
            self.output_idx = self.engine.get_binding_index(self.output_names[0])
            self.output_dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(self.output_idx))
            self.output_shape = tuple(self.engine.get_binding_shape(self.output_idx))
            self.output_device = torch_device_from_trt(self.engine.get_location(self.output_idx))
            self.input_idx = self.engine.get_binding_index(self.input_names[0])

    def export_onnx(self, vit, onnx_file_path):
        logging.info("Start converting ONNX model!")
        image = torch.randn(1, 3, self.image_size, self.image_size).to(self.device)
        torch.onnx.export(
            vit,
            image.to('cuda'),
            onnx_file_path,
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}}
        )

    def generate_trt_engine(self,
                            onnxFile,
                            planFile,
                            minBS=1,
                            optBS=2,
                            maxBS=4):
        logging.info("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)

        with open(onnxFile, 'rb') as model:
            if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
                logging.info("Failed parsing %s" % onnxFile)
                for error in range(parser.num_errors):
                    logging.info(parser.get_error(error))
            logging.info("Succeeded parsing %s" % onnxFile)

        nBS = -1
        nMinBS = minBS
        nOptBS = optBS
        nMaxBS = maxBS
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, self.image_size, self.image_size]
        profile.set_shape(inputT.name,
                          [nMinBS, 3, self.image_size, self.image_size],
                          [nOptBS, 3, self.image_size, self.image_size],
                          [nMaxBS, 3, self.image_size, self.image_size])

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        if engineString == None:
            logging.info("Failed building %s" % planFile)
        else:
            logging.info("Succeeded building %s in %d s" % (planFile, t1 - t0))
        with open(planFile, 'wb') as f:
            f.write(engineString)
    
    def load_trt_engine(self, filepath: str):
        logging.info("Start loading TRT engine!")
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(filepath, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            logging.info("Finish loading TRT engine!")
            return engine

    def forward(self, *inputs):
        batch_size = inputs[0].shape[0]
        
        shape = (batch_size, ) + self.output_shape[1:]
        output = torch.empty(size=shape, dtype=self.output_dtype, device=self.output_device)
        self.outputs[0] = output
        self.bindings[self.output_idx] = output.data_ptr()
        
        self.context.set_binding_shape(self.input_idx, tuple(inputs[0].shape))
        self.bindings[self.input_idx] = inputs[0].contiguous().data_ptr()

        self.context.execute_async(
            batch_size, self.bindings, torch.cuda.current_stream().cuda_stream
        )

        outputs = tuple(self.outputs)[0]
        return outputs

    def encode(self, image_paths: List[str]):
        images = self.image_pre_obj.encode(image_paths).to(device=self.device)
        return self(images)
    
    def image_embedding(self, images: List[str], device) -> torch.Tensor:
        if len(images) != 0:
            images = self.encode(images)
            assert images.shape[0] == len(images)
        return images.to(device=device)
