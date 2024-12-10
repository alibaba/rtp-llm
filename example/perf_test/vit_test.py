from maga_transformer.model_factory import ModelFactory, ModelConfig
from maga_transformer.model_factory_register import _model_factory
from maga_transformer.utils.multimodal_util import (MMUrlType,
                                                    MMPreprocessConfig)
import os
import argparse
import logging
from PIL import Image
from random import randint
import time
import torch
import tempfile

class TestCase:
    def __init__(self, model_type, ckpt_path, image_sizes):
        self.model_type = model_type
        self.ckpt_path = ckpt_path
        self.image_size = image_sizes
        self.images = []
        for size in self.image_size:
            image = Image.new('RGB', (size, size))
            pixels = image.load()
            for i in range(size):
                for j in range(size):
                    pixels[i, j] = (randint(0, 255), randint(0, 255), randint(0, 255))
            self.images.append(image)

    def get_args(self):
        return {
            "model_args": {
                "model_type": self.model_type,
                "ckpt_path": self.ckpt_path
            },
            "test_args": {
                "image_size": self.image_size
            }
        }

def create_model(test_config: TestCase):
    model_config = ModelConfig(test_config.model_type, test_config.ckpt_path, test_config.ckpt_path)
    global _model_factory
    if model_config.model_type not in _model_factory:
        raise Exception(f"model type {model_config.model_type} not registered!")
    model_cls = _model_factory[model_config.model_type]
    config = model_cls.create_config(model_config)
    config.layer_num = 1
    model = model_cls.from_config(config)
    return model

def run_test(testcase: TestCase):
    model = create_model(testcase)
    times = []
    for image in testcase.images:
        with tempfile.NamedTemporaryFile(suffix='.png') as temp_file:
            image.save(temp_file.name)
            begin_time = time.time()
            model.mm_part.mm_embedding(temp_file.name, MMUrlType.IMAGE, configs=MMPreprocessConfig())
            end_time = time.time()
        times.append(end_time - begin_time)
    return times

if __name__ == '__main__':
    os.environ['LOAD_CKPT_NUM_PROCESS'] = '0'
    logging.basicConfig(
        level="INFO",
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='vit perf')

    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--image_size', type=str, required=True)

    args, _ = parser.parse_known_args()
    t_case = TestCase(args.model_type, args.ckpt_path, [int(x) for x in args.image_size.split(',')])

    device_name = os.environ.get('DEVICE_NAME')
    if device_name is None:
        device_name = torch.cuda.get_device_name(0)
    
    logging.info(f"t_case.get_args() is {t_case.get_args()}")
    res = run_test(t_case)
    logging.info(f"test result is {res}")
