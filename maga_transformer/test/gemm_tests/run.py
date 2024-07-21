import os
import sys
import copy
import logging
import subprocess
from maga_transformer.test.utils.device_resource import DeviceResource

if __name__ == '__main__':
    args = sys.argv[1: ]
    with DeviceResource(1) as device_resource:
        gpu_ids = device_resource.gpu_ids
        process_env = copy.deepcopy(os.environ)
        process_env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)
        os.makedirs(f'{os.getcwd()}/logs/', exist_ok=True)
        log_file = f'{os.getcwd()}/logs/process.log'
        logging.info("write log in: " + log_file)
        with open(log_file, 'w') as f:
            p = subprocess.Popen(["/opt/conda310/bin/python", "-m", "maga_transformer.test.gemm_tests.gemm_test"] + args,
                    env=process_env, stdout=f, stderr=f)
            returncode = p.wait()
            if returncode != 0:
                raise Exception("subprocess run failed with return code != 0, please check file for detai: " + log_file)