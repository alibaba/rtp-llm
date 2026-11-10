"""Exercise the actual stdlib-only runtime-env function without Ray or CUDA."""
import ast
from pathlib import Path
import unittest


class RuntimeEnvironmentTest(unittest.TestCase):
    def test_toolchain_environment_forwarded_and_gpu_mask_excluded(self):
        source=Path(__file__).with_name('run_standalone_q36_candidate_v3.py').read_text()
        tree=ast.parse(source)
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='propagate_runtime_env')
        module=ast.Module(body=[function],type_ignores=[])
        namespace={};exec(compile(module,'<runtime-env>','exec'),namespace)
        source_env={'TORCH_EXTENSIONS_DIR':'/run/extensions','CUDA_HOME':'/cuda-13.2',
                    'TORCH_CUDA_ARCH_LIST':'10.3','MAX_JOBS':'4','PATH':'/cuda/bin:/usr/bin',
                    'UCX_TLS':'rc,self,sm,cuda_copy,cuda_ipc','Q36_USE_RPC_OVERLAY':'1','Q397_NCCL_SO':'/pinned/libnccl.so.2',
                    'CUDA_VISIBLE_DEVICES':'0','UNRELATED_SECRET':'not-propagated'}
        runtime={'worker_process_setup_hook':'standalone_bootstrap_rpc_v6.worker_setup',
                 'env_vars':{'CUDA_VISIBLE_DEVICES':'','existing':'kept'}}
        got=namespace['propagate_runtime_env'](runtime,source_env)
        self.assertEqual(got['worker_process_setup_hook'],'standalone_bootstrap_rpc_v6.worker_setup')
        for key in ('TORCH_EXTENSIONS_DIR','CUDA_HOME','TORCH_CUDA_ARCH_LIST','MAX_JOBS','PATH','UCX_TLS','Q36_USE_RPC_OVERLAY','Q397_NCCL_SO'):
            self.assertEqual(got['env_vars'][key],source_env[key])
        self.assertNotIn('CUDA_VISIBLE_DEVICES',got['env_vars'])
        self.assertNotIn('UNRELATED_SECRET',got['env_vars'])
        self.assertEqual(got['env_vars']['existing'],'kept')


if __name__=='__main__':unittest.main()
