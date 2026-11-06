"""Execute the real launcher prefix with synthetic files and a blocked main."""
from pathlib import Path
import argparse,ctypes.util,json,os,subprocess,sys,tempfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher',type=Path,required=True)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='q38-v8-path-isolation-') as folder:
        root=Path(folder);model=root/'mounted-model';model.mkdir();(model/'config.json').write_text('{}')
        manifest=root/'canonical.json';manifest.write_text('{}')
        mapping=root/'map.json';mapping.write_text('{}')
        capture=root/'capture.json';shim=root/'python-shim'
        shim.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
            +f'if sys.argv[1:2]==["-"]:os.execv({sys.executable!r},[{sys.executable!r},*sys.argv[1:]])\n'
            +'assert sys.argv[1:3]==["-m","qwen38_world72_entrypoint"], "Native main is blocked"\n'
            +'keys=["TOKENIZER_PATH","CHECKPOINT_PATH","PTUNING_PATH","EXTRA_DATA_PATH","LOCAL_EXTRA_DATA_PATH","SP_CHECKPOINT_PATH","SP_TYPE","SP_MODEL_TYPE","LORA_INFO","JSON_MODEL_OVERRIDE_ARGS","RTP_ARM_BF16_ROPE_BACKEND","QUANTIZATION","PYTORCH_CUDA_ALLOC_CONF"]\n'
            +f'open({str(capture)!r},"w").write(json.dumps({{"argv":sys.argv[1:],"environment":{{k:os.environ.get(k) for k in keys}}}}))\n')
        shim.chmod(0o700)
        environment=dict(os.environ)
        environment.update(Q38_DEPLOY_ROOT=str(root),RTP_SOURCE=str(root),RTP_PYTHON=str(shim),
            RTP_MODEL_PATH=str(model),RTP_WEIGHT_MANIFEST_DIR=str(root/'weights'),
            RTP_RL_NATIVE_LIBRARY_SHA256='a'*64,RTP_RL_BOOTSTRAP_POLICY_MANIFEST=str(manifest),
            RTP_PHY2LOG_PATH=str(mapping),RLLM_Q38_NCCL_LIBRARY_PATH=ctypes.util.find_library('c') or 'CPU-fixture-not-NCCL',RLLM_Q38_NCCL_LIBRARY_SHA256='b'*64,
            RTP_LLM_RL_RUN_ID='cpu-fixture-no-native',RTP_WORLD_RANK='0',
            GANG_CONFIG_STRING=';'.join(f'name:fixture{i},ip:127.0.0.1,port:18000' for i in range(18)),
            CUDA_VISIBLE_DEVICES=','.join(f'GPU-00000000-0000-0000-0000-{i:012x}' for i in range(4)),
            RLLM_RTP_ACCEPTANCE_DIR=str(root/'audit'),TOKENIZER_PATH='oss://dummy-bucket/foreign?OSS_ACCESS_KEY_ID=DUMMY',
            CHECKPOINT_PATH='oss://dummy-bucket/foreign',PTUNING_PATH='oss://dummy-bucket/ptuning',
            EXTRA_DATA_PATH='oss://dummy-bucket/extra',LOCAL_EXTRA_DATA_PATH='/foreign/extra',
            SP_CHECKPOINT_PATH='oss://dummy-bucket/draft',SP_TYPE='mtp',SP_MODEL_TYPE='foreign',
            LORA_INFO='{"foreign":"oss://dummy-bucket/lora"}',JSON_MODEL_OVERRIDE_ARGS='{"foreign":true}',
            QUANTIZATION='FP8',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        # No preload needed for this standard-library, CPU-only fixture.
        environment.pop('LD_PRELOAD',None)
        child=subprocess.Popen(['/bin/bash',str(args.launcher.absolute())],env=environment,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        output,_=child.communicate();actual=child.wait()
        assert actual==0,output
        result=json.loads(capture.read_text());env=result['environment'];argv=result['argv']
        assert env['TOKENIZER_PATH']==env['CHECKPOINT_PATH']==str(model)
        for key in ('PTUNING_PATH','EXTRA_DATA_PATH','LOCAL_EXTRA_DATA_PATH','SP_CHECKPOINT_PATH','SP_TYPE','SP_MODEL_TYPE'):
            assert env[key]=='',key
        assert env['LORA_INFO']==env['JSON_MODEL_OVERRIDE_ARGS']=='{}'
        assert env['QUANTIZATION'] is None and env['PYTORCH_CUDA_ALLOC_CONF'] is None
        assert env['RTP_ARM_BF16_ROPE_BACKEND']=='torch_bf16_v1'
        assert argv[argv.index('--checkpoint_path')+1]==argv[argv.index('--tokenizer_path')+1]==str(model)
        for option in ('--ptuning_path','--extra_data_path','--local_extra_data_path','--sp_checkpoint_path','--sp_type','--sp_model_type'):
            assert argv[argv.index(option)+1]=='',option
        assert 'oss://' not in json.dumps(result) and 'DUMMY' not in json.dumps(result)
        print(json.dumps(dict(passed=True,actual_child_wait_exit_code=actual,
            actual_launcher_stdin_prefix_executed=True,synthetic_files_and_allocation=True,
            poisoned_foreign_paths_removed=True,explicit_local_tokenizer=True,
            final_native_main_blocked=True,real_RTP_parser_executed=False,
            Torch_Ray_GPU_or_model_payload_executed=False)))

if __name__=='__main__':main()
