#!/usr/bin/env python3
"""Build current RTP sources against the installed ARM Torch 2.13/CUDA 13 SDK.

Run inside the authorized development container. This writes only the requested
fresh output directory plus Bazel output symlinks; it does not install packages
or start CUDA/model processes. No credentials or model files are needed.
"""
from __future__ import annotations
import argparse,hashlib,json,os,platform,shutil,subprocess,sys,sysconfig,time
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bazelisk',default='bazelisk')
    parser.add_argument('--jobs',type=int,default=32)
    parser.add_argument('--ram-mb',type=int,default=192000)
    args=parser.parse_args()
    source=args.source.resolve(strict=True);out=args.output.absolute();out.mkdir(parents=True,exist_ok=False)
    assert platform.machine()=='aarch64' and sys.version_info[:2]==(3,10)
    assert args.jobs>0 and args.ram_mb>0
    os.environ.update(CUDA_VISIBLE_DEVICES='',USE_BAZEL_VERSION='6.4.0')
    import torch
    assert str(torch.__version__)=='2.13.0+cu130' and torch._C._GLIBCXX_USE_CXX11_ABI
    assert not torch.cuda.is_initialized()
    torchroot=Path(torch.__file__).parent
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
    def run(label,argv,timeout=14400):
        with (out/(label+'.log')).open('x') as log:
            child=subprocess.Popen(argv,cwd=source,stdout=log,stderr=subprocess.STDOUT,env=os.environ)
            identity={'pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19]}
            save(label+'-process.json',dict(identity,command=argv))
            try:rc=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:rc=child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();rc=child.wait()
                save(label+'-result.json',dict(actual_wait_returncode=rc,timed_out=True));raise
        save(label+'-result.json',dict(actual_wait_returncode=rc,finished_unix=time.time()))
        if rc:raise RuntimeError(f'{label} actual_wait_returncode={rc}')
    override=out/'torch-system-repository';override.mkdir()
    # Keep Bazel's historical repository label, override its actual contents.
    repository_name='torch_2.11_py310_cuda-aarch64'
    (override/'WORKSPACE').write_text(f'workspace(name="{repository_name}")\n')
    (override/'torch').symlink_to(torchroot,target_is_directory=True)
    filtered=override/'torch_system_include';filtered.mkdir()
    for entry in (torchroot/'include').iterdir():
        if entry.name!='google':(filtered/entry.name).symlink_to(entry,target_is_directory=entry.is_dir())
    build=(source/'BUILD.pytorch').read_text()
    for old,new in [('strip_include_prefix = "torch/include",','includes = ["torch/include"],'),('strip_include_prefix = "torch/include/torch/csrc/api/include",','includes = ["torch/include/torch/csrc/api/include"],')]:
        assert build.count(old)==1,(old,build.count(old));build=build.replace(old,new)
    (override/'BUILD.bazel').write_text(build.replace('"torch/include','"torch_system_include'))
    probe=out/'torch_header_probe.cc';probe.write_text('#include <torch/extension.h>\nint main(){return 0;}\n')
    run('cxx17-header-probe',['g++','-std=c++17','-Werror','-fsyntax-only','-D_GLIBCXX_USE_CXX11_ABI=1','-isystem',str(filtered),'-isystem',str(filtered/'torch/csrc/api/include'),'-I'+sysconfig.get_paths()['include'],str(probe)],timeout=240)
    overrides=[f'--override_repository={repository_name}={override}'];dependencies=[]
    for short in ['nvshmem','cusparselt','nccl','cudnn']:
        name='pip_cuda13_arm_torch_nvidia_'+short+'_cu13';target=out/name;target.mkdir()
        sdk=torchroot.parent/'nvidia'/short;assert sdk.is_dir(),str(sdk)
        (target/'site-packages/nvidia').mkdir(parents=True)
        (target/'site-packages/nvidia'/short).symlink_to(sdk,target_is_directory=True)
        (target/'WORKSPACE').write_text('workspace(name='+json.dumps(name)+')\n')
        (target/'BUILD.bazel').write_text('exports_files(glob(["site-packages/**"]))\n')
        overrides.append('--override_repository='+name+'='+str(target));dependencies.append({'repository':name,'installed_sdk':str(sdk)})
    command=[args.bazelisk,'--batch','--output_user_root='+str(out/'bazel-user'),'--host_jvm_args=-Xmx4g','build','--config=cuda13_arm','--jobs='+str(args.jobs),'--local_ram_resources='+str(args.ram_mb),'--experimental_remote_downloader_local_fallback','--action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0','--host_action_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0','--repo_env=TF_CUDA_COMPUTE_CAPABILITIES=10.0','--action_env=NCCL_INSTALL_PATH='+str(torchroot.parent/'nvidia/nccl'),'--action_env=NCCL_HDR_PATH='+str(torchroot.parent/'nvidia/nccl/include'),'--action_env=CUDNN_INSTALL_PATH='+str(torchroot.parent/'nvidia/cudnn'),*overrides,'//:th_transformer','//:th_transformer_config','//:th_grammar_tokenizer_info','//:rtp_compute_ops']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    changed=subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=source,text=True).splitlines()
    save('BUILD-INPUT.json',{'source_revision':revision,'changed_tracked_files':{name:digest(source/name) for name in changed if (source/name).is_file()},'script_sha256':digest(__file__),'torch':str(torch.__version__),'CXX11_ABI':True,'compute_capability':'10.0','dependencies':dependencies,'command':command,'GPU_runtime_work':False})
    started=time.time();run('build',command)
    names=['libth_transformer.so','libth_transformer_config.so','libth_grammar_tokenizer_info.so','librtp_compute_ops.so']
    artifacts=[]
    for name in names:
        path=(source/'bazel-bin'/name).resolve(strict=True)
        needed=subprocess.check_output(['readelf','-d',str(path)],text=True)
        artifacts.append({'name':name,'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'readelf_dynamic':needed})
    save('RESULT.json',{'actual_build_wait_returncode':0,'artifacts':artifacts,'build_elapsed_seconds':time.time()-started,'GPU_runtime_work':False,'model_started':False,'full_RL_acceptance':False})
    assert not torch.cuda.is_initialized()
    print(json.dumps({'result':str(out/'RESULT.json'),'actual_build_wait_returncode':0},indent=2))


if __name__=='__main__':main()
