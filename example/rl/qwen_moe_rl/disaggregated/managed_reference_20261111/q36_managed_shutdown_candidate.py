"""Keep the actual rLLM backend/manager alive on an uncertain managed failure."""
import asyncio
import functools
import hashlib
import importlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time
import traceback

BOOTSTRAP_SHA = '9dce1870a334d8c867eebac8ef7f430c60737c47597febcff5d14810be4df976'


def _bootstrap():
    root=Path(os.environ['Q36_CODE_DIR']).resolve(strict=True)
    name='production_nixl_bootstrap_candidate';path=root/(name+'.py')
    spec=importlib.machinery.PathFinder.find_spec(name,sys.path);loaded=sys.modules.get(name)
    if (path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=BOOTSTRAP_SHA or
        spec is None or not spec.origin or Path(spec.origin).resolve()!=path or
        loaded is not None and Path(getattr(loaded,'__file__','')).resolve()!=path):
        raise RuntimeError('managed NIXL shutdown bootstrap source mismatch')
    return importlib.import_module(name)


def _receipt(kind, detail):
    root=Path(os.environ['Q36_RUN_ROOT']).resolve(strict=True)
    path=root/(f'managed-{kind}-{os.getpid()}.json')
    with path.open('x') as out:
        json.dump({'pid':os.getpid(),'kind':kind,**detail},out,sort_keys=True)
        out.flush();os.fsync(out.fileno())
    print('Q36_MANAGED_'+kind.upper()+' '+str(path),flush=True)


def hold(backend, error, phase):
    # This stack retains backend, manager, refs, and the triggering exception.
    # It does not imply that any pending RDMA or coroutine has stopped.
    try:_receipt('hold',{'phase':phase,'error_type':type(error).__name__})
    except BaseException:traceback.print_exc()
    while True:time.sleep(30)


def install_outer(module):
    """Catch failures around actual VerlTaskRunner construction/cleanup.

    The exception retains its traceback and hence original stack locals,
    including any partially constructed trainer/backend and actor handles.
    """
    actor_class=getattr(module,'VerlTaskRunner',None)
    if actor_class is None or actor_class.__module__ != module.__name__:
        raise RuntimeError('unexpected unified verl task runner identity')
    # The pinned launcher uses @ray.remote.  Mutating ActorClass.run only
    # changes the local wrapper; Ray serializes metadata.modified_class.
    metadata=getattr(actor_class,'__ray_metadata__',None)
    cls=getattr(metadata,'modified_class',None) if metadata is not None else actor_class
    if not isinstance(cls,type) or cls.__module__ != module.__name__:
        raise RuntimeError('unexpected Ray task runner implementation identity')
    original=cls.run
    if getattr(original,'_q36_managed_outer',False):return
    methods=None
    if metadata is not None:
        if getattr(metadata,'last_export_cluster_and_job',None):
            raise RuntimeError('task runner already exported before managed hook')
        methods=getattr(getattr(metadata,'method_meta',None),'methods',None)
        if not isinstance(methods,dict) or methods.get('run') is not original:
            raise RuntimeError('unexpected Ray task runner method metadata')

    @functools.wraps(original)
    def run(self,*args,**kwargs):
        try:return original(self,*args,**kwargs)
        except BaseException as error:
            if os.environ.get('Q36_USE_MANAGED_NIXL') == '1':
                hold(self,error,'task_runner_outer')
            raise
    run._q36_managed_outer=True
    cls.run=run
    if methods is not None:
        methods['run']=run


def install_unified(module):
    cls=getattr(module,'UnifiedTrainer',None)
    if cls is None or cls.__module__ != module.__name__:
        raise RuntimeError('unexpected unified trainer identity')
    init=cls.__init__
    if not getattr(init,'_q36_managed_outer',False):
        @functools.wraps(init)
        def guarded_init(self,*args,**kwargs):
            try:return init(self,*args,**kwargs)
            except BaseException as error:
                if os.environ.get('Q36_USE_MANAGED_NIXL') == '1':
                    hold(self,error,'unified_initialization')
                raise
        guarded_init._q36_managed_outer=True
        cls.__init__=guarded_init
    shutdown=cls.shutdown
    if not getattr(shutdown,'_q36_managed_outer',False):
        @functools.wraps(shutdown)
        def guarded_shutdown(self,*args,**kwargs):
            active_error=sys.exc_info()[1]
            if os.environ.get('Q36_USE_MANAGED_NIXL') == '1' and active_error is not None:
                return hold(self,active_error,'unified_shutdown_after_failure')
            try:return shutdown(self,*args,**kwargs)
            except BaseException as error:
                if os.environ.get('Q36_USE_MANAGED_NIXL') == '1':
                    hold(self,error,'unified_shutdown')
                raise
        guarded_shutdown._q36_managed_outer=True
        cls.shutdown=guarded_shutdown


def install(backend_class):
    original=backend_class.shutdown
    if getattr(original,'_q36_managed_shutdown',False):return

    @functools.wraps(original)
    def shutdown(self):
        if os.environ.get('Q36_USE_MANAGED_NIXL')!='1':return original(self)
        error=sys.exc_info()[1]
        manager=getattr(self,'checkpoint_manager',None)
        version=getattr(manager,'_q36_version_state',{}) if manager else {}
        gate=getattr(manager,'_q36_transaction_gate',None) if manager else None
        if error is not None or manager is None or version.get('poisoned') or getattr(gate,'poisoned',False):
            return hold(self,error or RuntimeError('manager unavailable or poisoned'),'backend_shutdown')
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return hold(self,RuntimeError('synchronous shutdown is inside a running event loop'),'backend_shutdown')
        try:
            asyncio.run(_bootstrap().shutdown_managed_contexts(manager))
            manager.close()
            _receipt('shutdown',{'managed_contexts_closed':True,'manager_closed':True})
        except BaseException as exc:
            traceback.print_exc()
            return hold(self,exc,'managed_context_shutdown')
        return original(self)

    shutdown._q36_managed_shutdown=True
    backend_class.shutdown=shutdown
