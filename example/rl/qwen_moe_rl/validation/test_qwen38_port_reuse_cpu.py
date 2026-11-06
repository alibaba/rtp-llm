"""Real Linux TCP restart checks using only this test's short-lived children.

The production probe function is compiled directly from its AST, so this test
does not import Torch/Ray/native libraries. Sockets and kernel TIME_WAIT states
are real, with ephemeral localhost ports and no pre-existing service touched.
"""
from __future__ import annotations
import argparse,ast,errno,hashlib,json,os,socket,subprocess,sys,tempfile,time
from pathlib import Path


SERVER = r'''
import json,os,socket,sys
mode=sys.argv[1];reuse=int(sys.argv[2])
with socket.socket() as listener:
 if reuse:listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
 listener.bind(('127.0.0.1',0));listener.listen(1)
 print(json.dumps({'pid':os.getpid(),'port':listener.getsockname()[1],'reuseaddr':reuse}),flush=True)
 if mode=='listening':sys.stdin.buffer.read(1)
 else:
  connection,_=listener.accept()
  with connection:
   connection.sendall(b'owned CPU server closes first')
   connection.shutdown(socket.SHUT_WR)
'''


def states(port):
    result=[]
    for path in [Path('/proc/net/tcp'),Path('/proc/net/tcp6')]:
        if not path.exists():continue
        for line in path.read_text().splitlines()[1:]:
            fields=line.split()
            if int(fields[1].rsplit(':',1)[1],16)==port:result.append(fields[3])
    return result


def old_probe(port):
    with socket.socket() as probe:probe.bind(('127.0.0.1',port))


def refused(function,port):
    try:function(port)
    except OSError as error:
        assert error.errno==errno.EADDRINUSE,(error.errno,str(error))
        return error.errno
    raise AssertionError('Expected actual EADDRINUSE from the kernel')


def start(mode,reuse):
    child=subprocess.Popen([sys.executable,'-u','-c',SERVER,mode,str(reuse)],stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    row=json.loads(child.stdout.readline());assert row['pid']==child.pid
    return child,row


def dispose(child):
    if child.poll() is None:
        # This is exclusively our own direct Popen child, never a service PID.
        child.terminate()
    child.wait(timeout=5)


def time_wait(reuse):
    child,row=start('time_wait',reuse)
    try:
        with socket.create_connection(('127.0.0.1',row['port']),timeout=5) as peer:
            while peer.recv(4096):pass
        code=child.wait(timeout=5);assert code==0
        deadline=time.monotonic()+3
        while '06' not in states(row['port']) and time.monotonic()<deadline:time.sleep(.01)
        observed=states(row['port']);assert '06' in observed,observed
        return dict(row,actual_Popen_wait=code,kernel_tcp_states=observed)
    finally:dispose(child)


def run(source):
    assert sys.platform.startswith('linux') and Path('/proc/net/tcp').is_file()
    parsed=ast.parse(source.read_text())
    selected=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name=='_check_native_port_available']
    assert len(selected)==1
    function_source=ast.get_source_segment(source.read_text(),selected[0])
    # The comment can mention it, but the executable AST must never set it.
    assert not any(isinstance(node,ast.Attribute) and node.attr=='SO_REUSEPORT' for node in ast.walk(selected[0]))
    scope={'socket':socket};exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope)
    actual=lambda port:scope['_check_native_port_available']('127.0.0.1',port)
    old=time_wait(1);old_error=refused(old_probe,old['port']);actual(old['port'])
    reusable=dict(old,old_probe_errno=old_error,production_probe_passed=True)
    live,listener=start('listening',1)
    try:
        assert '0A' in states(listener['port'])
        live_error=refused(actual,listener['port']);assert live.poll() is None
        live.stdin.write('x');live.stdin.flush();live_code=live.wait(timeout=5);assert live_code==0
        listening=dict(listener,kernel_state_before='0A',production_probe_errno=live_error,
            service_remained_alive_after_probe=True,actual_Popen_wait=live_code)
    finally:dispose(live)
    strict=time_wait(0);strict_error=refused(actual,strict['port'])
    with socket.socket() as free:
        free.bind(('127.0.0.1',0));free_port=free.getsockname()[1]
    actual(free_port)
    return dict(schema='q38-native-port-reuse-real-Linux-CPU-v1',passed=True,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),production_helper_source=function_source,
        reusable_TIME_WAIT=reusable,active_listener=listening,
        non_reusable_prior_TIME_WAIT=dict(strict,production_probe_errno=strict_error,
            scope='SO_REUSEADDR does not weaken a prior socket that did not opt into reuse'),
        free_ephemeral_port_passed=True,SO_REUSEPORT_enabled=False,existing_services_touched=False,
        GPU_Ray_Torch_native_or_model_execution=False,
        scope='Actual kernel TCP states and only this test\'s direct Popen children; no native GPU bootstrap or RL qualification')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-server-source',type=Path,default=Path(__file__).resolve().parents[1]/'adapter/qwen38_world_server.py')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=run(args.world_server_source)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
