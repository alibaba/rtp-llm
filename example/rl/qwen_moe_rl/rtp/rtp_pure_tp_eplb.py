"""Narrow EP=1/nonredundant multi-node identity-placement correction.

Physical TP hosts do not partition experts when EP=1. Delegate to the original
single-EP-node implementation for that topology; all other EP paths are intact.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import socket
import time

VERSION='pure_tp_ep1_identity_v1'


def corrected_create_redundant_expert(original,layer_num,expert_num,phy_exp_num,ep_size,num_nodes,phy2log_path=None):
    arguments=dict(layer_num=layer_num,expert_num=expert_num,phy_exp_num=phy_exp_num,
                   ep_size=ep_size,num_nodes=num_nodes,phy2log_path=phy2log_path)
    if ep_size!=1 or num_nodes<=1:
        return original(**arguments)
    for name,value in [('layer_num',layer_num),('expert_num',expert_num),
                       ('phy_exp_num',phy_exp_num),('num_nodes',num_nodes)]:
        if type(value) is not int or value<=0:
            raise ValueError(f'{VERSION}: {name} must be a positive integer')
    if phy_exp_num!=expert_num:
        raise ValueError(f'{VERSION}: multi-node EP=1 supports nonredundant experts only; '
                         f'phy_exp_num={phy_exp_num}, expert_num={expert_num}')
    # Preserve the original redundancy/divisibility/file-shape checks by using
    # the same EP=1 implementation, with one expert-placement node. The four
    # physical TP hosts remain unchanged in native distributed configuration.
    result=original(**dict(arguments,num_nodes=1))
    expected=list(range(expert_num))
    if len(result)!=layer_num or any(list(row)!=expected for row in result):
        raise ValueError(f'{VERSION}: pure TP requires complete identity expert placement in every layer')
    _record(arguments,result)
    return result


def _record(arguments,result):
    digest=hashlib.sha256(json.dumps(result,separators=(',',':')).encode()).hexdigest()
    proof=dict(schema='rtp-pure-tp-ep1-placement-v1',fix=VERSION,passed=True,
        pid=os.getpid(),hostname=socket.gethostname(),world_rank_env=os.getenv('WORLD_RANK'),
        replica_rank=os.getenv('RTP_REPLICA_RANK'),arguments=arguments,
        physical_tp_nodes=arguments['num_nodes'],expert_placement_nodes=1,
        every_tp_rank_loads_all_experts=True,identity_mapping_sha256=digest,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checked_unix=time.time())
    logging.info('RTP_PURE_TP_EPLB '+json.dumps(proof))
    audit=os.getenv('RLLM_RTP_ACCEPTANCE_DIR')
    if audit:
        directory=Path(audit)/'pure-tp-eplb';directory.mkdir(parents=True,exist_ok=True)
        path=directory/f"identity-replica{os.getenv('RTP_REPLICA_RANK','standalone')}-rank{os.getenv('WORLD_RANK','unset')}-pid{os.getpid()}.json"
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(proof,indent=2)+'\n');temporary.replace(path)


def install():
    """Install at native entrypoint top-level after qwen36_overlay.install()."""
    module=importlib.import_module('rtp_llm.model_loader.load_config')
    current=module.LoadConfig.create_redundant_expert
    if getattr(current,'_rtp_pure_tp_ep1_identity_fix',False):
        return current._rtp_pure_tp_ep1_provenance
    original=current
    @functools.wraps(original)
    def corrected(layer_num,expert_num,phy_exp_num,ep_size,num_nodes,phy2log_path=None):
        return corrected_create_redundant_expert(original,layer_num,expert_num,phy_exp_num,
                                               ep_size,num_nodes,phy2log_path)
    source=inspect.getsource(original)
    proof=dict(fix=VERSION,original_source_file=inspect.getsourcefile(original),
        original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        hook_source_file=__file__,hook_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='EP=1, physical nodes>1, nonredundant identity only; every other branch delegates unchanged')
    corrected._rtp_pure_tp_ep1_identity_fix=True
    corrected._rtp_pure_tp_ep1_provenance=proof
    module.LoadConfig.create_redundant_expert=staticmethod(corrected)
    logging.info('RTP_PURE_TP_EPLB_INSTALLED '+json.dumps(proof))
    return proof
