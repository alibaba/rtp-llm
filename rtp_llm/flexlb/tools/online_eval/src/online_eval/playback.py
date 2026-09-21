"""Scenario client playback parameters. Sources never decide when to send."""
import math

FIELDS={'mode','qps','speed','max_laps','identity','retain_probability','seed',
        'ramp_up_seconds','burst_factor','burst_period_seconds','burst_duty',
        'diurnal_amplitude','diurnal_period_seconds'}


def normalize(client):
    env=dict(client)
    p=env.pop('playback',None)
    if p is None:
        # Existing raw client controls are still accepted; LOOP must be explicit.
        return env,dict(mode=env.get('SEND_MODE','replay'),
            qps=float(env.get('SEND_MODE_QPS',0)),speed=float(env.get('REPLAY_SPEED',1)),
            max_laps=int(env.get('MAX_LAPS',0 if env.get('LOOP')=='true' else 1)),
            identity=env.get('LAP_IDENTITY','structural-relabel'),
            retain_probability=float(env.get('LAP_RETAIN_PROBABILITY',0)),
            seed=int(env.get('PLAYBACK_SEED',0)),
            ramp_up_seconds=float(env.get('RAMP_UP_SECONDS',0)),
            burst_factor=float(env.get('BURST_FACTOR',1)),burst_period_seconds=float(env.get('BURST_PERIOD_SECONDS',10)),
            burst_duty=float(env.get('BURST_DUTY',0.1)),diurnal_amplitude=float(env.get('DIURNAL_AMPLITUDE',0)),
            diurnal_period_seconds=float(env.get('DIURNAL_PERIOD_SECONDS',86400)))
    if not isinstance(p,dict) or set(p)-FIELDS:
        raise ValueError('unknown playback fields')
    raw={'SEND_MODE','SEND_MODE_QPS','REPLAY_SPEED','LOOP','MAX_LAPS','LAP_IDENTITY',
         'LAP_RETAIN_PROBABILITY','PLAYBACK_SEED','RAMP_UP_SECONDS','GRADIENT',
         'BURST_FACTOR','BURST_PERIOD_SECONDS','BURST_DUTY','DIURNAL_AMPLITUDE','DIURNAL_PERIOD_SECONDS'}
    if set(env)&raw:
        raise ValueError('playback cannot be combined with raw pacing/loop controls')
    mode=p.get('mode','uniform')
    if mode not in ('uniform','true-ts','burst','gradient'):
        raise ValueError('invalid playback mode')
    laps=p.get('max_laps',1)
    if type(laps) is not int or laps < 0 or (laps==0 and int(env.get('DURATION_S',0))<=0):
        raise ValueError('max_laps requires a positive count or 0 with bounded duration')
    identity=p.get('identity','structural-relabel')
    if identity not in ('none','structural-relabel','partial'):
        raise ValueError('invalid lap identity')
    def number(key,default,low,high=float('inf')):
        v=p.get(key,default)
        if type(v) not in (int,float) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError('invalid playback '+key)
        return v
    qps=number('qps',0,0)
    if mode!='true-ts' and qps<=0:
        raise ValueError('positive playback qps required')
    probability=number('retain_probability',0,0,1)
    if identity!='partial' and 'retain_probability' in p:
        raise ValueError('retain_probability belongs to partial identity')
    speed=number('speed',1,1e-9)
    ramp=number('ramp_up_seconds',0,0)
    if mode=='gradient' and ramp<=0:
        raise ValueError('gradient requires ramp_up_seconds')
    if type(p.get('seed',0)) is not int:
        raise ValueError('playback seed must be integer')
    env.update(SEND_MODE='replay' if mode=='true-ts' else 'uniform',SEND_MODE_QPS=str(qps),
        REPLAY_SPEED=str(speed),MAX_LAPS=str(laps),LOOP='true' if laps!=1 else 'false',
        LAP_IDENTITY=identity,LAP_RETAIN_PROBABILITY=str(probability),PLAYBACK_SEED=str(p.get('seed',0)),
        RAMP_UP_SECONDS=str(ramp),BURST_FACTOR=str(number('burst_factor',1,1)),
        BURST_PERIOD_SECONDS=str(number('burst_period_seconds',10,0.001)),
        BURST_DUTY=str(number('burst_duty',0.1,0.001,0.999)),
        DIURNAL_AMPLITUDE=str(number('diurnal_amplitude',0,0,0.99)),
        DIURNAL_PERIOD_SECONDS=str(number('diurnal_period_seconds',86400,0.001)))
    if mode=='burst' and float(env['BURST_FACTOR'])<=1:
        raise ValueError('burst mode requires burst_factor > 1')
    if mode=='true-ts' and any(k in p for k in ('qps','ramp_up_seconds','burst_factor','diurnal_amplitude')):
        raise ValueError('true-ts uses only source timestamps and replay speed')
    return env,dict(p,mode=mode,max_laps=laps,identity=identity,retain_probability=probability,seed=p.get('seed',0))


def comparison_notice(a,b):
    fields=('realism','tail','arrival')
    differences=[k for k in fields if a.get(k)!=b.get(k)]
    if a.get('source',{}).get('kind')!=b.get('source',{}).get('kind'):
        differences.append('source kind')
    for k in ('identity','retain_probability'):
        if a.get('playback',{}).get(k)!=b.get('playback',{}).get(k):differences.append(k)
    return ('不可直比命中率绝对值 / cache-hit absolute values are not directly comparable: '+', '.join(differences)) if differences else None


def iteration_windows(rows):
    """Actual client send windows per lap; missing iteration remains unknown."""
    groups={}
    for row in rows:
        lap=row.get('iteration')
        timestamp=row.get('send_start_epoch_ms')
        if type(lap) is not int or type(timestamp) not in (int,float):
            continue
        group=groups.setdefault(lap,dict(iteration=lap,requests=0,input_tokens=0,
            start_epoch_ms=timestamp,end_epoch_ms=timestamp))
        group['requests']+=1
        group['input_tokens']+=row.get('input_len',row.get('il',0)) or 0
        group['start_epoch_ms']=min(group['start_epoch_ms'],timestamp)
        group['end_epoch_ms']=max(group['end_epoch_ms'],timestamp)
    return [groups[k] for k in sorted(groups)]
