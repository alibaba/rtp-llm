"""Pure input-fidelity comparison, summaries and grading.

Metrics describe infinite-history input structure, not finite-cache performance.
Acquisition and production-generator orchestration live in diagnostics.
"""
import math
from collections import Counter

from traffic.statistics import distribution
from traffic.lineage_model import DEFAULT_PREFIX_BLOCK_SIZE as BLOCK

DEFAULT_THRESHOLDS = dict(length_ks=[.03, .10], joint_tv=[.15, .35], depth_ks=[.05, .25])
BINS = dict(length_log2_min=9, length_log2_max=21, length_bins=32,
            depth_max_blocks=2000, depth_bins=16, overflow='clamped_to_edge_bin')


def ks(a, b):
    if not a or not b:
        return None
    ca, cb = Counter(a), Counter(b)
    xa = xb = 0
    result = 0
    for value in sorted(ca.keys() | cb.keys()):
        xa += ca[value] / len(a)
        xb += cb[value] / len(b)
        result = max(result, abs(xa-xb))
    return result


def correlation(x, y):
    if not x or not y:
        return None
    if len(x) != len(y):
        raise ValueError("correlation requires paired observations")
    mx, my = sum(x)/len(x), sum(y)/len(y)
    vx = sum((v-mx)**2 for v in x)
    vy = sum((v-my)**2 for v in y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(vx*vy) if vx and vy else None


def histogram(shape):
    if (not shape["blocks"] or len(shape["blocks"]) != len(shape["shared"])
            or any(b <= 0 for b in shape["blocks"])):
        raise ValueError("shape requires nonempty paired observations with positive blocks")
    grid = [[0]*BINS["depth_bins"] for _ in range(BINS["length_bins"])]
    length_span = BINS["length_log2_max"] - BINS["length_log2_min"]
    for b,s in zip(shape['blocks'], shape['shared']):
        length_bin = int((math.log2(b*BLOCK)-BINS["length_log2_min"]) / length_span * BINS["length_bins"])
        depth_bin = int(s / BINS["depth_max_blocks"] * BINS["depth_bins"])
        x = min(BINS["length_bins"]-1, max(0, length_bin))
        y = min(BINS["depth_bins"]-1, max(0, depth_bin))
        grid[x][y] += 1
    return grid


def summarize(shape):
    blocks, shared = shape['blocks'], shape['shared']
    joint = histogram(shape)
    n = len(blocks)
    return dict(requests=n, input_tokens=distribution([b*BLOCK for b in blocks]),
                depth_blocks=distribution(shared), cold_fraction=sum(s==0 for s in shared)/n,
                token_weighted_sharing=sum(shared)/sum(blocks),
                families=len(shape['families']), top5=sum(shape['families'][:5])/n,
                corr_log_length_depth=correlation([math.log2(b*BLOCK) for b in blocks],shared),
                length_counts=sorted(Counter(b*BLOCK for b in blocks).items()),
                depth_counts=sorted(Counter(shared).items()), joint_counts=joint)


def compare(real, generated):
    h1, h2 = histogram(real), histogram(generated)
    n1,n2=len(real['blocks']),len(generated['blocks'])
    return dict(length_ks=ks(real['blocks'],generated['blocks']),
                depth_ks=ks(real['shared'],generated['shared']),
                joint_tv=sum(abs(a/n1-b/n2) for r1,r2 in zip(h1,h2) for a,b in zip(r1,r2))/2)


def thresholds(value=None):
    result = dict(DEFAULT_THRESHOLDS, **(value or {}))
    if set(result) != set(DEFAULT_THRESHOLDS) or any(
        not isinstance(v, list) or len(v)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in v)
        or not 0 <= v[0] <= v[1] <= 1 for v in result.values()):
        raise ValueError('thresholds require [pass_max, warn_max] in [0,1] for length_ks/joint_tv/depth_ks')
    return result


def grade(metrics, limits=None, valid=True):
    if not valid:
        return dict(length='UNASSESSED', cache='UNASSESSED', structure='UNASSESSED')
    limits = thresholds(limits)
    scores = {key: 0 if value <= limits[key][0] else 1 if value <= limits[key][1] else 2
              for key,value in metrics.items()}
    names=['PASS','WARN','FAIL']
    return dict(length=names[scores['length_ks']],
                structure=names[max(scores.values())],
                cache=names[max(1,max(scores.values()))])
