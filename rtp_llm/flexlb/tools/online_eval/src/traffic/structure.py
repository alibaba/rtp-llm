"""Input structure shared by parameter derivation and standalone fidelity analysis."""
from collections import Counter
from traffic.prefix_lineage import BLOCK


def capture_shape(events):
    # Only the first eight labels are needed for family identity. Avoid expanding
    # and retaining every full request in a long capture.
    prefixes, families = [], Counter()
    next_label = 1
    for _, blocks, parent, shared, reserve in events:
        prefix = prefixes[parent][:min(shared, 8)] if parent >= 0 else ()
        fresh = blocks - shared
        prefix += tuple(range(next_label, next_label + min(fresh, 8 - len(prefix))))
        next_label += fresh + reserve
        prefixes.append(prefix)
        families[prefix] += 1
    return dict(blocks=[e[1] for e in events], shared=[e[3] for e in events],
                families=sorted(families.values(), reverse=True))


def stratified(values, size):
    ordered = sorted(values)
    return [ordered[min(len(ordered)-1, int((i+.5)*len(ordered)/size))]
            for i in range(min(size, len(ordered)))] if len(ordered) >= size else ordered


def joint_distribution(events, strata=1024):
    return dict(schema_version=1, strata=strata, sort_key=['blocks', 'shared_blocks'],
                sampling='equal_mass_midpoint_strata',
                warm_pairs=[list(pair) for pair in stratified([(e[1], e[3]) for e in events if e[3]], strata)],
                cold_blocks=stratified([e[1] for e in events if not e[3]], strata))


class PrefixIndex:
    """Exact longest previously seen prefix using a radix trie of label runs.

    Consecutive integer labels occupy a single edge, including private suffixes.
    Explicit token blocks remain tuple keys. No probabilistic hashing is used.
    """
    def __init__(self):
        self.root = {}

    def observe(self, labels):
        runs = []
        for value in labels:
            value = tuple(value) if isinstance(value, list) else value
            if runs and isinstance(value, int) and isinstance(runs[-1][0], int) and value == runs[-1][0] + runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([value, 1])
        node, depth, matching = self.root, 0, True
        for start, length in runs:
            while length:
                edge = node.get(start)
                if edge is None:
                    node[start] = [length, {}]
                    node = node[start][1]
                    matching = False
                    break
                consumed = min(length, edge[0])
                if matching:
                    depth += consumed
                if consumed < edge[0]:
                    child = {start + consumed: [edge[0] - consumed, edge[1]]}
                    node[start] = [consumed, child]
                    node = child
                else:
                    node = edge[1]
                length -= consumed
                if length:
                    start += consumed
        return depth


def generated_shape(rows):
    index, families = PrefixIndex(), Counter()
    blocks, shared = [], []
    for row in rows:
        labels = row['input_token_blocks'][:row['il']//row['cache_key_block_size']]
        blocks.append(row['il'] / BLOCK)
        shared.append(index.observe(labels) * row['cache_key_block_size'] / BLOCK)
        families[tuple(tuple(v) if isinstance(v, list) else v for v in labels[:8])] += 1
    return dict(blocks=blocks, shared=shared, families=sorted(families.values(), reverse=True))
