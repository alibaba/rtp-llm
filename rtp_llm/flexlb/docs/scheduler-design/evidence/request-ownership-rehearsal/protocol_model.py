#!/usr/bin/env python3
"""Bounded design rehearsal, NOT a model extracted from Java or a JVM proof.

Each event is one hypothesized atomic boundary. Enumerates every interleaving
preserving each actor's program order, with no state merging. Failed variants
represent proposed simplifications, not bugs claimed in the current code.
Run: python3 protocol_model.py > results.json
"""
import copy
import json


def explore(name, initial, actors, step, check, finish):
    result = dict(name=name, visited_prefixes=0, complete_traces=0,
                  violations=0, deadlocks=0, first_counterexample=None)

    def failure(trace, state, message):
        result['violations'] += 1
        if result['first_counterexample'] is None:
            result['first_counterexample'] = dict(trace=trace, reason=message, state=state)

    def dfs(state, positions, trace):
        result['visited_prefixes'] += 1
        try:
            check(state)
        except AssertionError as error:
            failure(trace, state, str(error))
            return
        if all(p == len(a) for p, a in zip(positions, actors)):
            result['complete_traces'] += 1
            try:
                finish(state)
            except AssertionError as error:
                failure(trace, state, str(error))
            return
        advanced = False
        for i, actor in enumerate(actors):
            if positions[i] == len(actor):
                continue
            event = actor[positions[i]]
            candidate = copy.deepcopy(state)
            if not step(candidate, event):
                continue
            advanced = True
            next_positions = list(positions)
            next_positions[i] += 1
            dfs(candidate, next_positions, trace + [event])
        if not advanced:
            result['deadlocks'] += 1
            failure(trace, state, 'No enabled event: bounded actors cannot finish')

    dfs(initial, [0] * len(actors), [])
    return result


def response(eager=False):
    def step(s, e):
        if e == 'ack_observed':
            s['ack'] = True
            if eager and not s['terminal'] and s['winner'] is None:
                s['winner'] = 'success'
        elif e == 'delivery_select':
            s['selected_at_delivery_boundary'] = True
            if s['winner'] is None and not s['terminal']:
                s['winner'] = 'success'
            s['delivery_ticket'] = s['winner'] == 'success'
        elif e == 'inactivity_terminal':
            s['terminal'] = True
            if not s['selected_at_delivery_boundary']:
                s['timeout_before_selection'] = True
        elif e == 'resource_cleanup':
            s['clean'] = True
        elif e == 'terminal_select':
            if s['winner'] is None:
                s['winner'] = 'timeout'
            s['terminal_ticket'] = s['winner'] == 'timeout'
        elif e == 'delivery_publish' and s.get('delivery_ticket'):
            s['future'] = 'success'
            s['publications'] += 1
        elif e == 'terminal_publish' and s.get('terminal_ticket'):
            s['future'] = 'timeout'
            s['publications'] += 1
        return True

    def check(s):
        assert s['publications'] <= 1, 'Response published more than once'
        assert not (s['timeout_before_selection'] and s['winner'] == 'success'), \
            'ACK observation was incorrectly treated as the response-selection boundary'
        assert s['future'] is None or s['future'] == s['winner'], 'Selected response changed'

    def finish(s):
        assert s['clean'] and s['future'] is not None, 'Completion or resource release lost'

    return explore('response/eager_ack' if eager else 'response/explicit_selection',
                   dict(ack=False, terminal=False, clean=False, winner=None, future=None,
                        selected_at_delivery_boundary=False, timeout_before_selection=False,
                        publications=0),
                   [['ack_observed', 'delivery_select', 'delivery_publish'],
                    ['inactivity_terminal', 'resource_cleanup', 'terminal_select', 'terminal_publish']],
                   step, check, finish)


def retirement(ignore_pins=False):
    def step(s, e):
        if e == 'try_acquire':
            s['pin'] = not s['retiring']
        elif e == 'release_pin':
            s['pin'] = False
        elif e == 'close_gate':
            s['retiring'] = True
        elif e == 'cleanup':
            if s['pin'] and not ignore_pins:
                return False
            s['clean'] = True
        return True

    def check(s):
        assert not (s['clean'] and s['pin']), 'Retirement cleanup overlaps an admitted handoff'

    return explore('retirement/ignore_pins' if ignore_pins else 'retirement/drain_pins',
                   dict(pin=False, retiring=False, clean=False),
                   [['try_acquire', 'release_pin'], ['close_gate', 'cleanup']], step, check,
                   lambda s: None)


def publication_close(queue_only=False):
    def step(s, e):
        if e == 'reserve':
            s['accepted'] = not s['closing']
            s['reserved'] = int(s['accepted'])
        elif e == 'submit' and s['accepted']:
            s['lost'] = s['closed']
            s['queued'] = 1
        elif e == 'complete' and s['accepted']:
            s['queued'] = 0
            s['reserved'] = 0
            s['done'] = True
        elif e == 'begin_close':
            s['closing'] = True
        elif e == 'finish_close':
            if s['queued'] if queue_only else s['reserved']:
                return False
            s['closed'] = True
        return True

    def check(s):
        assert not s['lost'], 'Accepted publication lost between reservation and queue submission'
        assert not (s['closed'] and s['reserved']), 'Close returned while an accepted obligation exists'

    return explore('publication/queue_only_close' if queue_only else 'publication/reservation_drain',
                   dict(accepted=False, reserved=0, queued=0, closing=False, closed=False,
                        done=False, lost=False),
                   [['reserve', 'submit', 'complete'], ['begin_close', 'finish_close']],
                   step, check, lambda s: None)


def admission(published):
    def settle_cancel(s):
        if s['cancel'] and not s['operation'] and s['owned']:
            s['owned'] = False
            s['releases'] += 1

    def step(s, e):
        if e == 'bind':
            if not s['cancel']:
                s['operation'] = True
                s['bound'] = True
        elif e == 'publish_outside_request_lock' and s['bound']:
            s['owned'] = published
        elif e == 'finish_admission':
            s['operation'] = False
            if not s['owned']:
                s['bound'] = False
            settle_cancel(s)
        elif e == 'cancel':
            s['cancel'] = True
            settle_cancel(s)
        return True

    def check(s):
        assert s['releases'] <= 1, 'Route released twice'

    def finish(s):
        assert not s['owned'] and not s['operation'], 'Cancellation lost across external publication'

    return explore('admission/publish_' + str(published),
                   dict(cancel=False, operation=False, bound=False, owned=False, releases=0),
                   [['bind', 'publish_outside_request_lock', 'finish_admission'], ['cancel']],
                   step, check, finish)


def stale_callback(id_only=False):
    def step(s, e):
        if e == 'replace_route_or_request_generation':
            s['generation'] = 'new'
            s['active'] = True
            s['replaced'] = True
        elif e == 'old_callback':
            if id_only or s['generation'] == 'old':
                s['active'] = False
        return True

    def check(s):
        assert not s['replaced'] or s['active'], 'Old identity callback settled the new route/generation'

    return explore('identity/request_id_only' if id_only else 'identity/exact_object',
                   dict(generation='old', active=True, replaced=False),
                   [['replace_route_or_request_generation'], ['old_callback']], step, check,
                   lambda s: None)


def deadline(remember_early=True, with_cancel=False):
    def step(s, e):
        if e == 'fire':
            s['fired'] = True
            if s['state'] == 'prepared' and remember_early:
                s['state'] = 'early'
            elif s['state'] == 'armed':
                s['state'] = 'consumed'
                s['deliveries'] += 1
        elif e == 'install':
            s['installed'] = True
            if s['state'] == 'prepared':
                s['state'] = 'armed'
            elif s['state'] == 'early':
                s['state'] = 'consumed'
                s['deliveries'] += 1
        elif e == 'cancel' and s['state'] != 'consumed':
            s['state'] = 'canceled'
        return True

    def check(s):
        assert s['deliveries'] <= 1, 'Deadline delivered more than once'
        assert s['state'] != 'canceled' or s['deliveries'] == 0, 'Canceled registration delivered'

    def finish(s):
        assert s['state'] == 'canceled' or s['deliveries'] == 1, 'Fire-before-install lost a deadline'

    return explore('deadline/' + ('remember' if remember_early else 'forget') +
                   ('_cancel' if with_cancel else ''),
                   dict(state='prepared', fired=False, installed=False, deliveries=0),
                   [['install'], ['fire', 'fire']] + ([['cancel']] if with_cancel else []),
                   step, check, finish)


def synchronous_cancel(clean_first=False):
    # Queue owner is paused until the calling thread observes cancel() return.
    # Legal environmental schedule, not a dependency imposed on the real queue.
    def step(s, e):
        if e == 'choose_cancel':
            s['cancel'] = True
        elif e == 'publish_cancel':
            if clean_first and not s['clean']:
                return False
            s['future'] = 'canceled'
        elif e == 'return_from_cancel':
            s['returned'] = True
        elif e == 'resume_queue_cleanup':
            if not s['returned']:
                return False
            s['clean'] = True
        return True

    def check(s):
        assert not s['returned'] or s['future'] == 'canceled', 'cancel returned before future completion'

    return explore('cancel/cleanup_first' if clean_first else 'cancel/two_completion_dimensions',
                   dict(cancel=False, future=None, returned=False, clean=False),
                   [['choose_cancel', 'publish_cancel', 'return_from_cancel'], ['resume_queue_cleanup']],
                   step, check, lambda s: None)


if __name__ == '__main__':
    cases = [response(), retirement(), publication_close(), admission(True), admission(False),
             stale_callback(), deadline(), deadline(with_cancel=True), synchronous_cancel()]
    mutants = [response(True), retirement(True), publication_close(True), stale_callback(True),
               deadline(False), synchronous_cancel(True)]
    assert all(c['violations'] == 0 and c['complete_traces'] > 0 for c in cases)
    assert all(c['violations'] > 0 for c in mutants), 'A deliberately weakened design was not detected'
    print(json.dumps(dict(
        scope='Sequentially consistent bounded protocol model; no Java execution or extraction',
        omissions=['JVM memory visibility', 'real locks and reentrancy', 'executor failures',
                   'network/RPC outcomes', 'batch/preemption full transactions', 'performance'],
        corrected_design=cases, rejected_simplifications=mutants,
        total_complete_corrected_traces=sum(c['complete_traces'] for c in cases),
        total_corrected_prefixes=sum(c['visited_prefixes'] for c in cases)), indent=2))
