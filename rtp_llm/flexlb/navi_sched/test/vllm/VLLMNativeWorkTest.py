import dataclasses,gc,itertools,random,unittest,weakref
from navi_sched.adapters.vllm.VLLMNativeWork import NativeRequestProgress,NativeWorkPrefix,NativeWorkTracker,NativeWorkWindow

def vector(x):return dataclasses.astuple(x)[1:]
def expected(spans):
    # Independent direct window calculation, grouping contiguous request spans.
    by={}
    for request,context,tokens in spans:
        if id(request)not in by:by[id(request)]=[context,0]
        by[id(request)][1]+=tokens
    return (len(by),sum(h for h,m in by.values()),sum(m for h,m in by.values()),sum(m*m for h,m in by.values()),sum(h*m for h,m in by.values()))

class NativeWorkTrackerTest(unittest.TestCase):
    def test_earlier_request_only_partial_window_is_counted(self):
        t=NativeWorkTracker();long=NativeRequestProgress();short=NativeRequestProgress()
        t.dispatch([(long,4096,8192)]);window=NativeWorkWindow.capture(t.prefix,[long])
        end=t.dispatch([(long,12288,4096),(short,147456,2488)])
        self.assertEqual((2,159744,6584,4096**2+2488**2,417202176),vector(window.project(end)))
        t.dispatch([(long,16384,8192)])
        self.assertEqual((2,159744,6584,4096**2+2488**2,417202176),vector(window.project(end)))

    def test_late_ready_request_between_target_members_is_included(self):
        t=NativeWorkTracker();a,b,p=NativeRequestProgress(),NativeRequestProgress(),NativeRequestProgress();window=NativeWorkWindow.capture(t.prefix,[])
        t.dispatch([(a,106496,3957)]);t.dispatch([(p,8192,8192)]);end=t.dispatch([(p,16384,4096),(b,36864,4493)])
        self.assertEqual(expected([(a,106496,3957),(p,8192,8192),(p,16384,4096),(b,36864,4493)]),vector(window.project(end)))

    def test_request_with_no_work_in_window_does_not_count(self):
        t=NativeWorkTracker();p,q=NativeRequestProgress(),NativeRequestProgress();t.dispatch([(p,5,10)]);w=NativeWorkWindow.capture(t.prefix,[p]);end=t.dispatch([(q,20,4)])
        self.assertEqual((1,20,4,16,80),vector(w.project(end)))

    def test_step_with_multiple_spans_for_same_request(self):
        t=NativeWorkTracker();p=NativeRequestProgress();w=NativeWorkWindow.capture(t.prefix,[]);end=t.dispatch([(p,5,3),(p,8,4)])
        self.assertEqual((1,5,7,49,35),vector(w.project(end)));self.assertEqual([1],p._steps)

    def test_snapshot_deduplicates_shared_request_refs(self):
        t=NativeWorkTracker();p=NativeRequestProgress();t.dispatch([(p,5,3)]);w=NativeWorkWindow.capture(t.prefix,[p,p,p]);end=t.dispatch([(p,8,4)])
        self.assertEqual((1,8,4,16,32),vector(w.project(end)));self.assertEqual(1,len(w.old_requests))

    def test_released_request_is_kept_only_by_existing_window(self):
        t=NativeWorkTracker();p=NativeRequestProgress();t.dispatch([(p,0,3)]);w=NativeWorkWindow.capture(t.prefix,[p]);ref=weakref.ref(p);del p;gc.collect();self.assertIsNotNone(ref())
        del w;gc.collect();self.assertIsNone(ref())

    def test_tracker_retains_no_lifetime_request_map(self):
        t=NativeWorkTracker();refs=[]
        for i in range(1000):
            p=NativeRequestProgress();refs.append(weakref.ref(p));t.dispatch([(p,i,1)])
        del p;gc.collect();self.assertTrue(all(r()is None for r in refs));self.assertEqual(('prefix',),t.__slots__)

    def test_exact_large_integer_prefix_subtraction(self):
        t=NativeWorkTracker();huge=10**30;t.prefix=NativeWorkPrefix(100,100,huge,huge,huge**2,huge**2);w=NativeWorkWindow.capture(t.prefix,[]);p=NativeRequestProgress();end=t.dispatch([(p,37,64)])
        self.assertEqual((1,37,64,4096,2368),vector(w.project(end)))

    def test_rejected_step_does_not_partially_advance_progress(self):
        t=NativeWorkTracker();a,b=NativeRequestProgress(),NativeRequestProgress();t.dispatch([(b,10,3)]);before=t.prefix
        with self.assertRaises(ValueError):t.dispatch([(a,0,5),(b,12,1)])
        self.assertEqual(before,t.prefix);self.assertEqual(0,a.compute_tokens);self.assertEqual(3,b.compute_tokens)

    def test_empty_step_does_not_increment_compute_step_count(self):
        t=NativeWorkTracker();self.assertEqual(NativeWorkPrefix(),t.dispatch([]))

    def test_one_shot_iterator_cannot_silently_skip_dispatch(self):
        t=NativeWorkTracker();p=NativeRequestProgress()
        with self.assertRaises(TypeError):t.dispatch(iter([(p,0,1)]))
        self.assertEqual(NativeWorkPrefix(),t.prefix);self.assertEqual(0,p.compute_tokens)

    def test_overlapping_windows_are_independent_after_further_steps(self):
        t=NativeWorkTracker();p=NativeRequestProgress();first=NativeWorkWindow.capture(t.prefix,[]);t.dispatch([(p,5,3)]);second=NativeWorkWindow.capture(t.prefix,[p]);end=t.dispatch([(p,8,4)]);t.dispatch([(p,12,2)])
        self.assertEqual((1,5,7,49,35),vector(first.project(end)));self.assertEqual((1,8,4,16,32),vector(second.project(end)))

    def test_randomized_windows_match_direct_native_token_grouping(self):
        rng=random.Random(1701)
        for trial in range(100):
            t=NativeWorkTracker();requests=[NativeRequestProgress()for _ in range(12)];prefixes=[t.prefix];steps=[]
            for i in range(30):
                spans=[]
                for p in rng.sample(requests,rng.randint(1,5)):
                    context=rng.randint(0,10000)if p.initial_context_tokens is None else p.initial_context_tokens+p.compute_tokens
                    spans.append((p,context,rng.randint(1,8192)))
                steps.append(spans);prefixes.append(t.dispatch(spans))
            for begin,end in[(0,30),(1,3),(3,19),(15,30),(29,30)]:
                w=NativeWorkWindow.capture(prefixes[begin],requests)
                self.assertEqual(expected([s for step in steps[begin:end]for s in step]),vector(w.project(prefixes[end])))

if __name__=='__main__':unittest.main(verbosity=2)
