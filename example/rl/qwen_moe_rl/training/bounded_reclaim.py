"""Run-bound, read-only model range planning and bounded cache advice.

All four local rank checkpoint receipts must already be complete. The caller
must ensure every world rank has reached a pre-Adam barrier before invoking
this from the one elected coordinator on each node. This module never touches
global cache controls, writes model bytes, or assumes advice released memory.
"""
from pathlib import Path
from bisect import bisect_right
from collections import deque
import ctypes,hashlib,json,mmap,os,struct,threading,time
from source_range_planner import tensor_intervals,page_aligned_union,subtract_intervals

INDEX_SHA='e36c40d4e99b2714fff821218a0433bda2dec46afdb1ebee8ce96ced997928ee'
MODEL=Path('/mnt/fuse/Qwen/Qwen3.8-2.4T-A95B')
MAX_HEADER=16*1024**2
MAX_BATCH=4*1024**3
MAX_TOTAL=512*1024**3
MAX_SAMPLE_READ_BYTES=64*1024**2
MAX_WIDE_SCAN=5*1024**4
MAX_SCAN_WINDOW=2*1024**3
# Bound metadata/syscall overhead independently of byte budgets. A wide round
# may combine many small authorized intervals, but scans at most 2 GiB total.
MAX_WIDE_WINDOWS_PER_BATCH=32
MAX_ADVICE_RANGES_PER_BATCH=32

def _sha(data):return hashlib.sha256(data).hexdigest()

def build_plan(run_root,peers,hostname,*,model=MODEL,expected_index_sha=INDEX_SHA,
               expected_parameters=7324,expected_sources=1590,world=72):
    root=Path(run_root);model=Path(model)
    if len(peers)!=4 or len({p['rank'] for p in peers})!=4 or any(p['hostname']!=hostname for p in peers):
        raise ValueError('Exactly four distinct local rank identities required')
    raw_index=(model/'model.safetensors.index.json').read_bytes()
    if _sha(raw_index)!=expected_index_sha:raise ValueError('Official index SHA mismatch')
    index=json.loads(raw_index)['weight_map'];headers={};all_spans=[];receipt_proofs=[];rank_keys=[];rank_files=[]
    for peer in sorted(peers,key=lambda p:p['rank']):
        rank=peer['rank']
        path=root/'training'/('checkpoint-load-rank%d.json'%rank);raw=path.read_bytes();receipt=json.loads(raw)
        if (receipt.get('rank')!=rank or receipt.get('world_size')!=world
                or receipt.get('pid')!=peer['pid']
                or receipt.get('hostname')!=hostname or receipt.get('runtime_parameters')!=expected_parameters
                or receipt.get('source_keys')!=expected_sources or receipt.get('source_provider')!='platform_mounted'
                or receipt.get('source_provider_close_receipt',{}).get('source')!='existing_snapshot'
                or receipt.get('all_master_dtype')!='bfloat16' or receipt.get('official_source_dtype')!='bfloat16'
                or len(receipt.get('source_slices',[]))!=expected_parameters
                or receipt.get('finished_unix',0)<receipt.get('started_unix',1)):
            raise ValueError('Incomplete or mismatched current-run checkpoint receipt')
        counted=0;keys=set();files=set()
        for row in receipt['source_slices']:
            if not row.get('source_coordinates_verified') or not row.get('full_destination_BF16_bytes_verified'):
                raise ValueError('A current source/destination slice is not fully verified')
            if len(row.get('full_slice_bytes_sha256',[]))!=row.get('source_reads'):
                raise ValueError('Complete current source hash list is missing')
            name=row['checkpoint_file']
            keys.add(row['canonical_name']);files.add(name)
            if index.get(row['canonical_name'])!=name:raise ValueError('Canonical source differs from official index')
            if name not in headers:
                path=model/name;fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
                try:
                    file_stat=os.fstat(fd);size=file_stat.st_size;raw_length=os.pread(fd,8,0)
                    if len(raw_length)!=8:raise ValueError('Incomplete safetensors prefix')
                    length=struct.unpack('<Q',raw_length)[0]
                    if not 0<length<=MAX_HEADER:raise ValueError('Invalid header length')
                    raw_header=os.pread(fd,length,8)
                    if len(raw_header)!=length:raise ValueError('Incomplete safetensors header')
                    headers[name]=(json.loads(raw_header),8+length,size,_sha(raw_header),file_stat.st_dev,file_stat.st_ino)
                finally:os.close(fd)
            header,start,size,_,_,_=headers[name]
            spans=tensor_intervals(row,header,start,size)
            # The loader splits a fragment into <=128MiB reads. The ranges
            # remain exact even when one fragment used multiple reads.
            if row['source_reads']<len(spans):raise ValueError('Source read count cannot cover fragments')
            all_spans.extend(spans);counted+=row['local_bytes']
        if counted!=receipt['loaded_local_numel']*2:raise ValueError('Receipt shard byte total mismatch')
        if len(keys)!=expected_sources or len(files)!=receipt.get('main_policy_source_files_consumed'):
            raise ValueError('Current receipt key/file completion mismatch')
        rank_keys.append(keys);rank_files.append(files)
        receipt_proofs.append(dict(rank=rank,sha256=_sha(raw),local_bytes=counted))
    page=os.sysconf('SC_PAGE_SIZE');union=page_aligned_union(all_spans,page)
    common_keys=set.intersection(*rank_keys);common_files=set.intersection(*rank_files)
    if len(common_keys)!=expected_sources or any(keys!=common_keys for keys in rank_keys):
        raise ValueError('Four completed ranks disagree on canonical source keys')
    if any(files!=common_files for files in rank_files):raise ValueError('Four completed ranks disagree on source files')
    # Wider stage is still restricted to original main-policy tensors seen in
    # every completed receipt. No unlisted/MTP tensor or header is included.
    wide=[]
    for key in sorted(common_keys):
        name=index[key]
        if name not in common_files:raise ValueError('Common policy tensor not in common closed files')
        header,start,size,_,_,_=headers[name];item=header[key]
        if item['dtype']!='BF16':raise ValueError('Unexpected wider tensor dtype')
        left,right=item['data_offsets']
        if not 0<=left<=right or start+right>size:raise ValueError('Wider tensor outside file')
        wide.append((name,start+left,start+right))
    wider_only=subtract_intervals(page_aligned_union(wide,page),union)
    return dict(schema='q38-current-node-source-reclaim-plan-v1',run_id=root.name,run_root=str(root),hostname=hostname,
        ranks=receipt_proofs,index_sha256=_sha(raw_index),page_size=page,raw_intervals=len(all_spans),
        raw_bytes=sum(b-a for _,a,b in all_spans),union_bytes=sum(b-a for _,a,b in union),
        files={name:dict(size=data[2],header_sha256=data[3],device=data[4],inode=data[5]) for name,data in headers.items()},
        intervals=union,wider_only_intervals=wider_only,wider_only_bytes=sum(b-a for _,a,b in wider_only),
        shared_cache_IO_impact_possible=True,metadata_only=True)

def _open_verified(plan,model,name):
    fd=os.open(Path(model)/name,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        stat=os.fstat(fd);expected=plan['files'][name]
        if (stat.st_size,stat.st_dev,stat.st_ino)!=(expected['size'],expected['device'],expected['inode']):
            raise ValueError('Checkpoint file identity changed')
        length=struct.unpack('<Q',os.pread(fd,8,0))[0]
        if not 0<length<=MAX_HEADER or _sha(os.pread(fd,length,8))!=expected['header_sha256']:
            raise ValueError('Checkpoint header changed')
        return fd,stat
    except BaseException:
        os.close(fd);raise

def resident_runs(plan,name,start,end,*,model=MODEL):
    """mincore is only a selector, never evidence of charged bytes released."""
    page=plan['page_size'];length=end-start
    if start%page or length<=0 or length%page or length>MAX_SCAN_WINDOW:
        raise ValueError('Invalid bounded mincore scan')
    fd,_=_open_verified(plan,model,name)
    try:
        view=mmap.mmap(fd,length,access=mmap.ACCESS_COPY,offset=start)
        pointer=ctypes.c_char.from_buffer(view);vector=(ctypes.c_ubyte*(length//page))()
        libc=ctypes.CDLL(None,use_errno=True);libc.mincore.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
        if libc.mincore(ctypes.addressof(pointer),length,vector)!=0:raise OSError(ctypes.get_errno(),'mincore')
        values=[bool(x&1) for x in vector];del pointer;view.close()
    finally:os.close(fd)
    runs=[];begin=None
    for i,resident in enumerate(values+[False]):
        if resident and begin is None:begin=i
        elif not resident and begin is not None:
            runs.append((name,start+begin*page,start+i*page));begin=None
    return runs

def advise_one_batch(plan,offset_intervals,*,model=MODEL,max_bytes=MAX_BATCH):
    """One exact ≤4GiB batch; return facts, never implied reclaimed bytes."""
    page=plan['page_size']
    if max_bytes<=0 or max_bytes>MAX_BATCH or max_bytes%page:raise ValueError('Invalid hard batch bound')
    root=Path(plan['run_root'])
    for receipt in plan['ranks']:
        path=root/'training'/('checkpoint-load-rank%d.json'%receipt['rank'])
        if _sha(path.read_bytes())!=receipt['sha256']:raise ValueError('Current-run source receipt changed')
    allowed={}
    for name,start,end in plan['intervals']+plan.get('wider_only_intervals',[]):
        allowed.setdefault(name,[]).append((start,end))
    for choices in allowed.values():choices.sort()
    # The caller may consume prefixes only of exact planned ranges, after
    # keeping its cursor. No unplanned source byte can be included.
    for name,start,end in offset_intervals:
        choices=allowed.get(name,[]);index=bisect_right(choices,(start,2**64))-1
        if index<0 or not choices[index][0]<=start<=end<=choices[index][1]:
            raise ValueError('Advice outside exact plan')
    batch=[];remaining=max_bytes
    for name,start,end in offset_intervals:
        # Each source range consumes its own page again when the verification
        # read is followed by a second DONTNEED. Count both syscalls toward
        # the exact same hard cap; advice bytes are not released bytes.
        if (remaining<2*page or len(batch)>=MAX_ADVICE_RANGES_PER_BATCH
                or (len(batch)+1)*2*page>MAX_SAMPLE_READ_BYTES):break
        take=min(end-start,remaining-page);take=(take//page)*page
        if take:
            batch.append((name,start,start+take));remaining-=take+page
    total=sum(b-a for _,a,b in batch)
    if total>max_bytes:raise RuntimeError('Batch limit exceeded')
    result=[];started=time.monotonic()
    for name,start,end in batch:
        fd,stat=_open_verified(plan,model,name)
        try:
            sample=min(page,end-start);before=os.pread(fd,sample,start)
            if len(before)!=sample:raise ValueError('Incomplete pre-advice sample')
            first=_sha(before);os.posix_fadvise(fd,start,end-start,os.POSIX_FADV_DONTNEED)
            after=os.pread(fd,sample,start);second=_sha(after);endstat=os.fstat(fd)
            if len(after)!=sample or first!=second or (endstat.st_dev,endstat.st_ino,endstat.st_size)!=(stat.st_dev,stat.st_ino,stat.st_size):
                raise RuntimeError('Read-only file sample or identity changed')
            # Verification read can fault the first page back in. Advise that
            # same already-authorized page again; success still does not prove
            # it was reclaimed, which only a fresh cgroup reading can show.
            os.posix_fadvise(fd,start,sample,os.POSIX_FADV_DONTNEED)
            result.append(dict(file=name,offset=start,length=end-start,sample_bytes=sample,
                sample_sha256_before=first,sample_sha256_after=second))
        finally:os.close(fd)
    return dict(schema='q38-bounded-source-cache-advice-v1',advised_bytes=total,ranges=result,
        advice_range_count=len(result),advice_range_limit=MAX_ADVICE_RANGES_PER_BATCH,
        sample_read_bytes=2*len(result)*page,re_advised_sample_bytes=len(result)*page,
        syscall_advised_bytes=total+len(result)*page,
        elapsed_seconds=time.monotonic()-started,read_only_fd=True,
        actual_cgroup_bytes_released_not_inferred=True)

class ReclaimCursor:
    """One node-local exclusive, monotonic coordinator; never auto-resume."""
    def __init__(self,plan,*,max_total=MAX_TOTAL):
        if not 0<max_total<=MAX_TOTAL or max_total%plan['page_size']:
            raise ValueError('Invalid total advice budget')
        root=Path(plan['run_root'])
        if root.name!=plan['run_id']:raise ValueError('Run root mismatch')
        self.plan=plan;self.max_total=max_total;self.index=0;self.cursor=None;self.total=0;self.round=0
        self._failed=False;self._lock=threading.Lock();self._owner_pid=os.getpid()
        self.exact_count=len(plan['intervals'])
        self.segments=plan['intervals']+plan.get('wider_only_intervals',[])
        self.wide_scan_limit=sum(end-start for _,start,end in plan.get('wider_only_intervals',[]))
        if self.wide_scan_limit>MAX_WIDE_SCAN:raise ValueError('Wider plan exceeds official-model scan ceiling')
        self.wide_scanned=0
        self.pending_resident=deque()
        self.directory=root/'cpu-reclaim';self.directory.mkdir(parents=True,exist_ok=True)
        claim=dict(schema='q38-node-reclaim-exclusive-claim-v1',run_id=plan['run_id'],hostname=plan['hostname'],
            coordinator_pid=os.getpid(),ranks=plan['ranks'],max_total_bytes=max_total,created_unix=time.time())
        claim_text=json.dumps(claim,indent=2)+'\n'
        with (self.directory/'CLAIM.json').open('x') as stream:
            stream.write(claim_text);stream.flush();claim_stat=os.fstat(stream.fileno())
        self._claim_identity=(claim_stat.st_dev,claim_stat.st_ino)
        self._claim_size=len(claim_text.encode('utf-8'))
        self._claim_sha256=_sha(claim_text.encode('utf-8'))

    def _verify_owner_claim(self):
        if os.getpid()!=self._owner_pid:
            raise RuntimeError('Reclaim cursor used outside its original coordinator process')
        if os.path.lexists(self.directory/'FAILED.json'):
            raise RuntimeError('A reclaim failure marker forbids further advice')
        fd=os.open(self.directory/'CLAIM.json',os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
        try:
            stat=os.fstat(fd)
            if (stat.st_dev,stat.st_ino)!=self._claim_identity:
                raise RuntimeError('Exclusive reclaim claim identity changed')
            if stat.st_size!=self._claim_size:
                raise RuntimeError('Exclusive reclaim claim size changed')
            with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(self._claim_size+1)
            if _sha(raw)!=self._claim_sha256:
                raise RuntimeError('Exclusive reclaim claim contents changed')
        finally:os.close(fd)

    def next_batch(self,*,model=MODEL,max_bytes=MAX_BATCH):
        with self._lock:
            return self._next_batch_once(model=model,max_bytes=max_bytes)

    def _next_batch_once(self,*,model=MODEL,max_bytes=MAX_BATCH):
        if self._failed:
            raise RuntimeError('Reclaim cursor failed; its exclusive claim forbids all replay')
        try:
            self._verify_owner_claim()
            return self._next_batch(model=model,max_bytes=max_bytes)
        except BaseException as error:
            # Some irreversible advice calls may have succeeded before the
            # exception and before normal accounting or the receipt. Neither
            # this object nor a new object may retry ambiguous work.
            self._failed=True
            try:
                with (self.directory/'FAILED.json').open('x') as stream:
                    json.dump(dict(schema='q38-reclaim-cursor-failed-no-replay-v1',
                        run_id=self.plan['run_id'],hostname=self.plan['hostname'],
                        coordinator_pid=self._owner_pid,failure_observer_pid=os.getpid(),completed_rounds=self.round,
                        completed_syscall_advised_bytes=self.total,
                        partial_current_batch_may_have_executed=True,
                        failure_type=type(error).__name__,failed_unix=time.time()),stream,indent=2)
                    stream.write('\n')
            except BaseException:
                # The original exclusive claim and in-memory poison remain;
                # failure to record a receipt must never make retry legal.
                pass
            raise

    def _next_batch(self,*,model=MODEL,max_bytes=MAX_BATCH):
        if (not isinstance(max_bytes,int) or isinstance(max_bytes,bool) or max_bytes>MAX_BATCH
                or max_bytes<2*self.plan['page_size'] or max_bytes%self.plan['page_size']):
            raise ValueError('Invalid per-batch advice limit')
        if self.max_total-self.total<2*self.plan['page_size'] or (self.index>=len(self.segments) and not self.pending_resident):return None
        budget=min(max_bytes,self.max_total-self.total)
        if self.index<self.exact_count:
            remaining=[]
            # Never enter the wider stage in the same batch: the caller must
            # remeasure the actual cgroup current first.
            for index in range(self.index,self.exact_count):
                name,start,end=self.segments[index]
                if index==self.index and self.cursor is not None:start=self.cursor
                if start<end:remaining.append((name,start,end))
            result=advise_one_batch(self.plan,remaining,model=model,max_bytes=budget)
            if not result['advised_bytes']:raise RuntimeError('No progress within exact source plan')
            for record in result['ranges']:
                name,start,end=self.segments[self.index]
                expected=start if self.cursor is None else self.cursor
                if record['file']!=name or record['offset']!=expected:raise RuntimeError('Non-monotonic advice result')
                self.cursor=expected+record['length']
                if self.cursor==end:self.index+=1;self.cursor=None
                elif self.cursor>end:raise RuntimeError('Advice escaped source interval')
            result.update(stage='exact_completed_source_ranges',wide_scanned_bytes=0)
        else:
            scanned=0;windows=0
            if not self.pending_resident:
                # Advance scan position only after each mincore succeeds. Keep
                # selected runs queued until fully advised; no window is ever
                # rescanned or crossed into a different planned interval.
                while (self.index<len(self.segments) and scanned<MAX_SCAN_WINDOW
                       and windows<MAX_WIDE_WINDOWS_PER_BATCH):
                    name,start,end=self.segments[self.index]
                    start=start if self.cursor is None else self.cursor
                    window=min(end-start,MAX_SCAN_WINDOW-scanned,
                               self.wide_scan_limit-self.wide_scanned)
                    window=(window//self.plan['page_size'])*self.plan['page_size']
                    if not window:raise RuntimeError('Invalid wider scan progress')
                    selected=resident_runs(self.plan,name,start,start+window,model=model)
                    previous=start
                    for selected_name,left,right in selected:
                        if (selected_name!=name or left<previous or left>=right or right>start+window
                                or left%self.plan['page_size'] or right%self.plan['page_size']):
                            raise RuntimeError('Resident selector escaped or reordered its authorized window')
                        previous=right
                    self.pending_resident.extend(selected)
                    self.wide_scanned+=window;scanned+=window;windows+=1
                    self.cursor=start+window
                    if self.cursor==end:self.index+=1;self.cursor=None
                    elif self.cursor>end:raise RuntimeError('Wider scan escaped policy tensor')
            result=advise_one_batch(self.plan,self.pending_resident,model=model,max_bytes=budget)
            for record in result['ranges']:
                n,left,right=self.pending_resident[0]
                if record['file']!=n or record['offset']!=left:raise RuntimeError('Non-monotonic resident advice')
                consumed=left+record['length']
                if consumed==right:self.pending_resident.popleft()
                elif consumed<right:self.pending_resident[0]=(n,consumed,right)
                else:raise RuntimeError('Advice escaped selected resident run')
            result.update(stage='wider_closed_policy_tensor_pages',wide_scanned_bytes=scanned,
                wide_scanned_windows=windows,wide_windows_limit=MAX_WIDE_WINDOWS_PER_BATCH,
                resident_selector_is_not_reclaim_proof=True,shared_cache_IO_impact_possible=True)
        self.total+=result['syscall_advised_bytes'];self.round+=1
        result.update(run_id=self.plan['run_id'],round=self.round,total_advised_bytes=self.total,
                      next_interval=self.index,next_offset=self.cursor,coordinator_pid=os.getpid())
        # If the process dies before this receipt is written, the original
        # claim forbids automatic replay. Filesystem crash persistence is not
        # inferred from close alone; uncertain node loss needs external audit.
        with (self.directory/('batch-%05d.json'%self.round)).open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')
        return result
