"""Reconstruct exact HF bytes read by the pinned Full12 BF16 loader.

Pure metadata arithmetic: no checkpoint payload, CUDA, cache advice or writes.
The expert and vocabulary rules mirror the locked production loader; an
unknown mapping is rejected instead of expanding to a whole tensor/file.
"""
import math,re

CHUNK_EXPERTS=16
VOCAB_ROWS=16384

def tensor_intervals(row,header,data_start,file_size):
    filename=row['checkpoint_file'];name=row['name'];canonical=row['canonical_name']
    if '/' in filename or not re.fullmatch(r'model-\d{5}-of-00213\.safetensors',filename):
        raise ValueError('Unexpected checkpoint filename')
    item=header[canonical]
    if item['dtype']!='BF16':raise ValueError('Only official BF16 tensor bytes are supported')
    shape=item['shape'];offsets=item['data_offsets']
    if not shape or any(not isinstance(x,int) or x<=0 for x in shape):raise ValueError('Invalid shape')
    if len(offsets)!=2 or offsets[0]<0 or offsets[1]-offsets[0]!=math.prod(shape)*2:
        raise ValueError('Inconsistent safetensors offsets')
    base=data_start+offsets[0];limit=data_start+offsets[1]
    if limit>file_size:raise ValueError('Tensor exceeds file')
    begin,end=row['source_begin'],row['source_end']
    if not isinstance(begin,int) or not isinstance(end,int) or not 0<=begin<=end:
        raise ValueError('Invalid source interval')
    spans=[]
    expert=re.fullmatch(r'(.+)\.chunks\.(\d+)\.(gate_up_proj|down_proj)',name)
    tile=re.fullmatch(r'(model\.embed_tokens|lm_head)\.tiles\.(\d+)\.weight',name)
    if expert:
        prefix,index,field=expert.groups();index=int(index)
        if canonical!=prefix+'.'+field or len(shape)!=3:raise ValueError('Unexpected expert mapping')
        first=index*CHUNK_EXPERTS;last=first+CHUNK_EXPERTS
        if last>shape[0] or end>CHUNK_EXPERTS*shape[1]:raise ValueError('Expert interval out of bounds')
        cursor=begin
        while cursor<end:
            expert_index,feature=divmod(cursor,shape[1]);take=min(end-cursor,shape[1]-feature)
            start=base+((first+expert_index)*shape[1]+feature)*shape[2]*2
            spans.append((start,start+take*shape[2]*2));cursor+=take
    elif tile:
        prefix,index=tile.groups();index=int(index)
        if canonical!=prefix+'.weight' or len(shape)!=2:raise ValueError('Unexpected vocab mapping')
        first=index*VOCAB_ROWS
        if first+end>shape[0] or end>VOCAB_ROWS:raise ValueError('Vocab interval out of bounds')
        width=shape[1]*2;spans=[(base+(first+begin)*width,base+(first+end)*width)]
    elif name==canonical:
        if end>shape[0]:raise ValueError('Ordinary axis0 interval out of bounds')
        width=math.prod(shape[1:])*2;spans=[(base+begin*width,base+end*width)]
    else:raise ValueError('Unknown mapped parameter: '+name)
    if any(a<base or b>limit or a>b for a,b in spans):raise ValueError('Derived span out of tensor')
    if sum(b-a for a,b in spans)!=row['local_bytes']:raise ValueError('Derived bytes differ from receipt')
    return [(filename,a,b) for a,b in spans if b>a]

def page_aligned_union(intervals,page):
    if not isinstance(page,int) or page<=0:raise ValueError('Invalid page size')
    by_file={}
    for filename,start,end in intervals:
        if not 0<=start<=end:raise ValueError('Invalid source byte interval')
        # Never advise a page that contains bytes outside an observed interval.
        start=((start+page-1)//page)*page;end=(end//page)*page
        if end>start:by_file.setdefault(filename,[]).append((start,end))
    result=[]
    for filename in sorted(by_file):
        merged=[]
        for start,end in sorted(by_file[filename]):
            if merged and start<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],end))
            else:merged.append((start,end))
        result.extend((filename,start,end) for start,end in merged)
    return result

def bounded_batch(intervals,page,max_bytes):
    if max_bytes<page or max_bytes%page:raise ValueError('Batch budget must be whole pages')
    output=[];total=0
    for filename,start,end in intervals:
        if start%page or end%page or end<start:raise ValueError('Unaligned interval')
        take=min(end-start,max_bytes-total)
        if take:output.append((filename,start,start+take));total+=take
        if total==max_bytes:break
    return output

def subtract_intervals(allowed,excluded):
    """Return exact allowed\excluded without adding gaps or overlapping bytes."""
    by_file={}
    for name,start,end in excluded:by_file.setdefault(name,[]).append((start,end))
    for value in by_file.values():value.sort()
    output=[]
    for name,start,end in allowed:
        cursor=start
        for left,right in by_file.get(name,[]):
            if right<=cursor:continue
            if left>=end:break
            if left>cursor:output.append((name,cursor,min(left,end)))
            cursor=max(cursor,right)
            if cursor>=end:break
        if cursor<end:output.append((name,cursor,end))
    return output
