#pragma once
// V4.1 BF16 RMSNorm -> RoPE64 -> BF16 -> native MXFP8 cache528.
// Independent of Q GEMM; side warps own grid-stride KV rows.
namespace rtp_dsv41_qb {
struct SideArgs {
    bool wait_pdl=false;
    bool trigger_pdl=false;
    bool trap_invalid=false;
    const __nv_bfloat16* kv=nullptr;
    const __nv_bfloat16* gamma=nullptr;
    const int64_t* pos64=nullptr;
    const int* pos32=nullptr;
    const int64_t* slots64=nullptr;
    const int* slots32=nullptr;
    const float* freqs=nullptr; // interleaved [cos,sin], same storage as RTP complex64
    uint8_t* pool=nullptr;
    __nv_bfloat16* kv_out=nullptr;
    int64_t kv_stride=512,page_stride=0;
    int rows=0,pages=0,entries=64,num_freqs=0;
    const int* req=nullptr;
    const int* block_table=nullptr;
    const int* local_indices=nullptr;
    int* global_indices=nullptr;
    int topk=0,num_requests=0,blocks_per_req=0,logical_entries=0,index_entries=0;
    int64_t table_stride=0,local_stride=0;
    int logical_shift=-1,entries_shift=-1;
    float eps=1e-6f;
};

__device__ __forceinline__ void kv_row(const SideArgs& a,int row) {
    const int lane=threadIdx.x&31;
    const int64_t pos=a.pos64?a.pos64[row]:a.pos32[row];
    if(pos<0||pos>=a.num_freqs) {asm volatile("trap;");return;}
    // Emulate the reference's blocked<4,32,4> RMS reduction in one warp:
    // four contiguous values per virtual thread, XOR16..1 within each
    // virtual warp, then XOR2..1 across the four virtual warp totals.
    float v[16], partial[4];
    #pragma unroll
    for(int g=0;g<4;++g) {
        #pragma unroll
        for(int j=0;j<4;++j)
            v[g*4+j]=__bfloat162float(a.kv[int64_t(row)*a.kv_stride+g*128+lane*4+j]);
        float s=__fmul_rn(v[g*4+1],v[g*4+1]);
        s=__fmaf_rn(v[g*4],v[g*4],s);
        s=__fmaf_rn(v[g*4+2],v[g*4+2],s);
        s=__fmaf_rn(v[g*4+3],v[g*4+3],s);
        #pragma unroll
        for(int mask=16;mask;mask/=2)s=__fadd_rn(s,__shfl_xor_sync(0xffffffff,s,mask));
        partial[g]=s;
    }
    float sum=__fadd_rn(__fadd_rn(partial[0],partial[2]),__fadd_rn(partial[1],partial[3]));
    float variance=__fadd_rn(__fmul_rn(sum,1.0f/512),a.eps),inv;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(variance));
    #pragma unroll
    for(int g=0;g<4;++g) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int col=g*128+lane*4+j;
            v[g*4+j]=__fmul_rn(__fmul_rn(v[g*4+j],inv),__bfloat162float(a.gamma[col]));
        }
    }
    if(lane>=16) {
        #pragma unroll
        for(int j=0;j<4;j+=2) {
            const int pair=(lane-16)*2+j/2;
            float c=a.freqs[pos*64+pair*2],s=a.freqs[pos*64+pair*2+1];
            float real=v[12+j],imag=v[12+j+1];
            v[12+j]=__fmaf_rn(real,c,-__fmul_rn(imag,s));
            v[12+j+1]=__fmaf_rn(real,s,__fmul_rn(imag,c));
        }
    }
    #pragma unroll
    for(int g=0;g<4;++g) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            auto rounded=__float2bfloat16_rn(v[g*4+j]);
            a.kv_out[int64_t(row)*512+g*128+lane*4+j]=rounded;
            v[g*4+j]=__bfloat162float(rounded);
        }
    }
    int64_t slot=a.slots64?a.slots64[row]:a.slots32[row];
    if(slot<0)return;
    if(slot/a.entries>=a.pages){if(a.trap_invalid)asm volatile("trap;");return;}
    uint8_t* dst=a.pool+(slot/a.entries)*a.page_stride+(slot%a.entries)*528;
    #pragma unroll
    for(int g=0;g<4;++g) {
        float maximum=0;
        #pragma unroll
        for(int j=0;j<4;++j)maximum=fmaxf(maximum,fabsf(v[g*4+j]));
        #pragma unroll
        for(int mask=4;mask;mask/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,mask));
        maximum=fmaxf(maximum,1e-4f);
        int exponent=static_cast<int>(ceilf(log2f(maximum*(1.0f/448.0f))));
        #pragma unroll
        for(int j=0;j<4;++j) {
            float scaled=fminf(448.0f,fmaxf(-448.0f,v[g*4+j]*exp2f(-float(exponent))));
            dst[g*128+lane*4+j]=__nv_fp8_e4m3(scaled).__x;
        }
        if(lane%8==0)dst[512+g*4+lane/8]=static_cast<uint8_t>(max(0,min(255,exponent+127)));
    }
}


__device__ __forceinline__ void index_work(const SideArgs& a,int worker,int workers) {
    if(a.global_indices) {
        const int lane=threadIdx.x&31;
        for(int row=worker;row<a.rows;row+=workers) {
            const int req=a.req[row];
            const int64_t row_start=int64_t(row)*a.local_stride;
            const int64_t table_start=int64_t(req)*a.table_stride;
            // Eight independent lookups in flight per lane. Loading all tokens
            // first, then all table bases, avoids a serial token->table chain.
            for(int chunk=0;chunk<a.topk;chunk+=256) {
                int tok[8],block[8],base[8];bool valid[8];
                #pragma unroll
                for(int i=0;i<8;++i) {
                    int col=chunk+i*32+lane;
                    tok[i]=col<a.topk ? a.local_indices[row_start+col] : -1;
                }
                #pragma unroll
                for(int i=0;i<8;++i) {
                    block[i]=a.logical_shift>=0 ? tok[i]>>a.logical_shift : tok[i]/a.logical_entries;
                    valid[i]=tok[i]>=0&&req>=0&&req<a.num_requests&&block[i]>=0&&block[i]<a.blocks_per_req;
                    base[i]=valid[i]?a.block_table[table_start+block[i]]:0;
                }
                #pragma unroll
                for(int i=0;i<8;++i) {
                    int col=chunk+i*32+lane;
                    int offset=a.entries_shift>=0 ? tok[i]&(a.index_entries-1) : tok[i]%a.index_entries;
                    int out=valid[i]&&base[i]>0 ? base[i]*a.index_entries+offset : -1;
                    if(col<a.topk)a.global_indices[int64_t(row)*a.topk+col]=out;
                }
            }
        }
    }
}

} // namespace rtp_dsv41_qb
