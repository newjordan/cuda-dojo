/* ═══════════════════════════════════════════════════════════════════════════
 * frostmatrix_eval.cuh — FrostMatrix V3 FULL evaluator for gpu_forge.cu
 * v2: fixed pos-encoding (learned table), cross-attn layout, highway Wo,
 *     and multi-head attention (proper kernel replacing broken stride cuBLAS).
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef FROSTMATRIX_EVAL_CUH
#define FROSTMATRIX_EVAL_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define FM_N_FAMILIES     33
#define FM_N_UNKNOWNS     32
#define FM_N_FAM_NODES    65
#define FM_N_BOARD_SQ     64
#define FM_SEQ_LEN        129
#define FM_D_MODEL        128
#define FM_N_LAYERS       4
#define FM_N_HEADS        4
#define FM_D_HEAD         32
#define FM_D_FFN          341
#define FM_PIECE_VOCAB    13
#define FM_POLICY_DIM     64
#define FM_N_HW_DIRS      3
#define FM_D_HW_HEAD      32

#define FM_MAX_BATCH      256

#define FM_CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) fprintf(stderr, "[frostmatrix] CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) fprintf(stderr, "[frostmatrix] cuBLAS error %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
} while(0)

/* ── GLOBALS ──────────────────────────────────────────────────────────── */
static cublasHandle_t fm_cublas = NULL;
static bool fm_loaded = false;

// Weights
static float *fm_token_embed=NULL, *fm_board_pos_embed=NULL;
static float *fm_policy_from_W=NULL, *fm_policy_to_W=NULL, *fm_value_W=NULL, *fm_value_b=NULL;
static float *fm_Wq[FM_N_LAYERS],*fm_Wk[FM_N_LAYERS],*fm_Wv[FM_N_LAYERS],*fm_Wo[FM_N_LAYERS];
static float *fm_ffn_up[FM_N_LAYERS],*fm_ffn_gate[FM_N_LAYERS],*fm_ffn_down[FM_N_LAYERS];
static float *fm_ln1_w[FM_N_LAYERS],*fm_ln1_b[FM_N_LAYERS],*fm_ln2_w[FM_N_LAYERS],*fm_ln2_b[FM_N_LAYERS];
static float *fm_family_embed=NULL, *fm_unknown_embed=NULL, *fm_smooth_weights=NULL;
static float *fm_hw_Wq[FM_N_HW_DIRS],*fm_hw_Wk[FM_N_HW_DIRS],*fm_hw_Wv[FM_N_HW_DIRS],*fm_hw_Wo=NULL;
static float *fm_ca_Wq=NULL,*fm_ca_Wk=NULL,*fm_ca_Wv=NULL,*fm_ca_Wo=NULL;

// Workspace
float *fm_buf_x=NULL,*fm_buf_x_norm=NULL,*fm_buf_fam=NULL;
float *fm_buf_q=NULL,*fm_buf_k=NULL,*fm_buf_v=NULL;
float *fm_buf_attn_out=NULL,*fm_buf_ffn_h=NULL,*fm_buf_ffn_g=NULL;
float *fm_buf_pooled=NULL,*fm_buf_from=NULL,*fm_buf_to=NULL,*fm_buf_value=NULL;
int   *fm_buf_tokens=NULL;
float *fm_hw_q[FM_N_HW_DIRS],*fm_hw_k[FM_N_HW_DIRS],*fm_hw_v[FM_N_HW_DIRS],*fm_hw_attn=NULL;

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS — Sequence Construction
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void fm_build_family_axis_kernel(float *fam, const float *fe, const float *ue, int B) {
    int tok=blockIdx.x, b=blockIdx.y, d=threadIdx.x;
    if(b>=B||tok>=FM_N_FAM_NODES||d>=FM_D_MODEL) return;
    int is_unk=tok%2, idx=tok/2; float v=0;
    if(is_unk&&idx<FM_N_UNKNOWNS) v=ue[idx*FM_D_MODEL+d];
    else if(!is_unk&&idx<FM_N_FAMILIES) v=fe[idx*FM_D_MODEL+d];
    fam[(b*FM_N_FAM_NODES+tok)*FM_D_MODEL+d]=v;
}
__global__ void fm_embed_board_tokens_kernel(float *x, const int *tokens, const float *te, int B) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x, total=B*FM_N_BOARD_SQ*FM_D_MODEL;
    if(idx>=total) return;
    int b=idx/(FM_N_BOARD_SQ*FM_D_MODEL), sq=(idx/FM_D_MODEL)%FM_N_BOARD_SQ, d=idx%FM_D_MODEL;
    int seq=FM_N_FAM_NODES+sq, piece=tokens[b*FM_N_BOARD_SQ+sq];
    x[(b*FM_SEQ_LEN+seq)*FM_D_MODEL+d]=(piece>=0&&piece<FM_PIECE_VOCAB)?te[piece*FM_D_MODEL+d]:0;
}
__global__ void fm_board_pos_encode_kernel(float *x, const float *bpe, int B) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x, total=B*FM_N_BOARD_SQ*FM_D_MODEL;
    if(idx>=total) return;
    int b=idx/(FM_N_BOARD_SQ*FM_D_MODEL), sq=(idx/FM_D_MODEL)%FM_N_BOARD_SQ, d=idx%FM_D_MODEL;
    x[(b*FM_SEQ_LEN+FM_N_FAM_NODES+sq)*FM_D_MODEL+d]+=bpe[sq*FM_D_MODEL+d];
}
__global__ void fm_scatter_family_axis_kernel(float *x, const float *fam, int B) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x, total=B*FM_N_FAM_NODES*FM_D_MODEL;
    if(idx>=total) return;
    int b=idx/(FM_N_FAM_NODES*FM_D_MODEL), local=idx%(FM_N_FAM_NODES*FM_D_MODEL);
    x[(b*FM_SEQ_LEN+local/FM_D_MODEL)*FM_D_MODEL+local%FM_D_MODEL]=fam[idx];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS — Highway Attention
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void fm_highway_attn_scores_kernel(float *scores, const float *q, const float *k, const float *sw, int dir, int B) {
    int b=blockIdx.z, qi=blockIdx.y, ki=blockIdx.x, d=threadIdx.x;
    if(b>=B||qi>=FM_N_FAM_NODES||ki>=FM_N_FAM_NODES||d>=FM_D_HW_HEAD) return;
    bool qf=qi%2==0,kf=ki%2==0; int qid=qi/2,kid=ki/2; bool ok=false;
    if(dir==0) ok=qf&&kf;
    else if(dir==1){if(qf&&!kf) ok=(kid==qid-1||kid==qid)&&kid>=0&&kid<FM_N_UNKNOWNS; else if(!qf&&kf) ok=(kid==qid||kid==qid+1)&&kid<FM_N_FAMILIES;}
    else{if(qf&&!kf){int sk=abs(kid-qid); ok=(sk==2||sk==3)&&kid>=0&&kid<FM_N_UNKNOWNS;}}
    if(!ok){if(d==0)scores[(b*FM_N_FAM_NODES+qi)*FM_N_FAM_NODES+ki]=-1e9f; return;}
    float val=q[(b*FM_N_FAM_NODES+qi)*FM_D_HW_HEAD+d]*k[(b*FM_N_FAM_NODES+ki)*FM_D_HW_HEAD+d];
    for(int s=16;s>=1;s>>=1) val+=__shfl_down_sync(0xffffffff,val,s);
    if(d==0){float sc=val/sqrtf((float)FM_D_HW_HEAD);
    if(dir==0&&qf&&kf) sc+=logf(sw[qid*FM_N_FAMILIES+kid]+1e-8f);
    scores[(b*FM_N_FAM_NODES+qi)*FM_N_FAM_NODES+ki]=sc;}
}
__global__ void fm_hw_softmax_v_kernel(float *out, const float *scores, const float *v, int B) {
    int b=blockIdx.y, qi=blockIdx.x, d=threadIdx.x;
    if(b>=B||qi>=FM_N_FAM_NODES||d>=FM_D_HW_HEAD) return;
    const float *row=scores+(b*FM_N_FAM_NODES+qi)*FM_N_FAM_NODES;
    float mx=-1e9f; for(int j=0;j<FM_N_FAM_NODES;j++) mx=fmaxf(mx,row[j]);
    float es=0; for(int j=0;j<FM_N_FAM_NODES;j++) es+=expf(row[j]-mx);
    float acc=0; for(int j=0;j<FM_N_FAM_NODES;j++){float a=expf(row[j]-mx)/(es+1e-8f); acc+=a*v[(b*FM_N_FAM_NODES+j)*FM_D_HW_HEAD+d];}
    out[(b*FM_N_FAM_NODES+qi)*FM_D_HW_HEAD+d]=acc;
}
__global__ void fm_combine_highway_kernel(float *fam, const float *d0, const float *d1, const float *d2, const float *Wo, int B) {
    int tok=blockIdx.x, b=blockIdx.y, d=threadIdx.x;
    if(b>=B||tok>=FM_N_FAM_NODES||d>=FM_D_MODEL) return;
    int base=(b*FM_N_FAM_NODES+tok)*FM_D_HW_HEAD, Wo_cat=FM_N_HW_DIRS*FM_D_HW_HEAD;
    float acc=0;
    for(int h=0;h<FM_D_HW_HEAD;h++) acc+=d0[base+h]*Wo[d*Wo_cat+h];
    for(int h=0;h<FM_D_HW_HEAD;h++) acc+=d1[base+h]*Wo[d*Wo_cat+FM_D_HW_HEAD+h];
    for(int h=0;h<FM_D_HW_HEAD;h++) acc+=d2[base+h]*Wo[d*Wo_cat+2*FM_D_HW_HEAD+h];
    fam[(b*FM_N_FAM_NODES+tok)*FM_D_MODEL+d]+=acc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS — Cross-Attention
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void fm_board_cross_attn_kernel(float *x, const float *fam,
    const float *Wq,const float *Wk,const float *Wv,const float *Wo, int B) {
    int b=blockIdx.y, sq=blockIdx.x, d=threadIdx.x;
    if(b>=B||sq>=FM_N_BOARD_SQ||d>=FM_D_MODEL) return;
    int lane=d&31, warp=d>>5;
    __shared__ float qv[FM_D_MODEL], av[FM_D_MODEL], sb[FM_N_FAM_NODES], ws[4];
    float *bt=x+((b*FM_SEQ_LEN)+FM_N_FAM_NODES+sq)*FM_D_MODEL;
    float qd=0; for(int j=0;j<FM_D_MODEL;j++) qd+=bt[j]*Wq[d*FM_D_MODEL+j];
    qv[d]=qd; __syncthreads();
    for(int fi=0;fi<FM_N_FAM_NODES;fi++){
        const float *fx=fam+(b*FM_N_FAM_NODES+fi)*FM_D_MODEL;
        float kd=0; for(int j=0;j<FM_D_MODEL;j++) kd+=fx[j]*Wk[d*FM_D_MODEL+j];
        float p=qv[d]*kd; for(int o=16;o>0;o>>=1)p+=__shfl_down_sync(0xffffffff,p,o);
        if(lane==0) ws[warp]=p; __syncthreads();
        if(d==0){float t=0;for(int w=0;w<4;w++)t+=ws[w]; sb[fi]=t/sqrtf((float)FM_D_MODEL);}
        __syncthreads();
    }
    if(d==0){float mx=-1e30f; for(int fi=0;fi<FM_N_FAM_NODES;fi++)mx=fmaxf(mx,sb[fi]);
        float es=0; for(int fi=0;fi<FM_N_FAM_NODES;fi++){sb[fi]=expf(sb[fi]-mx);es+=sb[fi];}
        for(int fi=0;fi<FM_N_FAM_NODES;fi++) sb[fi]/=(es+1e-8f);}
    __syncthreads();
    float ad=0; for(int fi=0;fi<FM_N_FAM_NODES;fi++){
        const float *fx=fam+(b*FM_N_FAM_NODES+fi)*FM_D_MODEL;
        float vd=0; for(int j=0;j<FM_D_MODEL;j++) vd+=fx[j]*Wv[d*FM_D_MODEL+j];
        ad+=sb[fi]*vd;
    }
    av[d]=ad; __syncthreads();
    float proj=0; for(int j=0;j<FM_D_MODEL;j++) proj+=av[j]*Wo[d*FM_D_MODEL+j];
    bt[d]+=proj;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * KERNELS — Transformer Layer
 * ═══════════════════════════════════════════════════════════════════════════ */
__global__ void fm_layernorm_kernel(float *out, const float *x, const float *w, const float *b, int S, int D, int B) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=B*S) return;
    const float *xi=x+idx*D; float *oi=out+idx*D;
    float m=0,v=0; for(int d=0;d<D;d++)m+=xi[d]; m/=D;
    for(int d=0;d<D;d++){float t=xi[d]-m;v+=t*t;}
    float is=1.0f/sqrtf(v/D+1e-5f);
    for(int d=0;d<D;d++) oi[d]=(xi[d]-m)*is*w[d]+b[d];
}

__global__ void fm_mha_forward_kernel(float *out, const float *q, const float *k, const float *v, int S, int D) {
    int h=blockIdx.x/S, qi=blockIdx.x%S, b=blockIdx.y, lane=threadIdx.x;
    int ho=h*FM_D_HEAD, BS=b*S;
    float qd=q[BS*D+qi*D+ho+lane];
    extern __shared__ float sm[];
    float *scores=sm;
    float is=rsqrtf((float)FM_D_HEAD);
    for(int ki=0;ki<S;ki++){
        float kd=k[BS*D+ki*D+ho+lane];
        float dot=qd*kd;
        for(int o=FM_D_HEAD/2;o>0;o>>=1) dot+=__shfl_down_sync(0xffffffff,dot,o);
        if(lane==0) scores[ki]=dot*is;
    }
    __syncthreads();
    if(lane==0){
        float mx=-1e30f; for(int ki=0;ki<S;ki++) mx=fmaxf(mx,scores[ki]);
        float sm2=0; for(int ki=0;ki<S;ki++){scores[ki]=expf(scores[ki]-mx);sm2+=scores[ki];}
        for(int ki=0;ki<S;ki++) scores[ki]/=(sm2+1e-8f);
    }
    __syncthreads();
    float acc=0;
    for(int ki=0;ki<S;ki++) acc+=scores[ki]*v[BS*D+ki*D+ho+lane];
    out[BS*D+qi*D+ho+lane]=acc;
}

__global__ void fm_softmax_kernel(float *x, int B, int n){
    int b=blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5), lane=threadIdx.x&31;
    if(b>=B) return; float *row=x+b*n;
    float mx=-1e30f; for(int i=lane;i<n;i+=32) mx=fmaxf(mx,row[i]);
    for(int o=16;o>0;o>>=1) mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));
    mx=__shfl_sync(0xffffffff,mx,0);
    float sm=0; for(int i=lane;i<n;i+=32){float v=expf(row[i]-mx);row[i]=v;sm+=v;}
    for(int o=16;o>0;o>>=1) sm+=__shfl_down_sync(0xffffffff,sm,o);
    sm=__shfl_sync(0xffffffff,sm,0);
    for(int i=lane;i<n;i+=32) row[i]/=(sm+1e-8f);
}

__global__ void fm_swiglu_smeargate_kernel(float *out, const float *up, const float *gate, int S, int Dff, int B) {
    int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=B*S*Dff) return;
    int fi=idx%Dff, ti=idx/Dff;
    float u=up[ti*Dff+fi], g=gate[ti*Dff+fi], silu=g/(1.0f+expf(-g)), h=u*silu;
    int lane=fi&31; float nb=__shfl_down_sync(0xffffffff,h,1);
    if(lane==31) nb=h;
    float smear=(h*nb>0)?sqrtf(fabsf(h*nb))*(h>0?1:-1):0;
    out[ti*Dff+fi]=smear;
}

__global__ void fm_residual_add_kernel(float *x, const float *delta, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=delta[i];
}

__global__ void fm_mean_pool_board_kernel(float *pooled, const float *x, int B) {
    int b=blockIdx.x, d=threadIdx.x; if(b>=B||d>=FM_D_MODEL) return;
    float acc=0; for(int sq=0;sq<FM_N_BOARD_SQ;sq++) acc+=x[((b*FM_SEQ_LEN)+FM_N_FAM_NODES+sq)*FM_D_MODEL+d];
    pooled[b*FM_D_MODEL+d]=acc/(float)FM_N_BOARD_SQ;
}

__global__ void fm_value_head_kernel(float *val, const float *pooled, const float *w, const float *b, int B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=B) return;
    float v=b[0]; for(int j=0;j<FM_D_MODEL;j++) v+=pooled[i*FM_D_MODEL+j]*w[j];
    val[i]=tanhf(v);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * WEIGHT LOADING
 * ═══════════════════════════════════════════════════════════════════════════ */
static bool fm_file_exists(const char *p){struct stat st; return stat(p,&st)==0;}

static float* fm_load_binary(const char *path, int expected, const char *name){
    if(!fm_file_exists(path)){fprintf(stderr,"[frostmatrix] weight not found: %s\n",path); return NULL;}
    FILE*f=fopen(path,"rb"); if(!f){fprintf(stderr,"[frostmatrix] cannot open %s\n",path); return NULL;}
    fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
    int nf=(int)(sz/sizeof(float)); float*h=(float*)malloc(nf*sizeof(float));
    (void)!fread(h,sizeof(float),nf,f); fclose(f);
    float*d=NULL; FM_CUDA_CHECK(cudaMalloc(&d,nf*sizeof(float)));
    FM_CUDA_CHECK(cudaMemcpy(d,h,nf*sizeof(float),cudaMemcpyHostToDevice));
    free(h); fprintf(stderr,"[frostmatrix] loaded %-20s %d floats\n",name,nf);
    return d;
}

static bool frostmatrix_init(const char *dir){
    if(fm_loaded) return true;
    if(cublasCreate(&fm_cublas)!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"[frostmatrix] cublasCreate failed\n"); return false;}
    char p[1024];
    #define FLD(fname,var,ex) do{snprintf(p,sizeof(p),"%s/%s",dir,fname); var=fm_load_binary(p,ex,fname);}while(0)
    FLD("token_embed.bin",fm_token_embed,FM_PIECE_VOCAB*FM_D_MODEL);
    FLD("board_pos_embed.bin",fm_board_pos_embed,FM_N_BOARD_SQ*FM_D_MODEL);
    FLD("policy_from_W.bin",fm_policy_from_W,FM_POLICY_DIM*FM_D_MODEL);
    FLD("policy_to_W.bin",fm_policy_to_W,FM_POLICY_DIM*FM_D_MODEL);
    FLD("value_W.bin",fm_value_W,FM_D_MODEL);
    FLD("value_b.bin",fm_value_b,1);
    for(int l=0;l<FM_N_LAYERS;l++){char b[64];
        snprintf(b,sizeof(b),"Wq_%d.bin",l); FLD(b,fm_Wq[l],FM_D_MODEL*FM_D_MODEL);
        snprintf(b,sizeof(b),"Wk_%d.bin",l); FLD(b,fm_Wk[l],FM_D_MODEL*FM_D_MODEL);
        snprintf(b,sizeof(b),"Wv_%d.bin",l); FLD(b,fm_Wv[l],FM_D_MODEL*FM_D_MODEL);
        snprintf(b,sizeof(b),"Wo_%d.bin",l); FLD(b,fm_Wo[l],FM_D_MODEL*FM_D_MODEL);
        snprintf(b,sizeof(b),"ffn_up_%d.bin",l); FLD(b,fm_ffn_up[l],FM_D_FFN*FM_D_MODEL);
        snprintf(b,sizeof(b),"ffn_gate_%d.bin",l); FLD(b,fm_ffn_gate[l],FM_D_FFN*FM_D_MODEL);
        snprintf(b,sizeof(b),"ffn_down_%d.bin",l); FLD(b,fm_ffn_down[l],FM_D_MODEL*FM_D_FFN);
        snprintf(b,sizeof(b),"ln1_w_%d.bin",l); FLD(b,fm_ln1_w[l],FM_D_MODEL);
        snprintf(b,sizeof(b),"ln1_b_%d.bin",l); FLD(b,fm_ln1_b[l],FM_D_MODEL);
        snprintf(b,sizeof(b),"ln2_w_%d.bin",l); FLD(b,fm_ln2_w[l],FM_D_MODEL);
        snprintf(b,sizeof(b),"ln2_b_%d.bin",l); FLD(b,fm_ln2_b[l],FM_D_MODEL);
    }
    FLD("fax_family_embed.bin",fm_family_embed,FM_N_FAMILIES*FM_D_MODEL);
    FLD("fax_unknown_embed.bin",fm_unknown_embed,FM_N_UNKNOWNS*FM_D_MODEL);
    FLD("fax_smooth_weights.bin",fm_smooth_weights,FM_N_FAMILIES*FM_N_FAMILIES);
    for(int d=0;d<FM_N_HW_DIRS;d++){char b[64];
        snprintf(b,sizeof(b),"fax_hw_Wq_%d.bin",d); FLD(b,fm_hw_Wq[d],FM_D_HW_HEAD*FM_D_MODEL);
        snprintf(b,sizeof(b),"fax_hw_Wk_%d.bin",d); FLD(b,fm_hw_Wk[d],FM_D_HW_HEAD*FM_D_MODEL);
        snprintf(b,sizeof(b),"fax_hw_Wv_%d.bin",d); FLD(b,fm_hw_Wv[d],FM_D_HW_HEAD*FM_D_MODEL);
    }
    FLD("fax_hw_Wo.bin",fm_hw_Wo,FM_N_HW_DIRS*FM_D_HW_HEAD*FM_D_MODEL);
    FLD("fax_ca_Wq.bin",fm_ca_Wq,FM_D_MODEL*FM_D_MODEL);
    FLD("fax_ca_Wk.bin",fm_ca_Wk,FM_D_MODEL*FM_D_MODEL);
    FLD("fax_ca_Wv.bin",fm_ca_Wv,FM_D_MODEL*FM_D_MODEL);
    FLD("fax_ca_Wo.bin",fm_ca_Wo,FM_D_MODEL*FM_D_MODEL);
    #undef FLD

    int B=FM_MAX_BATCH, S=FM_SEQ_LEN, D=FM_D_MODEL;
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_x,B*S*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_x_norm,B*S*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_fam,B*FM_N_FAM_NODES*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_q,B*S*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_k,B*S*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_v,B*S*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_attn_out,B*S*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_ffn_h,B*S*FM_D_FFN*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_ffn_g,B*S*FM_D_FFN*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_pooled,B*D*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_from,B*FM_POLICY_DIM*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_to,B*FM_POLICY_DIM*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_value,B*sizeof(float)));
    FM_CUDA_CHECK(cudaMalloc(&fm_buf_tokens,B*FM_N_BOARD_SQ*sizeof(int)));
    for(int d=0;d<FM_N_HW_DIRS;d++){
        FM_CUDA_CHECK(cudaMalloc(&fm_hw_q[d],B*FM_N_FAM_NODES*FM_D_HW_HEAD*sizeof(float)));
        FM_CUDA_CHECK(cudaMalloc(&fm_hw_k[d],B*FM_N_FAM_NODES*FM_D_HW_HEAD*sizeof(float)));
        FM_CUDA_CHECK(cudaMalloc(&fm_hw_v[d],B*FM_N_FAM_NODES*FM_D_HW_HEAD*sizeof(float)));
    }
    FM_CUDA_CHECK(cudaMalloc(&fm_hw_attn,B*FM_N_FAM_NODES*FM_N_FAM_NODES*sizeof(float)));

    if(fm_token_embed&&fm_policy_from_W&&fm_policy_to_W&&fm_value_W){
        fm_loaded=true; fprintf(stderr,"[frostmatrix] initialized from %s (%d weights)\n",dir,64);
    }else{fprintf(stderr,"[frostmatrix] FAILED — missing critical weights\n");}
    return fm_loaded;
}
static bool frostmatrix_available(void){return fm_loaded;}

/* ═══════════════════════════════════════════════════════════════════════════
 * PIECE ENCODING
 * ═══════════════════════════════════════════════════════════════════════════ */
static int fm_convert_piece(int gp) {
    // gpu_forge encoding: 0=empty, 1-6=white(P,N,B,R,Q,K), black non-sequential
    // FrostMatrix tokens:    0=empty, 1-6 white, 7-12 black(P,N,B,R,Q,K)
    // gpu_forge black: p=8, n=7, b=10, r=9, q=11, k=13
    switch (gp) {
        case 0:  return 0;   // empty
        case 1:  return 1;   // wP
        case 2:  return 2;   // wN
        case 3:  return 3;   // wB
        case 4:  return 4;   // wR
        case 5:  return 5;   // wQ
        case 6:  return 6;   // wK
        case 8:  return 7;   // bP
        case 7:  return 8;   // bN
        case 10: return 9;   // bB
        case 9:  return 10;  // bR
        case 11: return 11;  // bQ
        case 13: return 12;  // bK
        default: return 0;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FORWARD PASS
 * ═══════════════════════════════════════════════════════════════════════════ */
static void frostmatrix_forward(const int *h_tokens, int B, int fam_id,
    float *h_from, float *h_to, float *h_value)
{
    if(!fm_loaded||B>FM_MAX_BATCH) return;
    const float one=1.0f, zero=0.0f;
    int S=FM_SEQ_LEN, D=FM_D_MODEL, F=FM_N_FAM_NODES, BS=B*S;

    FM_CUDA_CHECK(cudaMemcpy(fm_buf_tokens,h_tokens,B*FM_N_BOARD_SQ*sizeof(int),cudaMemcpyHostToDevice));

    // Step 1: Build family axis
    dim3 fg(F,B); fm_build_family_axis_kernel<<<fg,D>>>(fm_buf_fam,fm_family_embed,fm_unknown_embed,B);

    // Step 2: Embed board tokens + learned positional encoding
    {int et=B*FM_N_BOARD_SQ*D; fm_embed_board_tokens_kernel<<<(et+255)/256,256>>>(fm_buf_x,fm_buf_tokens,fm_token_embed,B);
     fm_board_pos_encode_kernel<<<(et+255)/256,256>>>(fm_buf_x,fm_board_pos_embed,B);}

    // Step 3: Geo projection — DISABLED

    // Step 4: Highway attention
    for(int dir=0;dir<FM_N_HW_DIRS;dir++){
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,FM_D_HW_HEAD,B*F,D,&one,fm_hw_Wq[dir],D,fm_buf_fam,D,&zero,fm_hw_q[dir],FM_D_HW_HEAD));
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,FM_D_HW_HEAD,B*F,D,&one,fm_hw_Wk[dir],D,fm_buf_fam,D,&zero,fm_hw_k[dir],FM_D_HW_HEAD));
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,FM_D_HW_HEAD,B*F,D,&one,fm_hw_Wv[dir],D,fm_buf_fam,D,&zero,fm_hw_v[dir],FM_D_HW_HEAD));
        dim3 hsg(F,F,B); fm_highway_attn_scores_kernel<<<hsg,FM_D_HW_HEAD>>>(fm_hw_attn,fm_hw_q[dir],fm_hw_k[dir],fm_smooth_weights,dir,B);
        dim3 hvg(F,B); fm_hw_softmax_v_kernel<<<hvg,FM_D_HW_HEAD>>>(fm_hw_q[dir],fm_hw_attn,fm_hw_v[dir],B);
    }
    {dim3 cg(F,B); fm_combine_highway_kernel<<<cg,D>>>(fm_buf_fam,fm_hw_q[0],fm_hw_q[1],fm_hw_q[2],fm_hw_Wo,B);}

    // Step 5: Cross-attention
    {dim3 cag(FM_N_BOARD_SQ,B); fm_board_cross_attn_kernel<<<cag,D>>>(fm_buf_x,fm_buf_fam,fm_ca_Wq,fm_ca_Wk,fm_ca_Wv,fm_ca_Wo,B);}
    {int ft=B*F*D; fm_scatter_family_axis_kernel<<<(ft+255)/256,256>>>(fm_buf_x,fm_buf_fam,B);}

    // Step 6: Transformer layers
    for(int l=0;l<FM_N_LAYERS;l++){
        int lb=(BS+255)/256;
        fm_layernorm_kernel<<<lb,256>>>(fm_buf_x_norm,fm_buf_x,fm_ln1_w[l],fm_ln1_b[l],S,D,B);
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,D,BS,D,&one,fm_Wq[l],D,fm_buf_x_norm,D,&zero,fm_buf_q,D));
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,D,BS,D,&one,fm_Wk[l],D,fm_buf_x_norm,D,&zero,fm_buf_k,D));
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,D,BS,D,&one,fm_Wv[l],D,fm_buf_x_norm,D,&zero,fm_buf_v,D));
        {dim3 mg(FM_N_HEADS*S,B); fm_mha_forward_kernel<<<mg,FM_D_HEAD, (S+FM_D_HEAD)*sizeof(float)>>>(fm_buf_attn_out,fm_buf_q,fm_buf_k,fm_buf_v,S,D);}
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,D,BS,D,&one,fm_Wo[l],D,fm_buf_attn_out,D,&zero,fm_buf_q,D));
        int nr=B*S*D; fm_residual_add_kernel<<<(nr+255)/256,256>>>(fm_buf_x,fm_buf_q,nr);
        fm_layernorm_kernel<<<lb,256>>>(fm_buf_x_norm,fm_buf_x,fm_ln2_w[l],fm_ln2_b[l],S,D,B);
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,FM_D_FFN,BS,D,&one,fm_ffn_up[l],D,fm_buf_x_norm,D,&zero,fm_buf_ffn_h,FM_D_FFN));
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,FM_D_FFN,BS,D,&one,fm_ffn_gate[l],D,fm_buf_x_norm,D,&zero,fm_buf_ffn_g,FM_D_FFN));
        int nf=B*S*FM_D_FFN; fm_swiglu_smeargate_kernel<<<(nf+255)/256,256>>>(fm_buf_ffn_h,fm_buf_ffn_h,fm_buf_ffn_g,S,FM_D_FFN,B);
        CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,D,BS,FM_D_FFN,&one,fm_ffn_down[l],FM_D_FFN,fm_buf_ffn_h,FM_D_FFN,&zero,fm_buf_attn_out,D));
        fm_residual_add_kernel<<<(nr+255)/256,256>>>(fm_buf_x,fm_buf_attn_out,nr);
    }

    // Step 7: Heads
    fm_mean_pool_board_kernel<<<B,D>>>(fm_buf_pooled,fm_buf_x,B);
    CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,FM_POLICY_DIM,B,D,&one,fm_policy_from_W,D,fm_buf_pooled,D,&zero,fm_buf_from,FM_POLICY_DIM));
    CUBLAS_CHECK(cublasSgemm(fm_cublas,CUBLAS_OP_T,CUBLAS_OP_N,FM_POLICY_DIM,B,D,&one,fm_policy_to_W,D,fm_buf_pooled,D,&zero,fm_buf_to,FM_POLICY_DIM));
    fm_softmax_kernel<<<B,32>>>(fm_buf_from,B,FM_POLICY_DIM);
    fm_softmax_kernel<<<B,32>>>(fm_buf_to,B,FM_POLICY_DIM);
    int vb=(B+255)/256; fm_value_head_kernel<<<vb,256>>>(fm_buf_value,fm_buf_pooled,fm_value_W,fm_value_b,B);
    FM_CUDA_CHECK(cudaDeviceSynchronize());
    FM_CUDA_CHECK(cudaMemcpy(h_from,fm_buf_from,B*FM_POLICY_DIM*sizeof(float),cudaMemcpyDeviceToHost));
    FM_CUDA_CHECK(cudaMemcpy(h_to,fm_buf_to,B*FM_POLICY_DIM*sizeof(float),cudaMemcpyDeviceToHost));
    FM_CUDA_CHECK(cudaMemcpy(h_value,fm_buf_value,B*sizeof(float),cudaMemcpyDeviceToHost));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CONVENIENCE
 * ═══════════════════════════════════════════════════════════════════════════ */
static float frostmatrix_single_eval(const int pieces[64], int stm) {
    if(!fm_loaded) return 0;
    int tokens[64]; for(int sq=0;sq<64;sq++) tokens[sq]=fm_convert_piece(pieces[sq]);
    float from[64],to[64],val; frostmatrix_forward(tokens,1,0,from,to,&val);
    return stm==0?val:-val;
}
#endif
