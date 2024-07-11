
__kernel void stim(
    const int t,
    __global float *out,
    __global float *x,
    __global float *stim_t,
    __global float *stim_w
)
{
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes
    
    int idx = i*L + l;
    out[idx] = x[idx] + stim_w[idx]*stim_t[t]; 
}
