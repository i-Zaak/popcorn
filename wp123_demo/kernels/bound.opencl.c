// this should be fused with the integrator if the overhead is too big 
__kernel void lower_bound(
    __global float *x,
    const float lb
)
{
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes
    
    int idx = i*L + l;
    if (x[idx] < lb){
	x[idx] = lb;
    }
}

