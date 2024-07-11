// predictor stage for Heun, same as Euler
__kernel void heun_pred(
    const float dt,
    __global float *out,
    __global float *x,
    __global float *dx,
    __global float *z
)
{
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes
    
    int idx = i*L + l;
    out[idx] = x[idx] + dt*dx[idx] + sqrt(dt)*z[idx];
}

// corrector stage for Heun
__kernel void heun_corr(
    const float dt,
    __global float *out,
    __global float *x,
    __global float *dx1,
    __global float *dx2,
    __global float *z
)
{
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes
    
    int idx = i*L + l;
    out[idx] = x[idx] + dt/2*(dx1[idx] + dx2[idx]) + sqrt(dt)*z[idx];
}
