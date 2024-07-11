__kernel void spmv(
    __global float *out,
    __global float *vec,
    __global float *data,
    __global int *indices,
    __global int *indptr
)
{
    int i = get_global_id(0); // node
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes
    
    float acc = 0.0f;
    for (int j=indptr[i]; j<indptr[i+1]; j++)
        acc += data[j] * vec[indices[j]*L + l];
    
    out[i*L + l] = acc;
}
