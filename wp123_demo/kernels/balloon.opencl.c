// defaults from Stefan 2007, cf tvb/analyzers/fmri_balloon.py
#define TAU_S 0.65f
#define TAU_F 0.41f
#define TAU_O 0.98f
#define ALPHA 0.32f
#define TE 0.04f
#define V0 4.0f
#define E0 0.4f
#define EPSILON 0.5f
#define NU_0 40.3f
#define R_0 25.0f

#define RECIP_TAU_S (1.0f / TAU_S)
#define RECIP_TAU_F (1.0f / TAU_F)
#define RECIP_TAU_O (1.0f / TAU_O)
#define RECIP_ALPHA (1.0f / ALPHA)
#define RECIP_E0 (1.0f / E0)

// "derived parameters"
#define k1 (4.3f * NU_0 * E0 * TE)
#define k2 (EPSILON * R_0 * E0 * TE)
#define k3 (1.0f - EPSILON)

/*
#ifndef OPENCL_VERSION
#define __kernel
#define __global
#define get_global_id(x) 0
#define get_global_size(x) 1
#endif
*/

__kernel void balloon_dfun(
    __global float *g_dsfvq,
    __global float *g_sfvq, // bold state
    __global float *g_x	    // neural state
)
{
    // get grid dims & position
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes

    // offset global pointers to this node & lane
    __global float *sfvq = g_sfvq + i * L + l;
    __global float *dsfvq = g_dsfvq + i * L + l;

    // unpack state variables
    float s = sfvq[0*N*L];
    float f = sfvq[1*N*L];
    float v = sfvq[2*N*L];
    float q = sfvq[3*N*L];

    // unpack neural state
    float x = g_x[i*L + l];

    // compute derivatives
    float ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0f);
    float df = s;
    float dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA));
    float dq = RECIP_TAU_O * (f * (1.0f - pow(1.0f - E0, 1.0f / f)) 
		    * RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v));

    // pack derivatives
    dsfvq[0*N*L] = ds;
    dsfvq[1*N*L] = df;
    dsfvq[2*N*L] = dv;
    dsfvq[3*N*L] = dq;

} // kernel

__kernel void balloon_readout(
    __global float *g_sfvq, // bold state
    __global float *g_bold  // bold out
)
{
    // get grid dims & position
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes

    // offset global pointers to this node & lane
    __global float *sfvq = g_sfvq + i * L + l;
    __global float *bold = g_bold + i * L + l;

    // unpack state variables
    float v = sfvq[2*N*L];
    float q = sfvq[3*N*L];

    // compute bold
    bold[0] = V0*(k1*(1.0f - q) + k2*(1.0f - q / v) + k3*(1.0f - v));
}
