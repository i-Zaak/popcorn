// Parameters below as in Goldman 2020
//
// sweeping parameters
#pragma OPENCL EXTENSION cl_intel_printf : enable

#define b_e (params[i*L + l])


// global parameters
#define g_L         10.0f
#define E_L_e       -63.0f
#define E_L_i       -65.0f
#define C_m         200.0f
//#define b_e         1.0f //sweeping this
#define a_e         0.0f
#define b_i         0.0f
#define a_i         0.0f
#define tau_w_e     500.0f
#define tau_w_i     1.0f
#define E_e         0.0f
#define E_i         -80.0f
#define Q_e         1.5f
#define Q_i         5.0f
#define tau_e       5.0f
#define tau_i       5.0f
#define N_tot       10000.0f
#define p_connect_e 0.05f
#define p_connect_i 0.05f
#define g           0.2f
#define K_ext_e     400.0f
#define K_ext_i     0.0f
#define T           40.0f

#define P0_e        -0.0498f
#define P1_e        0.00506f
#define P2_e        -0.025f
#define P3_e        0.0014f
#define P4_e        -0.00041f
#define P5_e        0.0105f
#define P6_e        -0.036f
#define P7_e        0.0074f
#define P8_e        0.0012f
#define P9_e        -0.0407f

#define P0_i        -0.0514f
#define P1_i         0.004f
#define P2_i         -0.0083f
#define P3_i         0.0002f
#define P4_i         -0.0005f
#define P5_i         0.0014f
#define P6_i         -0.0146f
#define P7_i         0.0045f
#define P8_i         0.0028f
#define P9_i         -0.0153f

#define external_input_ex_ex 0.315e-3f
#define external_input_ex_in 0.0f
#define external_input_in_ex 0.315e-3f
#define external_input_in_in 0.0f
#define tau_OU               5.0f
#define weight_noise         0.0001f
#define S_i                  1.2f

// helper functions

float square(float x) {return x * x;}

void get_fluct_regime_vars(float Fe, float Fi, float Fe_ext, float Fi_ext, float W, float E_L, float *mu_V, float *sigma_V, float *T_V){
    //constants: Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, N_tot, p_connect_e, p_connect_i, g, K_ext_e, K_ext_i
    /*
    Compute the mean characteristic of neurons.
    Inspired from the next repository :
    https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
    :param Fe: firing rate of excitatory population
    :param Fi: firing rate of inhibitory population
    :param Fe_ext: external excitatory input
    :param Fi_ext: external inhibitory input
    :param W: level of adaptation
    :param Q_e: excitatory quantal conductance
    :param tau_e: excitatory decay
    :param E_e: excitatory reversal potential
    :param Q_i: inhibitory quantal conductance
    :param tau_i: inhibitory decay
    :param E_i: inhibitory reversal potential
    :param E_L: leakage reversal voltage of neurons
    :param g_L: leak conductance
    :param C_m: membrane capacitance
    :param E_L: leak reversal potential
    :param N_tot: cell number
    :param p_connect_e: connectivity probability of excitatory neurons
    :param p_connect_i: connectivity probability of inhibitory neurons
    :param g: fraction of inhibitory cells
    :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
    */
    // firing rate
    // 1e-6 represent spontaneous release of synaptic neurotransmitter or some intrinsic currents of neurons
    float fe = (Fe + 1.0e-6) * (1. - g) * p_connect_e * N_tot + Fe_ext * K_ext_e;
    float fi = (Fi + 1.0e-6) * g * p_connect_i * N_tot + Fi_ext * K_ext_i;

    // conductance fluctuation and effective membrane time constant
    float mu_Ge = Q_e * tau_e * fe;  // Eqns 5 from [MV_2018]
    float mu_Gi = Q_i * tau_i * fi;  // Eqns 5 from [MV_2018]
    float mu_G = g_L + mu_Ge + mu_Gi;  // Eqns 6 from [MV_2018]
    float T_m = C_m / mu_G;  // Eqns 6 from [MV_2018]

    // membrane potential
    *mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G;  // Eqns 7 from [MV_2018]
    // post-synaptic membrane potential event s around muV
    float U_e = Q_e / mu_G * (E_e - *mu_V);
    float U_i = Q_i / mu_G * (E_i - *mu_V);
    // Standard deviation of the fluctuations
    // Eqns 8 from [MV_2018]
    *sigma_V = sqrt(
        fe * square(U_e * tau_e) / (2. * (tau_e + T_m)) + fi * square(U_i * tau_i) / (2. * (tau_i + T_m)));
    // Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
    float T_V_numerator = (fe * square(U_e * tau_e) + fi * square(U_i * tau_i));
    float T_V_denominator = (fe * square(U_e * tau_e) / (tau_e + T_m) + fi * square(U_i * tau_i) / (tau_i + T_m));
    // T_V = numpy.divide(T_V_numerator, T_V_denominator, out=numpy.ones_like(T_V_numerator),
    //                    where=T_V_denominator != 0.0) # avoid numerical error but not use with numba
    *T_V = T_V_numerator / T_V_denominator;
    //return mu_V, sigma_V, T_V
}


float threshold_func(float muV, float sigmaV, float TvN, float P0, float P1, float P2, float P3, float P4, float P5, float P6, float P7, float P8, float P9){
    /*
    The threshold function of the neurons
    :param muV: mean of membrane voltage
    :param sigmaV: variance of membrane voltage
    :param TvN: autocorrelation time constant
    :param P: Fitted coefficients of the transfer functions
    :return: threshold of neurons
    */
    // Normalization factors page 48 after the equation 4 from [ZD_2018]
    float muV0 = -60.0;
    float DmuV0 = 10.0;
    float sV0 = 4.0;
    float DsV0 = 6.0;
    float TvN0 = 0.5;
    float DTvN0 = 1.0;
    float V, S, _T;
    V = (muV - muV0) / DmuV0;
    S = (sigmaV - sV0) / DsV0;
    _T = (TvN - TvN0) / DTvN0;
    // Eqns 11 from [MV_2018]
    return P0 + P1 * V + P2 * S + P3 * _T + P4 * V * V + P5 * S * S + P6 * _T * _T + P7 * V * S + P8 * V * _T + P9 * S * _T;
}

float estimate_firing_rate(float muV, float sigmaV, float Tv, float Vthre){
    /*
    The threshold function of the neurons
    :param muV: mean of membrane voltage
    :param sigmaV: variance of membrane voltage
    :param Tv: autocorrelation time constant
    :param Vthre:threshold of neurons
    */
    // Eqns 10 from [MV_2018]
    return erfc((Vthre - muV) / (sqrt(2.0f) * sigmaV)) / (2.0f * Tv);
}



float TF(float  fe, float fi, float fe_ext, float fi_ext, float W, 
        float P0, float P1, float P2, float P3, float P4, float P5, float P6, float P7, float P8, float P9,
        float E_L){
    /*
    transfer function for inhibitory population
    Inspired from the next repository :
    https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
    :param fe: firing rate of excitatory population
    :param fi: firing rate of inhibitory population
    :param fe_ext: external excitatory input
    :param fi_ext: external inhibitory input
    :param W: level of adaptation
    :param P: Polynome of neurons phenomenological threshold (order 9)
    :param E_L: leak reversal potential
    :return: result of transfer function
    */
    float mu_V, sigma_V, T_V;
    get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, E_L, &mu_V, &sigma_V, &T_V);
    float V_thre = threshold_func(mu_V, sigma_V, T_V * g_L / C_m,
                                 P0, P1, P2, P3, P4, P5, P6, P7, P8, P9);
    V_thre = V_thre * 1e3;  // the threshold need to be in mv and not in Volt
    float f_out = estimate_firing_rate(mu_V, sigma_V, T_V, V_thre);
    return f_out;
}

float TF_excitatory(float fe, float fi, float fe_ext, float fi_ext, float W){
    /*
    transfer function for excitatory population
    :param fe: firing rate of excitatory population
    :param fi: firing rate of inhibitory population
    :param fe_ext: external excitatory input
    :param fi_ext: external inhibitory input
    :param W: level of adaptation
    :return: result of transfer function
    */
    return TF(fe, fi, fe_ext, fi_ext, W, P0_e, P1_e, P2_e, P3_e, P4_e, P5_e, P6_e, P7_e, P8_e, P9_e, E_L_e);
}

float TF_inhibitory(float  fe, float fi, float fe_ext, float fi_ext, float W){
    /*
    transfer function for inhibitory population
    :param fe: firing rate of excitatory population
    :param fi: firing rate of inhibitory population
    :param fe_ext: external excitatory input
    :param fi_ext: external inhibitory input
    :param W: level of adaptation
    :return: result of transfer function
    */
    return TF(fe, fi, fe_ext, fi_ext, W, P0_i, P1_i, P2_i, P3_i, P4_i, P5_i, P6_i, P7_i, P8_i, P9_i, E_L_i);
}

#define df 1e-7
#define dfsq 1e-8 // (df * 1e3) ** 2

float _diff_fe_TF_E( float fe, float fi, float fe_ext, float fi_ext, float W){
    return (TF_excitatory(fe + df, fi, fe_ext, fi_ext, W) - TF_excitatory(fe - df, fi, fe_ext, fi_ext, W)) / (2.0f * df * 1e3);
}

float _diff_fe_TF_I(float fe, float fi, float fe_ext, float fi_ext, float W){
    return (TF_inhibitory(fe + df, fi, fe_ext, fi_ext, W) - TF_inhibitory(fe - df, fi, fe_ext, fi_ext, W)) / (2.0f * df * 1e3);
}

float _diff_fi_TF_E(float fe, float fi, float fe_ext, float fi_ext, float W){
    return (TF_excitatory(fe, fi + df, fe_ext, fi_ext, W) - TF_excitatory(fe, fi - df, fe_ext, fi_ext, W)) / (2.0f * df * 1e3);
}
float _diff_fi_TF_I(float fe, float fi, float fe_ext, float fi_ext, float W){
    return (TF_inhibitory(fe, fi + df, fe_ext, fi_ext, W) - TF_inhibitory(fe, fi - df, fe_ext, fi_ext, W)) / (2.0f * df * 1e3);
}

float _diff2_fe_fe_e(float fe, float fi, float fe_ext, float fi_ext, float W, float TF_e){
    //TF = self.TF_excitatory
    return (TF_excitatory(fe + df, fi, fe_ext, fi_ext, W) - 2.0f * TF_e + TF_excitatory(fe - df, fi, fe_ext, fi_ext, W)) / dfsq;
}
float _diff2_fe_fe_i(float fe, float fi, float fe_ext, float fi_ext, float W, float TF_i){
    //TF = self.TF_inhibitory
    return (TF_inhibitory(fe + df, fi, fe_ext, fi_ext, W) - 2.0f * TF_i + TF_inhibitory(fe - df, fi, fe_ext, fi_ext, W)) / dfsq;
}


float _diff2_fi_fe_TF_E(float fe, float fi, float fe_ext, float fi_ext, float W){
    return (_diff_fi_TF_E(fe + df, fi, fe_ext, fi_ext, W) - _diff_fi_TF_E(fe - df, fi, fe_ext, fi_ext, W)) / (
            2.0f * df * 1e3);
}
float _diff2_fi_fe_TF_I(float fe, float fi, float fe_ext, float fi_ext, float W){
    return (_diff_fi_TF_I(fe + df, fi, fe_ext, fi_ext, W) - _diff_fi_TF_I(fe - df, fi, fe_ext, fi_ext, W)) / (
            2.0f * df * 1e3);
}


float _diff2_fe_fi_TF_E(float fe, float fi, float fe_ext, float fi_ext, float W){
    return (_diff_fe_TF_E(fe, fi + df, fe_ext, fi_ext, W) - _diff_fe_TF_E(fe, fi - df, fe_ext, fi_ext, W)) / (
            2.0f * df * 1e3);
}

float _diff2_fe_fi_TF_I(float fe, float fi, float fe_ext, float fi_ext, float W){
    return (_diff_fe_TF_I(fe, fi + df, fe_ext, fi_ext, W) - _diff_fe_TF_I(fe, fi - df, fe_ext, fi_ext, W)) / (
            2.0f * df * 1e3);
}


float _diff2_fi_fi_e(float fe, float fi, float fe_ext, float fi_ext, float W, float TF_e){
    //TF = self.TF_excitatory
    return (TF_excitatory(fe, fi + df, fe_ext, fi_ext, W) - 2.0f * TF_e + TF_excitatory(fe, fi - df, fe_ext, fi_ext, W)) / dfsq;
}
float _diff2_fi_fi_i(float fe, float fi, float fe_ext, float fi_ext, float W, float TF_i){
    //TF = self.TF_inhibitory
    return (TF_inhibitory(fe, fi + df, fe_ext, fi_ext, W) - 2.0f * TF_i + TF_inhibitory(fe, fi - df, fe_ext, fi_ext, W)) / dfsq;
}

__kernel void adex_dfun(
    // derivatives
    __global float *g_dE       ,
    __global float *g_dI       ,
    __global float *g_dC_ee    ,
    __global float *g_dC_ei    ,
    __global float *g_dC_ii    ,
    __global float *g_dW_e     ,
    __global float *g_dW_i     ,
    __global float *g_dou_drift,
    // state variables
    __global float *g_E       ,
    __global float *g_I       ,
    __global float *g_C_ee    ,
    __global float *g_C_ei    ,
    __global float *g_C_ii    ,
    __global float *g_W_e     ,
    __global float *g_W_i     ,
    __global float *g_ou_drift,
    // coupling variables
    __global float *g_c_0,
    __global float *g_lc_E,
    __global float *g_lc_I,
    // parameters
    __global float *params
)
{
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes

    // state variables
    float E        = g_E       [i*L + l];
    float I        = g_I       [i*L + l];
    float C_ee     = g_C_ee    [i*L + l];
    float C_ei     = g_C_ei    [i*L + l];
    float C_ii     = g_C_ii    [i*L + l];
    float W_e      = g_W_e     [i*L + l];
    float W_i      = g_W_i     [i*L + l];
    float ou_drift = g_ou_drift[i*L + l];


    // coupling variables
    float lc_E = g_lc_E[i*L + l];
    float lc_I = g_lc_I[i*L + l];
    float c_0 = g_c_0[i*L + l];
    

    // derived parameters
    //
    // number of neurons
    float N_e = N_tot * (1 - g);
    float N_i = N_tot * g;

    // compute derivatives

    // external firing rate for the different population
    float E_input_excitatory = c_0 + lc_E + external_input_ex_ex + weight_noise * ou_drift;
    if (E_input_excitatory < 0.0){
        E_input_excitatory = 0.0;
    }

    float E_input_inhibitory = S_i * c_0 + lc_E + external_input_in_ex + weight_noise * ou_drift;
    if (E_input_inhibitory < 0.0){
        E_input_inhibitory = 0.0;
    }

    float I_input_excitatory = lc_I + external_input_ex_in;
    if (I_input_excitatory < 0.0){ // this is not in TVB, but I'm pretty sure it should be...
        I_input_excitatory = 0.0;
    }
    float I_input_inhibitory = lc_I + external_input_in_in;
    if (I_input_inhibitory < 0.0){ // this is not in TVB, but I'm pretty sure it should be...
        I_input_inhibitory = 0.0;
    }
    
    float _diff_fe_TF_e = _diff_fe_TF_E(E, I, E_input_excitatory, I_input_excitatory, W_e);
    float _diff_fe_TF_i = _diff_fe_TF_I(E, I, E_input_inhibitory, I_input_inhibitory, W_i);
    float _diff_fi_TF_e = _diff_fi_TF_E(E, I, E_input_excitatory, I_input_excitatory, W_e);
    float _diff_fi_TF_i = _diff_fi_TF_I(E, I, E_input_inhibitory, I_input_inhibitory, W_i);

    float _TF_e = TF_excitatory(E, I, E_input_excitatory, I_input_excitatory, W_e);
    float _TF_i = TF_inhibitory(E, I, E_input_inhibitory, I_input_inhibitory, W_i);

    float mu_V, sigma_V, T_V; 

   // ported from TVB, adapted originally from:
   // https://github.com/yzerlaut/notebook_papers/blob/master/modeling_mesoscopic_dynamics/mean_field/master_equation.py

   // Excitatory firing rate derivation
    float dE        = (_TF_e - E
                         + 0.5f * C_ee * _diff2_fe_fe_e(E, I, E_input_excitatory, I_input_excitatory, W_e, _TF_e)
                         + 0.5f * C_ei * _diff2_fe_fi_TF_E(E, I, E_input_excitatory, I_input_excitatory, W_e)
                         + 0.5f * C_ei * _diff2_fi_fe_TF_E(E, I, E_input_excitatory, I_input_excitatory, W_e)
                         + 0.5f * C_ii * _diff2_fi_fi_e(E, I, E_input_excitatory, I_input_excitatory, W_e, _TF_e)
                         ) / T;

    // Inhibitory firing rate derivation
    float dI        = (_TF_i - I
                         + 0.5f * C_ee * _diff2_fe_fe_i(E, I, E_input_inhibitory, I_input_inhibitory, W_i, _TF_i)
                         + 0.5f * C_ei * _diff2_fe_fi_TF_I(E, I, E_input_inhibitory, I_input_inhibitory, W_i)
                         + 0.5f * C_ei * _diff2_fi_fe_TF_I(E, I, E_input_inhibitory, I_input_inhibitory, W_i)
                         + 0.5f * C_ii * _diff2_fi_fi_i(E, I, E_input_inhibitory, I_input_inhibitory, W_i, _TF_i)
                         ) / T;
    // Covariance excitatory-excitatory derivation
    float dC_ee     = (_TF_e * (1.0f / T - _TF_e) / N_e
                         + square(_TF_e - E)
                         + 2.0f * C_ee * _diff_fe_TF_e
                         + 2.0f * C_ei * _diff_fi_TF_e
                         - 2.0f * C_ee
                         ) / T;
    // Covariance excitatory-inhibitory or inhibitory-excitatory derivation
    float dC_ei     = ((_TF_e - E) * (_TF_i - I)
                         + C_ee * _diff_fe_TF_e
                         + C_ei * _diff_fe_TF_i
                         + C_ei * _diff_fi_TF_e
                         + C_ii * _diff_fi_TF_i
                         - 2.0f * C_ei
                         ) / T;
    // Covariance inhibitory-inhibitory derivation
    float dC_ii     = (_TF_i * (1.0f / T - _TF_i) / N_i
                         + square(_TF_i - I)
                         + 2.0f * C_ii * _diff_fi_TF_i
                         + 2.0f * C_ei * _diff_fe_TF_i
                         - 2.0f * C_ii
                         ) / T;

    // Adaptation excitatory
    get_fluct_regime_vars( E, I, E_input_excitatory, I_input_excitatory, W_e, E_L_e,
                           &mu_V, &sigma_V, &T_V);
    float dW_e      =  - W_e / tau_w_e + b_e * E + a_e * (mu_V - E_L_e) / tau_w_e;


    // Adaptation inhibitory
    get_fluct_regime_vars( E, I, E_input_inhibitory, I_input_inhibitory, W_i, E_L_i, 
                           &mu_V, &sigma_V, &T_V);
    float dW_i      = -W_i / tau_w_i + b_i * I + a_i * (mu_V - E_L_i) / tau_w_i;

    // Ohrnstein-Uhlenbeck drive
    float dou_drift =  - ou_drift/tau_OU;

    // write out the derivatives
    g_dE       [i*L + l] = dE;
    g_dI       [i*L + l] = dI;
    g_dC_ee    [i*L + l] = dC_ee;
    g_dC_ei    [i*L + l] = dC_ei;
    g_dC_ii    [i*L + l] = dC_ii;
    g_dW_e     [i*L + l] = dW_e;
    g_dW_i     [i*L + l] = dW_i;
    g_dou_drift[i*L + l] = dou_drift;
}
