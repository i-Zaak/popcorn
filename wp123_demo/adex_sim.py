"""
A hi-res AdEx w/ two sparse coupling matrices.

"""
import fire
import numpy as np
import scipy.io
import tqdm
import pyopencl as cl
import pyopencl.array as ca
import pyopencl.clrandom as cr
from scipy.sparse import csr_array
from itertools import product

import os
kernel_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            'kernels'
        )
)


class Util:
    # context with device arrays and kernels

    def __init__(self):
        self.init_cl()
        self._progs = []

    def init_cl(self):
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices(device_type=cl.device_type.GPU)[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        self.rng = cr.PhiloxGenerator(self.context)

    def randn(self, arr, mu=0, sigma=1):
        self.rng.fill_normal(arr, mu=mu, sigma=sigma)
        
        # https://github.com/inducer/pyopencl/issues/318
        # optimizations welcome
        while not (np.isfinite(ca.max(arr).get()) and np.isfinite(ca.min(arr).get())):
            print('ಠ_ಠ')
            self.rng.fill_normal(arr, mu=mu, sigma=sigma)

    def randu(self, *shape, a=0, b=1):
        return self.rng.uniform(cq=self.queue, dtype='f', shape=shape, a=a, b=b)

    def zeros(self, *shape):
        return ca.zeros(self.queue, shape, dtype='f')

    def move(self, *shape, values):
        return ca.to_device(self.queue, values) 

    def init_vectors(self, names, shape, method='zeros', **kwargs):
        for name in names.split(' '):
            setattr(self, name, getattr(self, method)(*shape, **kwargs))

    def load_kernel(self, fname, *knames):
        with open(fname, 'r') as f:
            src = f.read()
        prog = cl.Program(self.context, src).build()
        self._progs.append(prog)
        for kname in knames:
            setattr(self, kname, getattr(prog, kname))

    def move_csr(self, name, csr):
        setattr(self, name + '_data', ca.to_device(self.queue, csr.data))
        setattr(self, name + '_indices', ca.to_device(self.queue, csr.indices))
        setattr(self, name + '_indptr', ca.to_device(self.queue, csr.indptr))


def make_step(nvtx, num_sims, SC, LC, dt, ou_noise_scale, debug, stim=None):
    "Setup functions for time stepping."

    # CL kernels expect float32 scalars
    dt = np.float32(dt)

    util = Util()

    # state vars: E I C_ee C_ei C_ii W_e W_i ou_drift

    # allocate state vars
    # util.init_vectors('r V', (nvtx, num_sims))
    util.init_vectors('E I C_ee C_ei C_ii W_e W_i ou_drift', (nvtx, num_sims))
    
    # allocate coupling vars 
    # util.init_vectors('cr cV', (nvtx, num_sims))
    util.init_vectors('cE lcE lcI', (nvtx, num_sims))

    # allocate noise vars 
    #util.init_vectors('zr zV', (nvtx, num_sims))
    util.init_vectors('zou_drift', (nvtx, num_sims))

    # allocate integration vars (Heun)
    #util.init_vectors('ri Vi', (nvtx, num_sims))
    util.init_vectors('Ei Ii C_eei C_eii C_iii W_ei W_ii ou_drifti', (nvtx, num_sims))
    #util.init_vectors('dr1 dV1', (nvtx, num_sims))
    util.init_vectors('dE1 dI1 dC_ee1 dC_ei1 dC_ii1 dW_e1 dW_i1 dou_drift1', (nvtx, num_sims))
    #util.init_vectors('dr2 dV2', (nvtx, num_sims))
    util.init_vectors('dE2 dI2 dC_ee2 dC_ei2 dC_ii2 dW_e2 dW_i2 dou_drift2', (nvtx, num_sims))
    
    # allocate bold output
    util.init_vectors('bold', (nvtx, num_sims))
    
    # alloc balloon state
    util.init_vectors('dsfvq1 dsfvq2 sfvqi sfvq zsfvq', (4, nvtx, num_sims))
    util.sfvq[1:] = np.float32(1.0)

    # b_e is the only varying parameter for now
    util.init_vectors('params', (nvtx, num_sims), 'randu', a=0.0, b=0.01)
    #util.init_vectors('params', (nvtx, num_sims), 'zeros')

    if stim is not None:
        stim_t, stim_w = stim
        assert len(stim_w) == nvtx, f'given {len(stim_w)}, expected {nvtx}'
        util.init_vectors('stim_t', (len(stim_t),), 'move', values=stim_t)
        util.init_vectors('stim_w', (nvtx,), 'move', values=stim_w)

    # move sparse matrix data to GPU
    util.move_csr('sc', SC)
    util.move_csr('lc', LC)
    # load kernels
    util.load_kernel(f'{kernel_path}/spmv.opencl.c', 'spmv')
    util.load_kernel(f'{kernel_path}/adex.opencl.c', 'adex_dfun')
    util.load_kernel(f'{kernel_path}/bound.opencl.c', 'lower_bound')
    util.load_kernel(f'{kernel_path}/stim.opencl.c', 'stim')
    util.load_kernel(f'{kernel_path}/heun.opencl.c', 'heun_pred', 'heun_corr')
    util.load_kernel(f'{kernel_path}/heun_det.opencl.c', 'heun_det_pred', 'heun_det_corr')
    util.load_kernel(f'{kernel_path}/balloon.opencl.c', 'balloon_dfun', 'balloon_readout')

    # handle boilerplate for kernel launches
    def do(f, *args, nvtx=nvtx):
        args = [(arg.data if hasattr(arg, 'data') else arg) for arg in args]
        f(util.queue, (nvtx, num_sims), (1, num_sims), *args)

    def coupling(cE, lcE, lcI, E, I):
        "Compute coupling terms."
        # global coupling transmits excitatory rate 
        do(util.spmv, cE, E, util.sc_data, util.sc_indices, util.sc_indptr)
        # local coupling transmits both excitatory and inhibitory rates
        do(util.spmv, lcE, E , util.lc_data, util.lc_indices, util.lc_indptr)
        do(util.spmv, lcI, I , util.lc_data, util.lc_indices, util.lc_indptr)

    #def dfun(dr, dV, r, V):
    def dfun(dE, dI, dC_ee, dC_ei, dC_ii, dW_e, dW_i, dou_drift, E, I, C_ee, C_ei, C_ii, W_e, W_i, ou_drift):
        "Compute MPR derivatives."
        coupling(util.cE, util.lcE, util.lcI, E, I)
        if debug:
            check_nan([util.cE])        
            check_nan([util.lcE])        
            check_nan([util.lcI])        
        do(util.adex_dfun, 
           dE, dI, dC_ee, dC_ei, dC_ii, dW_e, dW_i, dou_drift,
           E, I, C_ee, C_ei, C_ii, W_e, W_i, ou_drift,
           util.cE, util.lcE, util.lcI,
           util.params)

    def step(i):
        "Do one Heun step."
        # sample noise
        util.randn(util.zou_drift, sigma=ou_noise_scale)

        # predictor step computes dvars from svars
        #dfun(util.dr1, util.dV1, util.r, util.V)
        dfun( util.dE1, util.dI1, util.dC_ee1, util.dC_ei1, util.dC_ii1, util.dW_e1, util.dW_i1, util.dou_drift1,
              util.E, util.I, util.C_ee, util.C_ei, util.C_ii, util.W_e, util.W_i, util.ou_drift)

        if stim is not None:
            i = np.int32(i)
            do(util.stim, i, util.dE1, util.dE1, util.stim_t, util.stim_w)

        if debug:
            check_nan([util.dE1])        
            check_nan([util.dI1])        
            check_nan([util.dC_ee1])     
            check_nan([util.dC_ei1])     
            check_nan([util.dC_ii1])     
            check_nan([util.dW_e1])      
            check_nan([util.dW_i1])      
            check_nan([util.dou_drift1]) 

            

        # and puts Euler result into intermediate arrays ri,Vi
        #do(util.heun_pred, dt, util.ri, util.r, util.dr1, util.zr)
        #do(util.heun_pred, dt, util.Vi, util.V, util.dV1, util.zV)
        do(util.heun_det_pred, dt, util.Ei,        util.E,        util.dE1)        
        do(util.heun_det_pred, dt, util.Ii,        util.I,        util.dI1)        
        do(util.heun_det_pred, dt, util.C_eei,     util.C_ee,     util.dC_ee1)     
        do(util.heun_det_pred, dt, util.C_eii,     util.C_ei,     util.dC_ei1)     
        do(util.heun_det_pred, dt, util.C_iii,     util.C_ii,     util.dC_ii1)     
        do(util.heun_det_pred, dt, util.W_ei,      util.W_e,      util.dW_e1)      
        do(util.heun_det_pred, dt, util.W_ii,      util.W_i,      util.dW_i1)      
        do(util.heun_pred,     dt, util.ou_drifti, util.ou_drift, util.dou_drift1, util.zou_drift)

        # no negative E or I values
        lb = np.float32(0.0)
        do(util.lower_bound, util.Ei, lb)
        do(util.lower_bound, util.Ii, lb)
        if debug:
            check_nan([util.Ei])        
            check_nan([util.Ii])        
            check_nan([util.C_eei])     
            check_nan([util.C_eii])     
            check_nan([util.C_iii])     
            check_nan([util.W_ei])      
            check_nan([util.W_ii])      
            check_nan([util.ou_drifti]) 

        # corrector step computes dr2,dV2 from intermediate states ri,Vi
        #dfun(util.dr2, util.dV2, util.ri, util.Vi)
        dfun( util.dE2, util.dI2, util.dC_ee2, util.dC_ei2, util.dC_ii2, util.dW_e2, util.dW_i2, util.dou_drift2,
              util.Ei, util.Ii, util.C_eei, util.C_eii, util.C_iii, util.W_ei, util.W_ii, util.ou_drifti)
        if stim is not None:
            i = np.int32(i)
            do(util.stim, i, util.dE2, util.dE2, util.stim_t, util.stim_w)

        if debug:
            check_nan([util.dE2])        
            check_nan([util.dI2])        
            check_nan([util.dC_ee2])     
            check_nan([util.dC_ei2])     
            check_nan([util.dC_ii2])     
            check_nan([util.dW_e2])      
            check_nan([util.dW_i2])      
            check_nan([util.dou_drift2]) 

        # and writes Heun result into arrays r,V
        #do(util.heun_corr, dt, util.r, util.r, util.dr1, util.dr2, util.zr)
        #do(util.heun_corr, dt, util.V, util.V, util.dV1, util.dV2, util.zV)
        do(util.heun_det_corr, dt, util.E,        util.E,        util.dE1,        util.dE2)        
        do(util.heun_det_corr, dt, util.I,        util.I,        util.dI1,        util.dI2)        
        do(util.heun_det_corr, dt, util.C_ee,     util.C_ee,     util.dC_ee1,     util.dC_ee2)     
        do(util.heun_det_corr, dt, util.C_ei,     util.C_ei,     util.dC_ei1,     util.dC_ei2)     
        do(util.heun_det_corr, dt, util.C_ii,     util.C_ii,     util.dC_ii1,     util.dC_ii2)     
        do(util.heun_det_corr, dt, util.W_e,      util.W_e,      util.dW_e1,      util.dW_e2)      
        do(util.heun_det_corr, dt, util.W_i,      util.W_i,      util.dW_i1,      util.dW_i2)      
        do(util.heun_corr,     dt, util.ou_drift, util.ou_drift, util.dou_drift1, util.dou_drift2, util.zou_drift)

        # no negative E or I values
        do(util.lower_bound, util.E, lb)
        do(util.lower_bound, util.I, lb)

        if debug:
            check_nan([util.E])        
            check_nan([util.I])        
            check_nan([util.C_ee])     
            check_nan([util.C_ei])     
            check_nan([util.C_ii])     
            check_nan([util.W_e])      
            check_nan([util.W_i])      
            check_nan([util.ou_drift]) 

    def bold_step(dt):
        "Do one step of the balloon model."
        dt = np.float32(dt)
        # do Heun step on balloon model, using E as neural input
        do(util.balloon_dfun, util.dsfvq1, util.sfvq, util.E)
        do(util.heun_pred, dt, util.sfvqi, util.sfvq, util.dsfvq1, util.zsfvq, nvtx=4*nvtx)
        do(util.balloon_dfun, util.dsfvq2, util.sfvqi, util.E)
        do(util.heun_corr, dt, util.sfvq, util.sfvq, util.dsfvq1, util.dsfvq2, util.zsfvq, nvtx=4*nvtx)
        # update bold signal
        do(util.balloon_readout, util.sfvq, util.bold)

    return util, step, bold_step


def load_csr_npz(file_path, keep_diag):
    dat = np.load(file_path)
    conn = csr_array(
            (dat['data'], dat['indices'], dat['indptr']), shape=dat['shape'], dtype=np.float32
    )
    if not keep_diag:
        conn = conn.tolil()
        conn.setdiag(0.)
        conn = conn.tocsr()
    return conn

def check_nan(arrs):
    for arr in arrs:
        a = arr.get()
        assert not np.any(np.isnan(a))
        assert np.all(np.isfinite(a))

def simulate(
        matrices='matrices.mat', 
        prefix='',
        output_path='',
        sc=None, lc=None, 
        sc_a=1., lc_a=1.,
        reduced=False, red_nvtx=512, 
        keep_diag=False, 
        normalize=True, 
        debug=False,
        sim_len=1000.,
        noise_scale=0.035,
        tavg=True,
        stim=None
        ):

    if sc is not None and lc is not None:
        SC = load_csr_npz(sc, keep_diag)
        LC = load_csr_npz(lc, keep_diag)
    else:
        # load the global and local connectivity matrices
        mat = scipy.io.loadmat(matrices)
        SC = mat['SC']
        LC = mat['LC']

    if reduced:
        # for testing, can run just a subset of the network, like first 512
        # vertices, but could also be a mask selecting just 5 regions, etc.
        nvtx = red_nvtx
        SC = SC[:nvtx, :nvtx]
        LC = LC[:nvtx, :nvtx]
        print('reduced network shape to', SC.shape, LC.shape)

    if normalize:
        LC /= (LC.data.max())
        SC /= SC.data.max()
        print('normalized to max weight = 1.0')

    if stim is not None:
        dat = np.load(stim)
        stim = dat['stim_t'].astype(np.float32), dat['stim_w'].astype(np.float32)

    SC *= sc_a;
    LC *= lc_a;

    if prefix != '':
        prefix += '_'
    if output_path != '':
        assert os.path.exists(output_path), f"output folder doesn't exist: {output_path}"
        prefix = f'{output_path}/{prefix}'

    dt = 0.1
    ou_noise_scale = noise_scale
    nvtx = SC.shape[0]
    num_sims = 1 # there should be num_params too...

    # prepare the GPU arrays and stepping function
    util, step, bold_step = make_step(nvtx, num_sims, SC, LC, dt, ou_noise_scale, debug, stim)

    # set initial conditions
    #util.rng.fill_uniform(util.r, a=0., b=2.0)
    #util.rng.fill_uniform(util.V, a=-2.0, b=1.5)

    #util.rng.fill_uniform(util.E,        a=1e-3,    b=250.e-3)
    #util.rng.fill_uniform(util.I,        a=1e-3,    b=250.e-3) 
    #util.rng.fill_uniform(util.C_ee,     a=0.0e-3,  b=0.5e-3) 
    #util.rng.fill_uniform(util.C_ei,     a=-0.5e-3, b=0.5e-3) 
    #util.rng.fill_uniform(util.C_ii,     a=0.0e-3,  b=0.5e-3) 
    #util.rng.fill_uniform(util.W_e,      a=0.0,     b=200.0) 
    #util.rng.fill_uniform(util.W_i,      a=0.0,     b=0.0) 
    #util.rng.fill_uniform(util.ou_drift, a=0.0,     b=0.0)
    util.W_e.fill(100.)

    # simulation times
    import time
    tic = time.time()

    # neural field iterations & output storage
    niter = int(sim_len /dt)  # this many iterations
    nskip = int(1/dt)       # but only save every nskip iterations to file
    if not tavg:
        nskip = niter + 1 # don't save tavg
    tavg_shape = niter//nskip+1, nvtx, num_sims
    def open_mmap_file(fname, tavg_shape):
        print(f'output {fname} ~{(np.prod(tavg_shape)*4) >> 30} GB')
        return np.lib.format.open_memmap(
                fname, mode='w+', dtype='f',
                shape=tavg_shape)
    Es        = open_mmap_file(f'{prefix}Es.npy', tavg_shape)
    Is        = open_mmap_file(f'{prefix}Is.npy', tavg_shape)
    C_ees     = open_mmap_file(f'{prefix}C_ees.npy', tavg_shape)
    C_eis     = open_mmap_file(f'{prefix}C_eis.npy', tavg_shape)
    C_iis     = open_mmap_file(f'{prefix}C_iis.npy', tavg_shape)
    W_es      = open_mmap_file(f'{prefix}W_es.npy', tavg_shape)
    W_is      = open_mmap_file(f'{prefix}W_is.npy', tavg_shape)
    ou_drifts = open_mmap_file(f'{prefix}ou_drifts.npy', tavg_shape)

    #Es_file = f'{prefix}Es.npy'
    #print(f'output {Es_file} ~{(np.prod(tavg_shape)*4) >> 30} GB')
    #Es = np.lib.format.open_memmap(
    #        Es_file, mode='w+', dtype='f',
    #        shape=tavg_shape)
    #Is_file = f'{prefix}Is.npy'
    #print(f'output {Is_file} ~{(np.prod(tavg_shape)*4) >> 30} GB')
    #Is = np.lib.format.open_memmap(
    #        Is_file, mode='w+', dtype='f',
    #        shape=tavg_shape)

    # bold iterations & output storage 
    bold_dtskip = int(1/dt) # 0.01 s rescaled MPR
    bold_nskip = int(70/dt)         # sample bold every 700 ms (HCP)
    bolds_shape = niter//bold_nskip+1, nvtx, num_sims
    bolds_file = f'{prefix}bolds.npy'
    print(f'output {bolds_file} ~{(np.prod(bolds_shape)*4) >> 30} GB')
    bolds = np.lib.format.open_memmap(
            bolds_file, mode='w+', dtype='f',
            shape=bolds_shape)

    # do time stepping
    for i in tqdm.trange(niter):
        step(i)
        # bold is slow, don't step it every time
        if i % bold_dtskip == 0:
            bold_step(dt*1e-2*bold_dtskip) # to seconds, but rescaled by 10 for the MPR
        # save bold & states every few steps
        if i % bold_nskip == 0:
            util.bold.get_async(util.queue, bolds[i//bold_nskip])
        if i % nskip == 0:
            util.E.get_async(util.queue,        Es       [i//nskip])
            util.I.get_async(util.queue,        Is       [i//nskip])
            util.C_ee.get_async(util.queue,     C_ees    [i//nskip])
            util.C_ei.get_async(util.queue,     C_eis    [i//nskip])
            util.C_ii.get_async(util.queue,     C_iis    [i//nskip])
            util.W_e.get_async(util.queue,      W_es     [i//nskip])
            util.W_i.get_async(util.queue,      W_is     [i//nskip])
            util.ou_drift.get_async(util.queue, ou_drifts[i//nskip])

    # opencl operations are asynchronous, wait for everything to finish & report time
    print('finishing...')
    util.queue.finish()
    toc = time.time() - tic
    print(f'done in {toc:0.2f}s, {toc/(niter)*1e3:0.3f} ms/iter of {num_sims}')


def plot():
    from tvb.simulator.models import ZerlautAdaptationSecondOrder
    import matplotlib.pylab as plt
    cl_results = {svar: np.load(f'{svar}s.npy') for svar in ZerlautAdaptationSecondOrder.state_variables}
    fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(8,8))
    for ax, (svar, data) in zip(axs.flatten(), cl_results.items()):
        ax.plot(data[:,:500,:].squeeze())
        ax.set(title=svar)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    fire.Fire({
        'simulate': simulate,
        'plot': plot,
    })
