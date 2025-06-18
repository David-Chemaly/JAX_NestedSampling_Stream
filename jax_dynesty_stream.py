import time
import pickle
import scipy

import numpy as np

import jax
print(jax.devices())

from data import get_data_stream
from loglikelihood import loglikelihood_stream

import dynesty
import dynesty.utils as dyut

def prior_transform(p):
    #ndim = 13
    logM, Rs, q, dirx, diry, dirz, \
    logm, rs, \
    x0, z0, vx0, vy0, vz0, \
    t0, a0 = p

    logM1  = (11 + 3*logM)
    Rs1    = (5 + 20*Rs)
    q1     = 0.5 + q
    dirx1, diry1, dirz1 = [
        scipy.special.ndtri(_) for _ in [dirx, diry, 0.5 + dirz/2]
    ]

    logm1 = (7 + 2*logm) 
    rs1   = (1 + 2*rs)

    x1, z1 = [
        scipy.special.ndtri(_) * 150 for _ in [0.5 + x0/2, 0.5 + z0/2]
    ]

    vx1, vy1, vz1 = [
        scipy.special.ndtri(_) * 250 for _ in [vx0, 0.5 + vy0/2, vz0]
    ]

    t1 = 1 + 3*t0

    a1 = 0.9 + 0.2*a0

    return [logM1, Rs1, q1, dirx1, diry1, dirz1, 
            logm1, rs1,
            x1, z1, vx1, vy1, vz1,
            t1, a1]

if __name__ == "__main__":
    # Get data
    q_true = 0.8
    seed   = 42
    sigma  = 1
    nlive  = 1000
    ndim   = 15
    PATH_SAVE = f'.'

    dict_data = get_data_stream(q_true, seed, sigma)

    dns = dynesty.DynamicNestedSampler(loglikelihood_stream,
                            prior_transform,
                            ndim,
                            logl_args=(dict_data, ),
                            nlive=nlive,
                            sample='rwalk')  # rslice
    dns.run_nested(n_effective=10000)

    # Extract results
    res = dns.results

    # Weighted posterior samples
    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
    samples_equal = dyut.resample_equal(samples, weights)
    logl = res.logl

    dict_data = {
                    'samps': samples_equal,
                    'logl': logl,
                }
    
    with open(f'{PATH_SAVE}/dns_results_stream.pkl', 'wb') as f:
        pickle.dump(dict_data, f)
    print("Saved results to disk.")
    print("Done!")