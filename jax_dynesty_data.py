import os
import pickle
import scipy
import corner
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

import jax
import jax.numpy as jnp
print(jax.devices())

from model import jax_stream_model, backward_integrate_orbit_leapfrog, forward_integrate_orbit_leapfrog
from loglikelihood import wrapper_loglikelihood_data
import dynesty
import dynesty.utils as dyut

BAD_VAL = -1e100

def prior_transform(p):
    #ndim = 15
    logM, Rs, dirx, diry, dirz, \
    logm, rs, \
    x0, z0, vx0, vy0, vz0, \
    t0, a0, sig0 = p

    logM1  = (11 + 3*logM)
    Rs1    = (5 + 20*Rs)
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

    sig1 = 10*sig0

    return [logM1, Rs1, dirx1, diry1, dirz1, 
            logm1, rs1,
            x1, z1, vx1, vy1, vz1,
            t1, a1, sig1]

if __name__ == "__main__":
    # Get data
    nlive  = 5000
    ndim   = 15

    PATH_DATA = '/data/dc824-2/SGA_Tracks'
    name      = 'UGC09239_factor2.5_pixscale0.6' #'NGC0804_factor3.0_pixscale0.6' #'PGC1092512_factor2.5_pixscale0.6' #'UGC09239_factor2.5_pixscale0.6' #'ESO356-012_factor2.5_pixscale0.6' #NAMES[i]
    PATH_SAVE = f'{PATH_DATA}/{name}'
    dict_data = pickle.load(open(f'{PATH_SAVE}/dict_data.pkl', 'rb'))

    nthreads = os.cpu_count()
    mp.set_start_method("spawn", force=True)
    with mp.Pool(nthreads) as poo:
        dns = dynesty.DynamicNestedSampler(wrapper_loglikelihood_data,
                                prior_transform,
                                ndim,
                                logl_args=(dict_data['r_data'], dict_data['r_err'], ),
                                nlive=nlive,
                                sample='rslice',  
                                pool=poo,
                                queue_size=nthreads * 2)
        dns.run_nested(n_effective=10000)

    # Extract results
    res = dns.results

    # Weighted posterior samples
    samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
    samples_equal = dyut.resample_equal(samples, weights)
    logl = res.logl
    
    dict_data['samps'] = samples_equal,
    dict_data['logl']  = logl
    
    with open(f'{PATH_SAVE}/dict_nlive{nlive}_N10000.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    samps = dict_data['samps'][0]
    r     = np.sqrt(samps[:, 2]**2 + samps[:, 3]**2 + samps[:, 4]**2)
    q     = np.exp(-r**2/2) * (np.sqrt(np.pi) * np.exp(r**2/2) * scipy.special.erf(r/np.sqrt(2)) - np.sqrt(2)*r)/np.sqrt(np.pi) + 0.5
    samps = np.column_stack([samps[:, :2], q, samps[:, 2:]])

    # Plot and Save corner plot
    labels = ['logM', 'Rs', 'q', 'dirx', 'diry', 'dirz', 'logm', 'rs', 'x0', 'z0', 'vx0', 'vy0', 'vz0', 'time', 'alpha', 'sig']
    figure = corner.corner(samps, 
                labels=labels,
                color='blue',
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True, 
                title_kwargs={"fontsize": 16})
    figure.savefig(f'{PATH_SAVE}/corner_plot_nlive{nlive}_N10000.pdf')

    # Plot and Save best fit
    best_samps = samps[np.argmax(dict_data['logl'])]

    plt.figure()
    plt.xlabel(r'x [kpc]')
    plt.ylabel(r'y [kpc]')

    best_params = dict_data['samps'][0][np.argmax(dict_data['logl'])]
    logM, Rs, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time, alpha, sig = best_params
    r = np.sqrt(dirx**2 + diry**2 + dirz**2)
    q  = np.exp(-r**2/2) * (np.sqrt(np.pi) * np.exp(r**2/2) * scipy.special.erf(r/np.sqrt(2)) - np.sqrt(2)*r)/np.sqrt(np.pi) + 0.5
    y0   =  0.

    theta_stream, x_stream, y_stream, vz_stream, r_meds, w_meds, x_meds, y_meds, vz_meds = jax_stream_model(logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha, tail=0, min_count=101)

    xv_sat, _ = backward_integrate_orbit_leapfrog(x0, y0, z0, vx0, vy0, vz0, logM, Rs, q, dirx, diry, dirz, time)
    xv_sat_forward, _ = forward_integrate_orbit_leapfrog(xv_sat[0, 0], xv_sat[0, 1], xv_sat[0, 2], xv_sat[0,3], xv_sat[0, 4], xv_sat[0, 5],
                                                logM, Rs, q, dirx, diry, dirz,
                                                time*alpha)
    x_sat = xv_sat_forward[-1, 0]
    y_sat = xv_sat_forward[-1, 1]

    plt.scatter(x_stream, y_stream, c=theta_stream, label='Stream Model', cmap='seismic', s=1, vmin=-2*np.pi, vmax=2*np.pi)
    plt.colorbar()
    plt.scatter(x_meds, y_meds, c='lime', label='Medians', s=50)
    plt.scatter(x_sat, y_sat, c='orange', label='Progenitor', s=100)
    theta_edges = np.linspace(-2*np.pi, 2*np.pi, 36+1)
    theta_bins = 0.5*(theta_edges[:-1] + theta_edges[1:])
    x_data = dict_data['r_data'] * np.cos(theta_bins)
    y_data = dict_data['r_data'] * np.sin(theta_bins)
    plt.scatter(x_data, y_data, c='k', label='Data')    
    plt.legend(loc='best')
    plt.axis('equal')
    plt.axvline(0, c='grey', ls='--')
    plt.axhline(0, c='grey', ls='--')

    # Plot best fit Axis
    norm = np.sqrt(dirx**2 + diry**2)
    nn = 1.2*np.nanmax(dict_data['r_data'])
    plt.plot([-nn*dirx/norm, nn*dirx/norm], [-nn*diry/norm, nn*diry/norm], c='lime', ls='--', linewidth=4)

    # Plot True Axis
    angle = np.deg2rad(dict_data['PA'] + 90)
    dirx = np.cos(angle)
    diry = np.sin(angle)
    norm = np.sqrt(dirx**2 + diry**2)
    nn = 1.2*np.nanmax(dict_data['r_data'])
    plt.plot([-nn*dirx/norm, nn*dirx/norm], [-nn*diry/norm, nn*diry/norm], c='k', ls='--', linewidth=2)

    plt.title(name.split('_')[0] + f" - logl={dict_data['logl'].max():.4f} - q={q:.4f}", fontsize=16)
    plt.xlabel(r'x [kpc]', fontsize=16)
    plt.ylabel(r'y [kpc]', fontsize=16)
    
    plt.savefig(f'{PATH_SAVE}/best_fit_nlive{nlive}_N10000.pdf')