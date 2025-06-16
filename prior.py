import distrax
import jax
import jax.numpy as jnp

# Define prior bounds for parameters
PRIOR_PARAMS = {
    'logM': {'min': 11.0,'max': 14.0},
    'Rs'  : {'min': 5.0, 'max': 25.0},
    'q'   : {'min': 0.5, 'max': 1.5},
    'dirx': {'mu' : 0.0, 'sigma': 1.0},
    'diry': {'mu' : 0.0, 'sigma': 1.0},
    'dirz': {'mu' : 0.0, 'sigma': 1.0},
    'logm': {'min': 7.0, 'max': 9.0},
    'rs'  : {'min': 1.0, 'max': 3.0},
    'x0'  : {'mu' : 0.0, 'sigma': 150.0},
    'z0'  : {'mu' : 0.0, 'sigma': 150.0},
    'vx0' : {'mu' : 0.0, 'sigma': 250.0},
    'vy0' : {'mu' : 0.0, 'sigma': 250.0},
    'vz0' : {'mu' : 0.0, 'sigma': 250.0},
    't0'  : {'min': 1.0, 'max': 4.0},
    'a0'  : {'min': 0.9, 'max': 1.1}
}

### Define prior distributions for stream parameters ###
prior_dists_stream = {
    'logM': distrax.Uniform(low=PRIOR_PARAMS['logM']['min'], high=PRIOR_PARAMS['logM']['max']),
    'Rs'  : distrax.Uniform(low=PRIOR_PARAMS['Rs']['min'], high=PRIOR_PARAMS['Rs']['max']),
    'q'   : distrax.Uniform(low=PRIOR_PARAMS['q']['min'], high=PRIOR_PARAMS['q']['max']),
    'dirx': distrax.Normal(loc=PRIOR_PARAMS['dirx']['mu'], scale=PRIOR_PARAMS['dirx']['sigma']),
    'diry': distrax.Normal(loc=PRIOR_PARAMS['diry']['mu'], scale=PRIOR_PARAMS['diry']['sigma']),
    'dirz': distrax.Normal(loc=PRIOR_PARAMS['dirz']['mu'], scale=PRIOR_PARAMS['dirz']['sigma']),
    'logm': distrax.Uniform(low=PRIOR_PARAMS['logm']['min'], high=PRIOR_PARAMS['logm']['max']),
    'rs'  : distrax.Uniform(low=PRIOR_PARAMS['rs']['min'], high=PRIOR_PARAMS['rs']['max']),
    'x0'  : distrax.Normal(loc=PRIOR_PARAMS['x0']['mu'], scale=PRIOR_PARAMS['x0']['sigma']),
    'z0'  : distrax.Normal(loc=PRIOR_PARAMS['z0']['mu'], scale=PRIOR_PARAMS['z0']['sigma']),
    'vx0' : distrax.Normal(loc=PRIOR_PARAMS['vx0']['mu'], scale=PRIOR_PARAMS['vx0']['sigma']),
    'vy0' : distrax.Normal(loc=PRIOR_PARAMS['vy0']['mu'], scale=PRIOR_PARAMS['vy0']['sigma']),
    'vz0' : distrax.Normal(loc=PRIOR_PARAMS['vz0']['mu'], scale=PRIOR_PARAMS['vz0']['sigma']),
    't0'  : distrax.Uniform(low=PRIOR_PARAMS['t0']['min'], high=PRIOR_PARAMS['t0']['max']),
    'a0'  : distrax.Uniform(low=PRIOR_PARAMS['a0']['min'], high=PRIOR_PARAMS['a0']['max']),
}


@jax.jit
def logprior_stream(params):
    logp = (prior_dists_stream['logM'].log_prob(params[0]) +
            prior_dists_stream['Rs'].log_prob(params[1]) +
            prior_dists_stream['q'].log_prob(params[2]) +
            prior_dists_stream['dirx'].log_prob(params[3]) +
            prior_dists_stream['diry'].log_prob(params[4]) +
            prior_dists_stream['dirz'].log_prob(params[5]) +
            prior_dists_stream['logm'].log_prob(params[6]) +
            prior_dists_stream['rs'].log_prob(params[7]) +
            prior_dists_stream['x0'].log_prob(params[8]) +
            prior_dists_stream['z0'].log_prob(params[9]) +
            prior_dists_stream['vx0'].log_prob(params[10]) +
            prior_dists_stream['vy0'].log_prob(params[11]) +
            prior_dists_stream['vz0'].log_prob(params[12]) +
            prior_dists_stream['t0'].log_prob(params[13]) +
            prior_dists_stream['a0'].log_prob(params[14]))
    return logp

def sample_from_priors_stream(rng_key, n_samples):
    keys = jax.random.split(rng_key, len(prior_dists_stream))
    return jnp.array([prior_dists_stream[key].sample(seed=keys[i], sample_shape=(n_samples,)) for i, key in enumerate(prior_dists_stream.keys())]).T

### Define prior distributions for orbit parameters ###
prior_dists_orbit = {
    'logM': distrax.Uniform(low=PRIOR_PARAMS['logM']['min'], high=PRIOR_PARAMS['logM']['max']),
    'Rs'  : distrax.Uniform(low=PRIOR_PARAMS['Rs']['min'], high=PRIOR_PARAMS['Rs']['max']),
    'q'   : distrax.Uniform(low=PRIOR_PARAMS['q']['min'], high=PRIOR_PARAMS['q']['max']),
    'dirx': distrax.Normal(loc=PRIOR_PARAMS['dirx']['mu'], scale=PRIOR_PARAMS['dirx']['sigma']),
    'diry': distrax.Normal(loc=PRIOR_PARAMS['diry']['mu'], scale=PRIOR_PARAMS['diry']['sigma']),
    'dirz': distrax.Normal(loc=PRIOR_PARAMS['dirz']['mu'], scale=PRIOR_PARAMS['dirz']['sigma']),
    'x0'  : distrax.Normal(loc=PRIOR_PARAMS['x0']['mu'], scale=PRIOR_PARAMS['x0']['sigma']),
    'z0'  : distrax.Normal(loc=PRIOR_PARAMS['z0']['mu'], scale=PRIOR_PARAMS['z0']['sigma']),
    'vx0' : distrax.Normal(loc=PRIOR_PARAMS['vx0']['mu'], scale=PRIOR_PARAMS['vx0']['sigma']),
    'vy0' : distrax.Normal(loc=PRIOR_PARAMS['vy0']['mu'], scale=PRIOR_PARAMS['vy0']['sigma']),
    'vz0' : distrax.Normal(loc=PRIOR_PARAMS['vz0']['mu'], scale=PRIOR_PARAMS['vz0']['sigma']),
    't0'  : distrax.Uniform(low=PRIOR_PARAMS['t0']['min'], high=PRIOR_PARAMS['t0']['max']),
    'a0'  : distrax.Uniform(low=PRIOR_PARAMS['a0']['min'], high=PRIOR_PARAMS['a0']['max']),
}


@jax.jit
def logprior_orbit(params):
    logp = (prior_dists_orbit['logM'].log_prob(params[0]) +
            prior_dists_orbit['Rs'].log_prob(params[1]) +
            prior_dists_orbit['q'].log_prob(params[2]) +
            prior_dists_orbit['dirx'].log_prob(params[3]) +
            prior_dists_orbit['diry'].log_prob(params[4]) +
            prior_dists_orbit['dirz'].log_prob(params[5]) +
            prior_dists_orbit['x0'].log_prob(params[8]) +
            prior_dists_orbit['z0'].log_prob(params[9]) +
            prior_dists_orbit['vx0'].log_prob(params[10]) +
            prior_dists_orbit['vy0'].log_prob(params[11]) +
            prior_dists_orbit['vz0'].log_prob(params[12]) +
            prior_dists_orbit['t0'].log_prob(params[13]) +
            prior_dists_orbit['a0'].log_prob(params[14]))
    return logp

def sample_from_priors_orbit(rng_key, n_samples):
    keys = jax.random.split(rng_key, len(prior_dists_orbit))
    return jnp.array([prior_dists_orbit[key].sample(seed=keys[i], sample_shape=(n_samples,)) for i, key in enumerate(prior_dists_orbit.keys())]).T
