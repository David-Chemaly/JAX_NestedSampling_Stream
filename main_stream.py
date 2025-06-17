import time
import pickle
from tqdm import tqdm

import numpy as np

import jax
import jax.numpy as jnp
import blackjax
from blackjax.ns.utils import finalise
print(jax.devices())

from data import get_data_stream
from prior import prior_dists_stream, logprior_stream, sample_from_priors_stream
from loglikelihood import loglikelihood_stream

if __name__ == "__main__":
    # Get data
    q_true = 0.8
    seed   = 42
    sigma  = 1
    n_live = 500
    PATH_SAVE = f'.'

    dict_data = get_data_stream(q_true, seed, sigma)

    # | Define the Nested Sampling algorithm
    n_dims   = len(prior_dists_stream)
    n_delete = int(n_live*0.5) # 50% if GPU
    num_mcmc_steps = n_dims * 3

    # Initialise the nested sampling algorithm using Blackjax
    print("Setting up nested sampling algorithm...")
    algo = blackjax.nss(
        logprior_fn=logprior_stream,
        loglikelihood_fn=lambda p: loglikelihood_stream(p, dict_data),
        num_delete=n_delete,
        num_inner_steps=num_mcmc_steps
    )

    print("Algorithm set up with parameters:")
    print(f"n_dims: {n_dims}, n_delete: {n_delete}, num_mcmc_steps: {num_mcmc_steps}")

    print("Looking for good initial particles...")
    ### Make sure all initial particles are good ###
    index_good = 0
    good_initial_particles = jnp.array([])
    while index_good < n_live:
        start = time.time()
        # Initialise random key and generate initial particles
        rng_key = jax.random.PRNGKey(np.random.randint(10000000000))
        rng_key, init_key = jax.random.split(rng_key)

        initial_particles = sample_from_priors_stream(init_key, n_live)
        logl = jax.vmap(lambda p: loglikelihood_stream(p, dict_data))(initial_particles)

        arg_good = jnp.where(jnp.isfinite(logl))[0]

        if len(good_initial_particles) == 0:
            good_initial_particles = initial_particles[arg_good]
        else:
            good_initial_particles = jnp.concatenate([good_initial_particles, initial_particles[arg_good]], axis=0)

        index_good += len(arg_good)
        end = time.time()
        print(f"Found {index_good} good initial particles out of {n_live} sampled in {end - start:.4f} seconds.")

    good_initial_particles = good_initial_particles[:n_live]
    print("Initial particles generated, shape:", good_initial_particles.shape)

    # Initialise the sampler state
    state = algo.init(good_initial_particles)

    # Define a one-step function for the nested sampling (JIT compiled for efficiency)
    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = algo.step(subk, state)
        return (state, k), dead_point

    # Evaluate initial loglikelihoods (optional pre-check)
    start = time.time()
    init_key, _ = jax.random.split(init_key, 2)
    results = jax.vmap(lambda p: loglikelihood_stream(p, dict_data))(sample_from_priors_stream(init_key, n_live))
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")
    print(jax.devices())
    
    dead = []

    rng_key = jax.random.PRNGKey(0)
    print("Running nested sampling...")
    with tqdm(desc="Dead points", unit=" dead points") as pbar:
        while (not state.logZ_live - state.logZ < -3):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            print(f'logZ: {state.logZ_live - state.logZ:.4f}')
            pbar.update(n_delete)

    final_state = finalise(state, dead)

    with open(f'{PATH_SAVE}/stream_final_state_nlive{n_live}_ndelete{n_delete}_seed{seed}.pkl', 'wb') as f:
        pickle.dump(final_state, f)