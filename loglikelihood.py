import jax
import jax.numpy as jnp
from model import jax_stream_model, jax_stream_orbit

BAD_VAL = -1e100

@jax.jit
def loglikelihood_stream(p, dict_data):
    logM, Rs, q, dirx, diry, dirz, logm, rs, x0, z0, vx0, vy0, vz0, time, alpha = p
    dirz = jnp.abs(dirz)
    x0   = jnp.abs(x0)
    z0   = jnp.abs(z0)
    vy0  = jnp.abs(vy0)

    r_data = dict_data['r_data']
    w_data = dict_data['w_data']
    r_err = dict_data['r_err']
    w_err = dict_data['w_err']

    y0 = 0.

    _, _, _, _, r_meds, w_meds, _, _, _  = jax_stream_model(logM, Rs, q, dirx, diry, dirz, logm, rs, x0, y0, z0, vx0, vy0, vz0, time, alpha, tail=0, min_count=101)

    mask = ~jnp.isnan(r_data)

    # Count how many predictions are bad
    nan_mask = jnp.isnan(jnp.where(mask, r_meds, 0.0))
    n_bad = jnp.sum(nan_mask)

    def all_nan_case(_):
        return -jnp.inf #BAD_VAL * r_data.shape[0]

    def some_good_case(_):
        def good_fit_case(_):
            res = ((r_meds - r_data) / r_err) ** 2
            return -0.5 * jnp.nansum(res)

        def bad_fit_case(_):
            return -jnp.inf #BAD_VAL * n_bad

        return jax.lax.cond(n_bad == 0, good_fit_case, bad_fit_case, operand=None)

    logl = jax.lax.cond(jnp.all(jnp.isnan(r_meds)), all_nan_case, some_good_case, operand=None)

    return logl

@jax.jit
def loglikelihood_orbit(p, dict_data):
    logM, Rs, q, dirx, diry, dirz, x0, z0, vx0, vy0, vz0, time, alpha = p
    dirz = jnp.abs(dirz)
    x0   = jnp.abs(x0)
    z0   = jnp.abs(z0)
    vy0  = jnp.abs(vy0)

    r_data = dict_data['r_data']
    w_data = dict_data['w_data']
    r_err = dict_data['r_err']
    w_err = dict_data['w_err']

    y0 = 0.

    _, _, _, _, r_meds, w_meds, _, _, _ = jax_stream_orbit(logM, Rs, q, dirx, diry, dirz, x0, y0, z0, vx0, vy0, vz0, time, alpha)

    mask = ~jnp.isnan(r_data)

    # Count how many predictions are bad
    nan_mask = jnp.isnan(jnp.where(mask, r_meds, 0.0))
    n_bad = jnp.sum(nan_mask)

    def good_fit_case(_):
        res = ((r_meds - r_data) / r_err) ** 2
        return -0.5 * jnp.nansum(res)
    
    def bad_fit_case(_):
        return -jnp.inf #BAD_VAL * r_data.shape[0]

    logl = jax.lax.cond(n_bad == 0, good_fit_case, bad_fit_case, operand=None)

    return logl