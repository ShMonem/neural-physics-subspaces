from functools import partial
import numpy as np
import os
import jax
import jax.numpy as jnp

# Imports from this project
import utils
import minimize

import polyscope as ps
import polyscope.imgui as psim
from jax import debug
#########################################################################
### Proximal operator implicit integrator
#########################################################################

def proximal_func(system, system_def, timestep_h, q_tm1, q_t, q_tp1, subspace_fn=None):

    if subspace_fn is not None:
        # potential energy optimization is done on full dimension
        q_tm1 = subspace_fn(system_def, q_tm1)
        q_t = subspace_fn(system_def, q_t)
        q_tp1 = subspace_fn(system_def, q_tp1)

    q_inertial = 2 * q_t - q_tm1

    l2_term = system.kinetic_energy(system_def, q_inertial, q_tp1 - q_inertial)   # 0.5 * norm((q_tp1 - q_inertial) Mass * (q_tp1 - q_inertial))
    E_term = timestep_h**2 * system.potential_energy(system_def, q_tp1)

    return l2_term + E_term   # eq(6)


def proximal_step(system,
                  system_def,
                  int_state,
                  int_opts,
                  subspace_fn=None,
                  subspace_domain_dict=None):

    q_t = int_state['q_t']
    q_tm1 = int_state['q_tm1']
    timestep_h = int_opts['timestep_h']
    solver_type = int_opts['solver_type']

    initial_guess = 2 * q_t - q_tm1

    def objective(y):
        if subspace_domain_dict is not None:
            y, _, _ = subspace_domain_dict['domain_project_fn'](y, None, None)
        return proximal_func(system, system_def, timestep_h, q_tm1, q_t, y, subspace_fn)

    result, solver_info = minimize.minimize_nonlinear(objective, initial_guess, solver_type)

    # TODO do something with solver_info?

    int_state['q_tm1'] = q_t
    int_state['q_t'] = result

    apply_domain_projection(int_state, subspace_domain_dict)


methods = ['implicit-proximal']


def state_style(method):
    if method in ['implicit-proximal']:
        return '2-state'
    elif method in []:
        return 'state-vel'
    else:
        raise ValueError("unrecognized method")


@partial(jax.jit,
         static_argnames=['system', 'int_opts_tuple', 'subspace_fn', 'subspace_domain_tuple'])
def timestep_internal(system,
                      system_def,
                      int_state,
                      int_opts_tuple,
                      subspace_fn=None,
                      subspace_domain_tuple=None):

    # Un-tuple-ify the dict (done for hashing reasons)
    int_opts = utils.tuple_to_dict(int_opts_tuple)
    subspace_domain_dict = utils.tuple_to_dict(subspace_domain_tuple)

    other_outs = None
    #debug.print("states: {}, {}", int_state['q_t'] , int_state['q_tm1'] )
    if int_opts['method'] == 'implicit-proximal':
        proximal_step(system, system_def, int_state, int_opts, subspace_fn, subspace_domain_dict)

    else:
        raise ValueError("unrecognized integrator method")

    return int_state, other_outs


def timestep(system, system_def, int_state, int_opts, subspace_fn=None, subspace_domain_dict=None,
             collect_velo_snapshots=False, file_name=""):
    if collect_velo_snapshots:
        # linear transformations (4, 3)
        T_t = jnp.concatenate((system_def['fixed_pos'], int_state['q_t'])).reshape(-1, 4, 3)
        T_tm1 = jnp.concatenate((system_def['fixed_pos'], int_state['q_tm1'])).reshape(-1, 4, 3)

        rot_t = T_t[:, :3, :]   # rotational part
        translation = T_t[:, 3, :]   # rotational part
        rot_dot_t = (rot_t - jnp.concatenate((system_def['fixed_pos'], int_state['q_tm1'])).reshape(-1, 4, 3)[:, :3, :]) \
                  / int_opts['timestep_h']

        pos, velo, omega, angular = None, None, None, None
        for bid in range(system.n_bodies):
            # positional velocity
            if bid == 0:
                pos = np.array(jnp.matmul(system.bodiesRen[bid]['W'], T_t[bid]))
                velo = (np.array(jnp.matmul(system.bodiesRen[bid]['W'], T_t[bid]))
                            - np.array(jnp.matmul(system.bodiesRen[bid]['W'], T_tm1[bid]))) / int_opts['timestep_h']

                omega_bid = jnp.matmul(rot_dot_t[bid], rot_dot_t[bid])
                omega = omega_bid
                angular = jnp.reshape(np.array([
                    omega_bid[2, 1] - omega_bid[1, 2],
                    omega_bid[0, 2] - omega_bid[2, 0],
                    omega_bid[1, 0] - omega_bid[0, 1]]) / 2.0, (-1, 1))
            else: # TODO!
                pos = jnp.vstack([pos, np.array(jnp.matmul(system.bodiesRen[bid]['W'], T_t[bid])) ])  # (v_bid, 3)
                velo = jnp.vstack([velo, (np.array(jnp.matmul(system.bodiesRen[bid]['W'], T_t[bid]))
                            - np.array(jnp.matmul(system.bodiesRen[bid]['W'], T_tm1[bid]))) / int_opts['timestep_h']])  # (v_bid, 3)
                # Extract angular velocity from skew-symmetric matrix
                omega_bid = jnp.matmul(rot_dot_t[bid], rot_dot_t[bid])
                omega = jnp.vstack([omega, omega_bid])
                angular = jnp.vstack([angular, jnp.reshape(np.array([
                    omega_bid[2, 1] - omega_bid[1, 2],
                    omega_bid[0, 2] - omega_bid[2, 0],
                    omega_bid[1, 0] - omega_bid[0, 1]]) / 2.0, (-1, 1))])

        np.savez(file_name + ".npz", omega_mat=omega, angular=angular,
                 rot=rot_t, vel=velo, pos=pos, tranz=translation, full_transform=T_t, dt=int_opts['timestep_h'])

    # Pass args along, but tuple-ify the dict, so we can hash it
    new_int_state, other_outs = timestep_internal(
        system,
        system_def,
        int_state,
        utils.dict_to_tuple(int_opts),
        subspace_fn=subspace_fn,
        subspace_domain_tuple=utils.dict_to_tuple(subspace_domain_dict))

    # process other outputs (printing errors, etc)
    if int_opts['method'] == 'implicit-proximal':
        pass

    else:
        raise ValueError("unrecognized integrator method")

    return new_int_state


def initialize_integrator(int_opts, int_state, new_method):

    def soft_add(key, value):
        if key not in int_opts:
            int_opts[key] = value

    int_opts['method'] = new_method

    soft_add('timestep_h', 0.05)

    # TODO do something about velocity / 2-state style swtiching?

    if int_opts['method'] == 'implicit-proximal':
        soft_add('solver_type', 'JAX-LBFGS')

    else:
        raise ValueError("unrecognized integrator method")


def build_ui(int_opts, int_state):

    if psim.TreeNode("integrator"):

        _, int_opts['timestep_h'] = psim.InputFloat("timestep", int_opts['timestep_h'])

        changed, new_method = utils.combo_string_picker("integrator mode", int_opts['method'],
                                                        methods)

        if changed:
            initialize_integrator(int_opts, int_state, new_method)

        if int_opts['method'] == 'implicit-proximal':
            changed, int_opts['solver_type'] = utils.combo_string_picker(
                "solver type", int_opts['solver_type'], minimize.methods)
        else:
            raise ValueError("unrecognized integrator method")

        psim.TreePop()


def update_state(int_opts, int_state, new_q, with_velocity):

    this_style = state_style(int_opts['method'])

    if this_style == '2-state':
        int_state['q_t'] = new_q

        if with_velocity:
            # leave int_state['q_tm1'] untouched
            pass
        else:
            int_state['q_tm1'] = new_q

    elif this_style == 'state-vel':
        if with_velocity:
            int_state['qdot_t'] = (new_q - int_state['q_t']) / int_opts['timestep_h']
        else:
            int_state['qdot_t'] = jnp.zeros_like(int_state['qdot_t'])

        int_state['q_t'] = new_q

    else:
        raise ValueError("unrecognized style")


def apply_domain_projection(int_state, subspace_domain_dict):
    if subspace_domain_dict is None: return

    int_state['q_t'], int_state['q_tm1'], int_state['qdot_t'] = subspace_domain_dict[
        'domain_project_fn'](int_state['q_t'], int_state['q_tm1'], int_state['qdot_t'])
