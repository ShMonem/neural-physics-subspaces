import sys, os
from functools import partial
import argparse, json

import numpy as np
import scipy
import scipy.optimize
import jax
import jax.numpy as jnp
import jax.scipy

import equinox as eqx

import polyscope as ps
import polyscope.imgui as psim
from jax import debug
# import igl

# Imports from this project
import autoencoder_config as config
import autoencoder_layers as layers
import integrators
from utils import ensure_dir_exists

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

FRAME = 1
RECORD_FRAME = False
RECORD_SNAPSHOTS = False
NUM_SNAPSHOTS = 700
colour=(0.0, 0.7, 0.2) # blue
def main():
    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_system_args(parser)
    config.add_jax_args(parser)

    # Arguments specific to this program
    parser.add_argument("--integrator", type=str, default="implicit-proximal")
    parser.add_argument("--output_dir", type=str, default="../output")
    parser.add_argument("--output_nn_dir", type=str, default="pretrained_models")
    parser.add_argument("--subspace_name", type=str, default="_ReLU_epochs_50_rot_latent_dim_9_tranz_latent_dim_3")
    parser.add_argument("--subspace_model", type=str, default="checkpoint_50")
    parser.add_argument("--subspace_info", type=str, default="info_50")
    parser.add_argument("--framesFolder", type=str, default="frames")
    parser.add_argument("--snapsFolder", type=str, default="snapshots")

    # if to use neural network
    parser.add_argument("--use_nn_subsapce", type=str, default=True)

    # build correct paths arguments
    args = parser.parse_args()
    args.subspace_model = os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, args.subspace_name, args.subspace_model)
    args.subspace_info = os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, args.subspace_name, args.subspace_info)

    # Process args
    config.process_jax_args(args)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)

    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode('none')

    #########################################################################
    ### Load subspace map (if given)
    #########################################################################

    # If we're running on a use_subspace system, load it
    model_params = None
    subspace_dim = -1
    subspace_domain_dict = None
    if args.use_nn_subsapce:
        print(f"Loading subspace from {args.subspace_model}")

        # load autoencoder dictionary .json
        with open(args.subspace_model + '.json', 'r') as json_file:
            autoencoder_model_dict = json.loads(json_file.read())

        autoencoder = layers.Autoencoder(autoencoder_model_dict)
        _, params_static = eqx.partition(autoencoder, eqx.is_array)
        model_params = eqx.tree_deserialise_leaves(args.subspace_model + "_autoencoder.eqx", autoencoder)

        # load other info
        d = np.load(args.subspace_info + ".npy", allow_pickle=True).item()

        subspace_dim = d['subspace_dim']
        subspace_domain_dict = layers.get_latent_domain_dict(d['subspace_domain_type'])
        latent_comb_dim = system_def['interesting_states'].shape[0]
        t_schedule_final = d['t_schedule_final']

        def apply_subspace(auto_model_params, x, cond_params):
            autoencoder_model = eqx.combine(auto_model_params, params_static)
            return autoencoder_model.decoder(jnp.concatenate((x, cond_params), axis=-1))

        def get_full_from_subspace(auto_model_params, z):
            autoencoder_model = eqx.combine(auto_model_params, params_static)
            return autoencoder_model.encoder(z)

        if args.system_name != d['system']:
            raise ValueError("system name does not match loaded weights")
        if args.problem_name != d['problem_name']:
            raise ValueError("problem name does not match loaded weights")
    use_subspace = model_params is not None

    print("System dimension: " + str(system_def['init_pos'].shape[0]))
    if use_subspace:
        print("Subspace dimension: " + str(subspace_dim))

    #########################################################################
    ### Set up state & UI params
    #########################################################################

    ## Integrator setup
    int_opts = {}
    int_state = {}
    integrators.initialize_integrator(int_opts, int_state, args.integrator)

    ## State of the system

    # UI state
    run_sim = True
    eval_energy_every = True
    update_viz_every = True

    # Set up state parameters

    if use_subspace:
        base_latent = jnp.zeros(subspace_dim) + subspace_domain_dict['initial_val']
    else:
        base_latent = None

    def reset_state():
        if use_subspace:
            # int_state['q_t'] = base_latent
            int_state['q_t'] = get_full_from_subspace(model_params, system_def['init_pos'])
        else:
            int_state['q_t'] = system_def['init_pos']
        int_state['q_tm1'] = int_state['q_t']
        int_state['qdot_t'] = jnp.zeros_like(int_state['q_t'])

        system.visualize(system_def, state_to_system(system_def, int_state['q_t']), colour=colour)

    def state_to_system(system_def, state):
        if use_subspace:
            return apply_subspace(model_params, state, system_def['cond_param'])
        else:
            # in the non-latent state, it's the identity
            return state

    if use_subspace:
        baseState = state_to_system(system_def, base_latent)
    else:
        baseState = system_def['init_pos']

    if use_subspace:
        subspace_fn = state_to_system
    else:
        subspace_fn = None

    ps.set_automatically_compute_scene_extents(False)
    reset_state()  # also creates initial viz

    print(f"state_to_system dtype: {state_to_system(system_def, int_state['q_t']).dtype}")

    @jax.jit
    def eval_potential_energy(system_def, q):
        return system.potential_energy(system_def, state_to_system(system_def, q))

    #########################################################################
    ### Main loop, sim step, and UI
    #########################################################################

    def main_loop():

        nonlocal int_opts, int_state, run_sim, base_latent, update_viz_every, eval_energy_every
        global FRAME

        # Define the GUI

        # some latent sliders
        if use_subspace:

            psim.TextUnformatted(f"subspace domain type: {subspace_domain_dict['domain_name']}")

            if psim.TreeNode("explore current latent"):

                psim.TextUnformatted("This is the current state of the system.")

                any_changed = False
                tmp_state_q = int_state['q_t'].copy()
                low = subspace_domain_dict['viz_entry_bound_low']
                high = subspace_domain_dict['viz_entry_bound_high']
                for i in range(subspace_dim):
                    s = f"latent_{i}"
                    val = tmp_state_q[i]
                    changed, val = psim.SliderFloat(s, val, low, high)
                    if changed:
                        any_changed = True
                        tmp_state_q = tmp_state_q.at[i].set(val)

                if any_changed:
                    integrators.update_state(int_opts, int_state, tmp_state_q, with_velocity=True)
                    integrators.apply_domain_projection(int_state, subspace_domain_dict)
                    system.visualize(system_def, state_to_system(system_def, int_state['q_t']), colour=colour)

                psim.TreePop()

        # Helpers to build other parts of the UI
        integrators.build_ui(int_opts, int_state)
        system.build_system_ui(system_def)

        ps.look_at((-0.00356241, 0.06179327, 0.34390159), (-0.00356241, 0.06179327, 0.03906773))
        # update visualization every frame
        if update_viz_every or run_sim:
            system.visualize(system_def, state_to_system(system_def, int_state['q_t']), colour=colour)

            if run_sim and RECORD_FRAME:
                if use_subspace:
                    output_dir = os.path.join(args.output_dir, system.problem_name, args.output_nn_dir,
                                          args.subspace_name, args.framesFolder)
                else:
                    output_dir = os.path.join(args.output_dir, system.problem_name, args.framesFolder)
                if FRAME ==1:
                    ensure_dir_exists(output_dir)
                if FRAME < NUM_SNAPSHOTS+1:
                    filename = os.path.join(output_dir, f"frame_{FRAME:05d}.png")
                    print(filename)

                    ps.screenshot(filename, transparent_bg=False)
            FRAME += 1

        # print energy
        if eval_energy_every:
            E = eval_potential_energy(system_def, int_state['q_t'])
            E_str = f"Potential energy: {E}"
            psim.TextUnformatted(E_str)

        _, eval_energy_every = psim.Checkbox("eval every", eval_energy_every)
        psim.SameLine()
        _, update_viz_every = psim.Checkbox("viz every", update_viz_every)

        if psim.Button("reset"):
            reset_state()

        psim.SameLine()

        if psim.Button("stop velocity"):
            integrators.update_state(int_opts, int_state, int_state['q_t'], with_velocity=False)

        psim.SameLine()

        _, run_sim = psim.Checkbox("run simulation", run_sim)
        psim.SameLine()
        if run_sim or psim.Button("single step"):

            # all-important timestep happens here
            int_state = integrators.timestep(system,
                                             system_def,
                                             int_state,
                                             int_opts,
                                             subspace_fn=subspace_fn,
                                             subspace_domain_dict=subspace_domain_dict,
                                             collect_velo_snapshots=RECORD_SNAPSHOTS,
                                             file_name=os.path.join(args.output_dir, system.problem_name, args.snapsFolder, f"snap_{FRAME-1:05d}"))

    ps.set_user_callback(main_loop)
    ps.show()


if __name__ == '__main__':
    main()