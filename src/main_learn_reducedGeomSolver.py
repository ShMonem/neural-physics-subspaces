import sys, os
from functools import partial
import subprocess

import argparse, json
import pickle
import numpy as np
import scipy
import scipy.optimize

import jax
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize
import jax.nn
from jax.example_libraries import optimizers
from jax import debug
import equinox as eqx
# Imports from this project
import utils
import config_geomSubspace as config
import layers_geomSubspace as layers
import os
import matplotlib.pyplot as plt


SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

import jax
import jax.numpy as jnp
import equinox as eqx

import typing
from main_arrange_snapshots import read_train_and_test_snapshots, read_train_snapshots

def main():

    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_system_args(parser)
    config.add_learning_args(parser)
    config.add_training_args(parser)
    config.add_jax_args(parser)

    #  Arguments specific to this program
    parser.add_argument("--output_dir", type=str, default="../output")
    parser.add_argument("--output_nn_dir", type=str, default="nn_solvers")
    parser.add_argument("--snapshots_input_dir", type=str, default="snapshots")
    parser.add_argument("--output_prefix", type=str, default="")

    # network defaults
    parser.add_argument("--model_type", type=str, default='learnGeometricalAwareSolver')
    parser.add_argument("--activation", type=str, default='ReLU')
    parser.add_argument("--subspace_dim", type=int, default=9)        # adjust --
    parser.add_argument("--subspace_domain_type", type=str, default='V2latent2T')
    parser.add_argument("--numSnapshots", type=int, default=9990)
    parser.add_argument("--snapshots_mats_available", type=bool, default=True)

    # repulsion energy # TODO: is necessary?!
    parser.add_argument("--expand_type", type=str, default="iso")
    parser.add_argument("--sigma_scale", type=float, default=1.0)  # adjust --
    parser.add_argument("--weight_expand", type=float, default=1.0)  # adjust --


    # Parse arguments
    args = parser.parse_args()

    # build network dictionary

    # Process args
    config.process_jax_args(args)
    # jax init
    _ = jnp.zeros(())

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)
    target_dim = system.dim

    # Subspace domain
    subspace_domain_dict = layers.get_latent_domain_dict(args.subspace_domain_type)

    # Build an informative output name
    network_filename_base = f"{args.output_prefix}neural_subspace_{args.activation}_{args.system_name}_{args.problem_name}"
    utils.ensure_dir_exists(args.output_dir)
    utils.ensure_dir_exists(os.path.join(args.output_dir,args.problem_name, args.output_nn_dir))

    # some random state
    rngkey = jax.random.PRNGKey(0)
    rngkey, subkey = jax.random.split(rngkey)

    # Read or upload simulation snapshots
    shift = 25  # shifts between traint and test snapshots
    jump = 50  # jumps between snapshots frames
    case_extension = "numsnapshots_" + f"{args.numSnapshots:05d}"+ "_jump_"+f"{jump:05d}" + "_shift_" + f"{shift:05d}"
    if args.snapshots_mats_available:
        sim = read_train_snapshots(args, shift, jump)
        with open(os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "_nn_dict"), 'rb') as f:
            nn_dict = pickle.load(f)
    else:
        sim, nn_dict = read_train_and_test_snapshots(args, shift=shift, jump=jump,
                                                 file_patern_p1= "snapshot",
                                                 file_patern_ext="_" + system.problem_name + "_.npz",
                                                 store_npy=True)
    print("Training snapshots have been loaded, network dict was constructed:\n", nn_dict)

    # Initiate  models (encoder and decoder)
    rngkey, subkey = jax.random.split(rngkey)
    model_pos2latent, model_latent2T = layers.create_model(nn_dict, subkey)
    # model1 is the NN learning angular velocity from positional velocity, and finds latent space
    model_p2l_params, model_p2l_static = eqx.partition(model_pos2latent, eqx.is_array)
    # model2 is the NN learning rotation from latent
    model_l2T_params, model_l2T_static = eqx.partition(model_latent2T, eqx.is_array)

    # Define optimizer for the encoder
    opt = optimizers.adam(1e-3)
    all_params = (model_p2l_params, model_l2T_params)
    model_static = (model_p2l_static, model_l2T_static)
    opt_state = opt.init_fn(all_params)

    # Dictionary of extra external parameters which are non-constant but not updated by the optimizer
    ex_params = {'t_schedule': jnp.array(0.0)}

    def get_full_transformation(decode_model_params, x, cond_params, t_schedule):
        decoder_model = eqx.combine(decode_model_params, model_l2T_static)
        return decoder_model(jnp.concatenate((x.squeeze(), cond_params), axis=0))[1] #---> TODO
    
    def get_latent(encode_model_params, q ):
        encoder_model = eqx.combine(encode_model_params, model_p2l_static)
        return encoder_model(q)  # omega_predicted, alpha  ----> TODO

    def sample_system_and_Epot(system_def, encoder_model_params, decoder_model_params , ex_params, rngkey):
        system_def = system_def.copy()
        t_schedule = ex_params['t_schedule']

        # Sample a conditional values
        rngkey, subkey = jax.random.split(rngkey)
        cond_params = system.sample_conditional_params(system_def, subkey, rho=t_schedule)
        system_def['cond_param'] = cond_params

        subspace_f = lambda zz: get_full_transformation(decoder_model_params, zz, cond_params, t_schedule)

        # Sample latent value
        @jax.jit
        def sample_q(dim, rngkey):
            random_col = jax.random.randint(rngkey, shape=(), minval=0, maxval=dim)
            return jax.lax.dynamic_index_in_dim(sim.linear, random_col, axis=1)

        rngkey, subkey = jax.random.split(rngkey)
        rand_q = sample_q(args.numSnapshots, subkey)
        z = get_latent(encoder_model_params, rand_q)[1].squeeze()

        # Map the latent state to config space
        q = subspace_f(z)
        E_pot = system.potential_energy(system_def, q)

        return z, cond_params, q, E_pot

    def batch_repulsion(z_batch, q_batch, t_schedule):
        # z_batch: [B,Z]
        # q_batch: [B,Q]
        # Returns [B] vector sum along columns
        DIST_EPS = 1e-8

        stats = {}

        def q_dists_one(q):
            return jax.vmap(partial(system.kinetic_energy, system_def,
                                    q))(q_batch - q[None, :]) + DIST_EPS

        all_q_dists = jax.vmap(q_dists_one)(q_batch)  # [B,B]

        def z_dists_one(z):
            z_delta = z_batch - z[None, :]
            return jnp.sum(z_delta * z_delta, axis=-1)

        all_z_dists = jax.vmap(z_dists_one)(z_batch)  # [B,B]

        if args.expand_type == 'iso':

            factor = jnp.log(t_schedule * args.sigma_scale * all_z_dists + DIST_EPS) - jnp.log(jnp.expand_dims(all_q_dists, axis=-1))
            repel_term = jnp.sum(jnp.square(0.5 * factor), axis=-1)

            stats['mean_scale_log'] = jnp.mean(-factor)

        else:
            raise ValueError("expand type should be 'iso'")

        return repel_term, stats

    # Create an optimizer
    print(f"Creating optimizer...")

    def step_func(i_iter):
        out = args.lr * (args.lr_decay_frac**(i_iter // args.lr_decay_every))
        return out

    def batch_loss_fn(params, ex_params, rngkey):

        encoder_params, decoder_params = params
        t_schedule = ex_params['t_schedule']

        subkey_b = jax.random.split(rngkey, args.batch_size)
        z_samples, cond_samples, q_samples, E_pots = jax.vmap(
            partial(sample_system_and_Epot, system_def, encoder_params, decoder_params, ex_params,))(subkey_b)

        expand_loss, repel_stats = batch_repulsion(z_samples, q_samples, t_schedule)
        expand_loss = expand_loss * args.weight_expand  # eq(2): impose isometry

        loss_dict = {}
        loss_dict['E_pot'] = E_pots  # eq(1): potential energy
        loss_dict['E_expand'] = expand_loss

        out_stats_b = {}
        out_stats_b.update(repel_stats)

        # sum up a total loss (mean over batch)
        total_loss = 0.
        for _, v in loss_dict.items():
            total_loss += jnp.mean(v)  # eq(3)

        return total_loss, (loss_dict, out_stats_b)

    @jax.jit
    def train_step(i_iter, rngkey, ex_params, opt_state):

        opt_params = opt.params_fn(opt_state)
        # optimize for the parameters
        (value, (loss_dict, out_stats_b)), grads = jax.value_and_grad(batch_loss_fn,
                                                                      has_aux=True)(opt_params,
                                                                                    ex_params,
                                                                                    rngkey)
        # Update the optimizer states and parameters for both models
        opt_state = opt.update_fn(i_iter, grads, opt_state)
        # out_stats_b currently unused

        return value, loss_dict,  opt_state, out_stats_b

    print(f"Training...")

    # Parameters tracked for each stat round
    losses = []
    n_sum_total = 0
    loss_term_sums = {}
    i_save = 0
    mean_scale_log = []

    energy_itr = []
    ## Main training loop: optimizing the neural network parameters (weights and bias arrays for 5 layers)
    for i_train_iter in range(args.n_train_iters):

        ex_params['t_schedule'] = i_train_iter / args.n_train_iters

        rngkey, subkey = jax.random.split(rngkey)
        # optimize for \theta parameters in opt_state(all_params)
        loss, loss_dict, opt_state, out_stats = train_step(i_train_iter, subkey, ex_params,
                                                           opt_state)

        # track statistics
        loss = float(loss)
        losses.append(loss)
        if 'mean_scale_log' in out_stats:
            mean_scale_log.append(out_stats['mean_scale_log'])

        for k in loss_dict:
            if k not in loss_term_sums:
                loss_term_sums[k] = 0.
            loss_term_sums[k] += jnp.sum(loss_dict[k])

        n_sum_total += args.batch_size

        def save_model(this_name):

            network_filename_pre = os.path.join(args.output_dir, args.problem_name, args.output_nn_dir)
            utils.ensure_dir_exists(network_filename_pre)
            network_filename_pre = os.path.join(network_filename_pre,
                                                network_filename_base) + this_name
            print(f"Saving result to {network_filename_pre}")

            encoder_model = eqx.combine(model_p2l_params, model_p2l_static)
            decoder_model = eqx.combine(model_l2T_params, model_l2T_static)

            eqx.tree_serialise_leaves(network_filename_pre + "_encoder.eqx", encoder_model)
            eqx.tree_serialise_leaves(network_filename_pre + "_decoder.eqx", decoder_model)
            with open(network_filename_pre + '.json', 'w') as json_file:
                json_file.write(json.dumps(nn_dict))
            np.save(
                network_filename_pre + "_info", {
                    'system': args.system_name,
                    'problem_name': args.problem_name,
                    'activation': args.activation,
                    'subspace_domain_type': args.subspace_domain_type,
                    'subspace_dim': args.subspace_dim,
                    't_schedule_final': ex_params['t_schedule'],
                })

            print(f"  ...done saving")

        if i_train_iter % args.report_every == 0:

            print(
                f"\n== iter {i_train_iter} / {args.n_train_iters}  ({100. * i_train_iter / args.n_train_iters:.2f}%)"
            )
            # print some statistics
            mean_loss = np.mean(np.array(losses))
            print(f"      loss: {mean_loss:.6f}")
            energy_itr.append(loss)
            opt_params = opt.params_fn(opt_state)
            model_p2l_params, model_l2T_params = opt_params

            for k in loss_term_sums:
                print(f"   {k:>30}: {(loss_term_sums[k] / n_sum_total):.6f}")

            print("  Stats:")
            if len(mean_scale_log) > 0:
                mean_scale = jnp.exp(jnp.mean(jnp.stack(mean_scale_log)))
                print(
                    f"    mean metric stretch: {mean_scale:g}    (scaled so current target is 1.)")

            # save
            out_name = f"_save{i_save:04d}"
            if args.expand_type == 'iso':
                scale_combined = args.sigma_scale * ex_params['t_schedule']
                out_name += f'_sigma{scale_combined:g}'
            save_model(out_name)
            i_save += 1

            # reset statistics
            losses = []
            n_sum_total = 0
            loss_term_sums = {}
            mean_scale_log = []

    # save results one last time
    save_model("_final")


    plt.plot(energy_itr, 'go--', label='energy vals')
    # plt.yscale
    plt.xlabel('training iter')
    plt.ylabel('Energy')
    plt.savefig(os.path.join(args.output_dir, args.problem_name, args.output_nn_dir,
                             f"energy_autoencoder_training_"+system.problem_name+"_.png"))


    plt.show()

if __name__ == '__main__':
    main()