import sys, os
from functools import partial
import subprocess

import argparse, json

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

# jax.disable_jit(disable=True)

def main():

    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_system_args(parser)
    config.add_learning_args(parser)
    config.add_training_args(parser)
    config.add_jax_args(parser)

    ###  Arguments specific to this program
    parser.add_argument("--output_dir", type=str, default="../output")
    parser.add_argument("--snapshots_input_dir", type=str, default="snapshots")
    parser.add_argument("--output_prefix", type=str, default="")

    ### TODO: whant to add more training arguments?
    # network defaults
    parser.add_argument("--model_type", type=str, default='learnGeometricalAwareSolver')
    parser.add_argument("--activation", type=str, default='ReLU')
    parser.add_argument("--subspace_dim", type=int, default=2)        # adjust --
    parser.add_argument("--subspace_domain_type", type=str, default='velo2T')

    # Parse arguments
    args = parser.parse_args()

    # build network dictionary

    # Process args
    config.process_jax_args(args)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)

    # Build an informative output name TODO: edit
    network_filename_base = f"{args.output_prefix}neural_subspace_{args.activation}_{args.system_name}_{args.problem_name}_learnRate{0}"

    utils.ensure_dir_exists(args.output_dir)
    utils.ensure_dir_exists(args.snapshots_input_dir)
    # some random state
    rngkey = jax.random.PRNGKey(0)
    rngkey, subkey = jax.random.split(rngkey)

    # configure other parameters
    # target_dim = system.dim
    base_state = system_def['interesting_states'][0, :]
    rngkey, subkey = jax.random.split(rngkey)
    
    # Construct the learned subspace operator
    # in_dim = args.subspace_dim + system.cond_dim
    # model_spec = layers.model_spec_from_args(args, in_dim, target_dim)

    def test_model_accuracy(model, test_pos_velos, test_omega):
        # forward pass
        omega_predicted = model(test_pos_velos)
        return jnp.mean(jnp.square(omega_predicted - test_omega))

    def test_autocoder_accuracy(encoder, decoder, test_decode_pos_snaps, test_decode_omega_snaps, test_decode_T_snaps):
        # forward pass
        omega_predicted, alpha = encoder(test_decode_pos_snaps)
        omega_reconstructed, transform_predicted = decoder(alpha)

        accuracy = jnp.mean(jnp.square(omega_predicted - test_decode_omega_snaps)) +\
                    jnp.mean(jnp.square(omega_reconstructed - test_decode_omega_snaps)) + \
                   jnp.mean(jnp.square(transform_predicted - test_decode_T_snaps))

        return accuracy
    # Define the training step
    @jax.jit
    def train_step(i_iter, pos_snaps, omega_snaps, transf_snaps, rot_snaps, t_snaps,  opt_state_encoder, opt_state_decoder):

        encoder_params = opt.params_fn(opt_state_encoder)
        decoder_params = opt.params_fn(opt_state_decoder)

        def loss_fn_encoder(encoder_pars):
            return encoder_energy(encoder_pars, pos_snaps, omega_snaps)

        def loss_fn_decoder(decoder_pars, alpha_):
            return decoder_energy(decoder_pars, alpha_, omega_snaps, transf_snaps, rot_snaps, t_snaps)

        # optimize for the parameters
        (encoder_loss, latent), encoder_grads = jax.value_and_grad(loss_fn_encoder, has_aux=True)(encoder_params)
        decoder_loss, decoder_grads = jax.value_and_grad(loss_fn_decoder, has_aux=False)(decoder_params, latent)

        # Update the optimizer states and parameters for both models
        opt_state_encoder = opt.update_fn(i_iter, encoder_grads, opt_state_encoder)
        opt_state_decoder = opt.update_fn(i_iter, decoder_grads, opt_state_decoder)

        return encoder_loss + decoder_loss, opt_state_encoder, opt_state_decoder

    def decoder_energy(decoder_params, alpha, omega_snapshots, transf_snaps, rot_snaps, t_snaps):
        """
        Compute the total loss for the encoder and decoder.
        """
        decoder = eqx.combine(decoder_params, model_l2r_static)

        omega_r, transf_p, rot_p, t_p= decoder(alpha)
        # Decoder loss: Difference between reconstructed omega and ground truth
        loss_omega = jnp.mean(jnp.square(omega_r - omega_snapshots))
        loss_transf = jnp.mean(jnp.square(transf_p - transf_snaps))

        loss_rot = jnp.mean(jnp.square(rot_p - rot_snaps))
        loss_t = jnp.mean(jnp.square(t_p - t_snaps))

        return loss_omega + loss_transf + loss_rot + loss_t

    def encoder_energy(encoder_params, positions_snapshots, omega_snapshots):
        """
        Compute the total loss for the encoder and decoder.
        """
        encoder = eqx.combine(encoder_params, model_p2l_static)

        omega_predicted, alpha = encoder(positions_snapshots)
        # Encoder loss: Difference between predicted omega and ground truth
        encoder_loss = jnp.mean(jnp.square(omega_predicted - omega_snapshots))

        return encoder_loss, alpha

    class body():
        def __int__(self):
            self.id = -1
            self.linear = None
            self.pos = None
            self.angular = None
            self.omega = None
            self.rot = None
            self.tranz = None
            self.fullT = None

            self.linear_test = None
            self.pos_test = None
            self.omega_test = None
            self.rot_test = None
            self.tranz_test = None
            self.fullT_test = None

            # parameters
    number_snapshots = 9990
    number_training_snapshots = 500
    number_test_snapshots = 700

    # Read snapshots data
    def read_train_and_test_snapshots(file_patern_p1="snapshot", file_patern_ext="_body_0.npz"):
        character = body()
        snap_posV = []
        snap_pos = []
        snap_omega =[]
        snap_rot = []
        test_snap_posV = []
        test_snap_pos = []
        test_snap_omega = []
        test_snap_rot = []
        snap_tranz = []
        snap_fullT = []
        test_snap_tranz = []
        test_snap_fullT = []

        shift = 25
        count = 0
        for s in range(1, number_snapshots+1):
            if s % 50 == 0:
                data = np.load(os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                                            file_patern_p1+f"{s:05d}"+file_patern_ext))
                snap_posV.append(data['vel'])
                snap_pos.append(data['pos'])
                snap_omega.append(data['omega_mat'])
                snap_rot.append(data['rot'])
                snap_tranz.append(data['tranz'])
                snap_fullT.append(data['full_transform'])

                test_data = np.load(os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                                                 file_patern_p1+f"{s+shift:05d}"+file_patern_ext))
                test_snap_posV.append(test_data['vel'])
                test_snap_pos.append(test_data['pos'])
                test_snap_omega.append(test_data['omega_mat'])
                test_snap_rot.append(test_data['rot'])
                test_snap_tranz.append(data['tranz'])
                test_snap_fullT.append(data['full_transform'])
                count += 1

        character.linear = jnp.array(snap_posV).reshape(count, -1).T    # (Sum 3V_i , K)  (i <= n bodies)
        character.pos = jnp.array(snap_pos).reshape(count, -1).T    # (Sum 3V_i , K)
        character.rot = jnp.array(snap_rot).reshape(count, -1).T        # (9n, K)
        character.omega = jnp.array(snap_omega).reshape(count, -1).T  # (9n, K)
        character.tranz = jnp.array(snap_tranz).reshape(count, -1).T  # (3n, K)
        character.fullT = jnp.array(snap_fullT).reshape(count, -1).T  # (12n, K)

        character.linear_test = jnp.array(test_snap_posV).reshape(count, -1).T    # (Sum 3V_i , K)
        character.pos_test = jnp.array(test_snap_pos).reshape(count, -1).T  # (Sum 3V_i , K)
        character.rot_test = jnp.array(test_snap_rot).reshape(count, -1).T
        character.omega_test = jnp.array(test_snap_omega).reshape(count, -1).T
        character.tranz_test = jnp.array(test_snap_tranz).reshape(count, -1).T  # (3n, K)
        character.fullT_test = jnp.array(test_snap_fullT).reshape(count, -1).T  # (9n, K)

        return character
    sim = read_train_and_test_snapshots(file_patern_p1="snapshot", file_patern_ext="_"+system.problem_name+"_.npz")

    # Construct the learned subspace operator dictionary
    nn_dict = {}
    nn_dict['model_type'] = args.model_type
    nn_dict['activation'] = args.activation
    nn_dict['MLP_hidden_layers'] = args.MLP_hidden_layers  # 3
    nn_dict['MLP_hidden_layer_width'] = args.MLP_hidden_layer_width
    nn_dict['vel_dim'] = sim.linear.shape[0]  # 3V
    nn_dict['pos_dim'] = sim.pos.shape[0]  # 3V
    nn_dict['rot_dim'] = sim.rot.shape[0]  # 9
    nn_dict['omega_dim'] = sim.omega.shape[0]  # 9
    nn_dict['tranz_dim'] = sim.tranz.shape[0]  # 9
    nn_dict['latent_dim'] = args.subspace_dim  # TODO: to adjust for each case

    rngkey, subkey = jax.random.split(rngkey)
    model_pos2latent, model_latent2rot = layers.create_model(nn_dict, subkey)
    # model1 is the NN learning angular velocity from positional velocity, and finds latent space
    model_p2l_params, model_p2l_static = eqx.partition(model_pos2latent, eqx.is_array)
    # model2 is the NN learning rotation from latent
    model_l2r_params, model_l2r_static = eqx.partition(model_latent2rot, eqx.is_array)
    # Define optimizer for the encoder
    # import optax
    opt = optimizers.adam(1e-3)
    # all_params = (model_p2l_params, model_l2r_params)
    opt_state_encoder = opt.init_fn(model_p2l_params)
    opt_state_decoder = opt.init_fn(model_l2r_params)

    # Training loop
    energy_itr = []
    for i_train_iter in range(args.n_train_iters):
        energy_val, opt_state_encoder, opt_state_decoder = train_step(i_train_iter,
                                                            sim.linear,
                                                            sim.omega,
                                                            sim.fullT,
                                                            sim.rot,
                                                            sim.tranz,
                                                            opt_state_encoder,
                                                            opt_state_decoder)
        if i_train_iter % 2 == 0:
            print(f"Epoch {i_train_iter}, Loss: {energy_val:.8f}")
            energy_itr.append(energy_val)

        def save_model(this_name):

            network_filename_pre = os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                                            "autocoder_geometrical_solver" + this_name)
            print(f"Saving result to {network_filename_pre}")
            
            encoder_model = eqx.combine(model_p2l_params, model_p2l_static)
            decoder_model = eqx.combine(model_l2r_params, model_l2r_static)
            # model = eqx.combine(model_params, model_static)
            # eqx.tree_serialise_leaves(network_filename_pre + ".eqx", model)
            eqx.tree_serialise_leaves(network_filename_pre + "_encoder.eqx", encoder_model)
            eqx.tree_serialise_leaves(network_filename_pre + "_decoder.eqx", decoder_model)
            with open(network_filename_pre + '.json', 'w') as json_file:
                json_file.write(json.dumps(nn_dict))
            np.save(
                network_filename_pre + "_info", {
                    'system': args.system_name,
                    'problem_name': args.problem_name,
                    'activation': args.activation,
                    'subspace_dim': args.subspace_dim,
                    'subspace_domain_type': args.subspace_domain_type,
                })

            print(f"  ...done saving to ", network_filename_pre + "_info")
    save_model("_final")

    plt.plot(energy_itr, 'go--', label='energy vals')
    # plt.yscale
    plt.xlabel('training iter')
    plt.ylabel('Energy')
    plt.savefig(os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                             f"energy_autoencoder_training_"+system.problem_name+"_.png"))


    plt.show()

    print("Testing NN on different snapshot set", test_autocoder_accuracy(model_pos2latent, model_latent2rot,
                                                                          sim.linear_test,
                                                                          sim.omega_test,
                                                                          sim.fullT_test))


main()