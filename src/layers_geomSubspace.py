
import os
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.scipy.linalg import expm
from scipy.linalg import logm
import typing
import jax.lax as lax
from jax import random, vmap
# from torch.utils.tensorboard import SummaryWriter  # launch/visualize logs
# for the learning optimizer
from jax.example_libraries import optimizers
import jax
from tqdm import tqdm
from typing import Any
import pickle
# from torch.utils.tensorboard import SummaryWriter
from flax.training import train_state, checkpoints
from utils import ensure_dir_exists
import json
import matplotlib.pyplot as plt
def str_to_act(s):
    d = {
        'ReLU': jax.nn.relu,
        'LeakyReLU': jax.nn.leaky_relu,
        'ELU': jax.nn.elu,
        'Cos': jnp.cos,
    }

    if s not in d:
        raise ValueError(f'Unrecognized activation {s}. Should be one of {d.keys()}')

    return d[s]

def get_latent_domain_dict(domain_name):
    subspace_domain_dict = {'domain_name': domain_name}

    if domain_name == 'normal':

        def domain_sample_fn(size, rngkey):
            return jax.random.normal(rngkey, size)

        def domain_project_fn(q_t, q_tm1, qdot_t):
            return q_t, q_tm1, qdot_t

        def domain_dist2_fn(q_a, q_b):
            return jnp.sum(jnp.square(q_a - q_b), axis=-1)

        subspace_domain_dict['domain_sample_fn'] = domain_sample_fn
        subspace_domain_dict['domain_project_fn'] = domain_project_fn
        subspace_domain_dict['domain_dist2_fn'] = domain_dist2_fn
        subspace_domain_dict['initial_val'] = 0.
        subspace_domain_dict['viz_entry_bound_low'] = -3.
        subspace_domain_dict['viz_entry_bound_high'] = 3.

    else:
        raise ValueError("unrecognized subspace domain name")

    return subspace_domain_dict
def model_spec_from_args(args, in_dim, out_dim):
    """
    Build the dictionary which we feed into the more general create_network
    (this dictionary can be saved to recreate the same network later)
    """

    spec_dict = {}

    # spec_dict['in_dim'] = in_dim
    # spec_dict['out_dim'] = out_dim

    spec_dict['model_type'] = args.model_type

    # Add MLP-specific args
    if spec_dict['model_type'] in ["learnGeometricalAwareSolver"]:

        spec_dict['activation'] = args.activation
        spec_dict['MLP_hidden_layers'] = args.MLP_hidden_layers
        spec_dict['MLP_hidden_layer_width'] = args.MLP_hidden_layer_width

    # Add linear-specific args
    elif spec_dict['model_type'] == "Linear":
        # TODO implement
        pass

    else:
        raise ValueError(f"unrecognized model_type {spec_dict['model_type']}")

    return spec_dict

def create_model(spec_dict, rngkey=None, base_output=None):

    if rngkey is None:
        rngkey = jax.random.PRNGKey(0)

    if spec_dict['model_type'] == 'learnGeometricalAwareSolver':

        encoder = transf2Latent_Encoder(spec_dict, rngkey)
        decoder = latent2Transf_Decoder(spec_dict, rngkey)

    else:
        raise ValueError(f"unrecognized model_type {spec_dict['model_type']}")

    # Always create the model in 32-bit, even if the system is defaulting to 64 bit.
    # Otherwise serialization things fail. Passing 64-bit inputs to the model should give
    # the expected up-conversion at evaluation time.
    model_encoder = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, encoder)
    model_decoder = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, decoder)
    print(f"\n== Created encoding network ({spec_dict['model_type']}):")
    print(model_encoder)
    print(model_decoder)
    return model_encoder, model_decoder

class PredefinedFunctionLayer(eqx.Module):
    fn: typing.Callable  # Store the global function as a callable field

    def __call__(self, x):
        return self.fn(x)

def exp_omega(omega_flat_column):
    omega_matrix = omega_flat_column.reshape(3, 3)

    # Apply expm to each matrix
    expm_matrices = expm(omega_matrix)
    # Flatten back to (9,)
    return expm_matrices.reshape(-1)


class transf2Latent_Encoder(eqx.Module):
    rot2omega_layers: typing.List[eqx.nn.Linear]
    rot_dim : int
    tranz2latent_layers: typing.List[eqx.nn.Linear]
    tranz_dim: int
    omega2latent_layers: typing.List[eqx.nn.Linear]
    activation: typing.Callable

    def __init__(self, dict, key):
        # MLP layers
        self.activation = str_to_act(dict['activation'])
        self.rot2omega_layers = []
        self.tranz2latent_layers = []
        self.omega2latent_layers = []
        self.rot_dim = dict['rot_dim']
        self.tranz_dim = dict['tranz_dim']
        # building the layers
        # First:
        # -- from rotation to angular velocity mat
        prev_width = dict['rot_dim']  # first layer has full transformation dim
        keys = jax.random.split(key, dict['MLP_hidden_layers'])  # first round of keys
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have omega dim, otherwise a user pre-defined 'width'
            next_width = dict['omega_dim'] if is_last else dict['MLP_hidden_layer_width']

            self.rot2omega_layers.append(eqx.nn.Linear(prev_width, next_width, use_bias=True, key=keys[i_layer]))
            prev_width = next_width

        # -- from full translation to translation_latent vector
        key, subkey = jax.random.split(key)
        prev_width = dict['tranz_dim']  # first layer has full transformation dim
        keys = jax.random.split(key, dict['MLP_hidden_layers'])  # first round of keys
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            next_width = dict['tranz_latent_dim'] if is_last else dict['MLP_hidden_layer_width']

            self.tranz2latent_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=True, key=keys[i_layer]))
            prev_width = next_width

        # Second:
        # -- from omega matrix and translation vector to latent space
        prev_width = dict['omega_dim']  # first layer has full transformation dim
        key, subkey = jax.random.split(key)
        keys2 = jax.random.split(key, dict['MLP_hidden_layers'])  # second round of keys
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have latent dim, otherwise a user pre-defined 'width'
            next_width = dict['rot_latent_dim'] if is_last else dict['MLP_hidden_layer_width']
            self.omega2latent_layers.append(eqx.nn.Linear(prev_width, next_width, use_bias=True, key=keys2[i_layer]))
            prev_width = next_width

    def __call__(self, y, return_details=False):
        omega = None
        # MLP layes
        # Learn omega from Rotation slice
        x = lax.dynamic_slice(y, start_indices=(0,), slice_sizes=(self.rot_dim,))
        for i_layer in range(len(self.rot2omega_layers)):
            is_last = (i_layer + 1 == len(self.rot2omega_layers))
            x = self.rot2omega_layers[i_layer](x)
            if not is_last:
                x = self.activation(x)
        if return_details:
            omega = x
        # now find the rot_latent from the learnt omega and z
        for i_layer in range(len(self.omega2latent_layers)):
            is_last = (i_layer + 1 == len(self.omega2latent_layers))
            x = self.omega2latent_layers[i_layer](x)
            if not is_last:
                x = self.activation(x)
        # x is the latent part that reflects rotation

        # Learn translation_latent from Translation slice
        z = lax.dynamic_slice(y, start_indices=(self.rot_dim,), slice_sizes=(self.tranz_dim,))
        for i_layer in range(len(self.tranz2latent_layers)):
            is_last = (i_layer + 1 == len(self.tranz2latent_layers))
            z = self.tranz2latent_layers[i_layer](z)
            if not is_last:
                z = self.activation(z)
        # z is the latent part that reflects translation
        
        if return_details:
            return omega, jnp.concatenate([x, z], axis=0)  # omega, latent subspace
        else:
            return jnp.concatenate([x, z], axis=0)  # latent subspace


class latent2Transf_Decoder(eqx.Module):
    activation: callable
    latent2omega_layers: typing.List[eqx.nn.Linear]
    rot_latent_dim: int
    latent2tranz_layers: typing.List[eqx.nn.Linear]
    tranz_latent_dim: int
    def __init__(self, dict, rngkey):
        self.activation = jax.nn.relu
        self.latent2omega_layers = []
        self.rot_latent_dim = dict['rot_latent_dim']
        self.latent2tranz_layers = []
        self.tranz_latent_dim = dict['tranz_latent_dim']

        prev_width = dict['rot_latent_dim']
        # first, from latent to omega layers
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have output dim, otherwise 'width'
            next_width = dict['omega_dim'] if is_last else dict['MLP_hidden_layer_width']

            rngkey, subkey = jax.random.split(rngkey)
            self.latent2omega_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

        # second, latent to translation layers
        prev_width = dict['tranz_latent_dim']
        for t_layer in range(dict['MLP_hidden_layers']):
            is_last = (t_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have output dim, otherwise 'width'
            next_width = dict['tranz_dim'] if is_last else dict['MLP_hidden_layer_width']

            rngkey, subkey = jax.random.split(rngkey)
            self.latent2tranz_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

    def __call__(self, y, return_details=False):
        omega = None
        # from latent to omega
        x = lax.dynamic_slice(y, start_indices=(0,), slice_sizes=(self.rot_latent_dim,))
        for i_layer in range(len(self.latent2omega_layers)):
            is_last = (i_layer + 1 == len(self.latent2omega_layers))
            x = self.latent2omega_layers[i_layer](x)

            if not is_last:
                x = self.activation(x)
        # omega = x
        if return_details:
            omega = x
        # one exponential layer to compute rotation from omega
        rotations = exp_omega(x)

        # translation layers
        z = lax.dynamic_slice(y, start_indices=(self.rot_latent_dim,), slice_sizes=(self.tranz_latent_dim,))
        for i_layer in range(len(self.latent2tranz_layers)):
            is_last = (i_layer + 1 == len(self.latent2tranz_layers))
            z = self.latent2tranz_layers[i_layer](z)
            if not is_last:
                z = self.activation(z)
        # translation = z

        if return_details:
            return rotations, z   # rotation, translation
        else:
            # combine between rot and translation into complete stacked linear transformations
            return jnp.concatenate([rotations, z], axis=0)  # full transformation

class Autoencoder(eqx.Module):
    encoder: transf2Latent_Encoder
    decoder: latent2Transf_Decoder

    def __init__(self, dict, rngkey=None):

        if rngkey is None:
            rngkey = jax.random.PRNGKey(0)

        encoder = transf2Latent_Encoder(dict, rngkey)
        rngkey, _ = random.split(rngkey)
        decoder = latent2Transf_Decoder(dict, rngkey)

        self.encoder = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, encoder)
        self.decoder = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, decoder)

    def __call__(self, y, return_details=False):
        if return_details:
            omega, latent = self.encoder(y, return_details)
            rot, tranz = self.decoder(latent, return_details)
            return omega, rot, tranz
        else:
            latent = self.encoder(y)
            transform_hat = self.decoder(latent)
            return transform_hat


"""
Classes and functions for training purpose
"""


class GenerateCallback:
    def __init__(self, input_transf, every_n_epochs=1):
        super().__init__()
        self.input_transformations = input_transf  # motion transformations for reconstruction measure during training
        self.every_n_epochs = every_n_epochs  # Only save every N epochs

    def log_generations(self, model, state, logger, epoch):
        if epoch % self.every_n_epochs == 0:
            # TODO reconstruct model from state (?)
            reconst_transf = model(self.input_transformations)
            reconst_transf = jax.device_get(reconst_transf)

            # TODO: print some measures
            print("TODO: choose what to do with the reconstructed info!")



class TrainerModule:
    """
    A Trainer for Equinox models using jax.example_libraries.optimizers.
    """
    model: Autoencoder

    def __init__(self, model, nn_dict, rng, loader_input_index, log_dir: str, lr: float = 1e-3):
        """
        Args:
            model (eqx.Module): The Equinox model to train.
            lr (float): Learning rate.
            log_dir (str): Directory for logs and checkpoints.
        """
        self.model = Autoencoder(nn_dict, rng)
        self.lr = lr
        self.loader_input_index = loader_input_index
        # adam returns init_fun, update_fun, get_parameters
        self.optimizer_init, self.optimizer_update, self.get_params = optimizers.adam(lr)
        # opt_state tracks model optimization
        params, _ = eqx.partition(model, eqx.is_array)
        self.opt_state = self.optimizer_init(eqx.filter(self.model, eqx.is_array))
        self.step = 0  # training step
        self.log_dir = log_dir # dir to safe logs
        ensure_dir_exists(self.log_dir)
        # self.logger = SummaryWriter(log_dir=self.log_dir)

        # tracking loss value during training
        self.training_energy_tracker = []
        self.training_epoch = []

    def compute_batched_loss(self, model, data):
        """
        Mean Squared Error loss.
        """
        # transfors = jax.device_put(jnp.array(data[self.loader_input_index].numpy()))
        batched_model = vmap(model, in_axes=0)
        preds = batched_model(data)
        return jnp.mean((preds - data) ** 2)

    def compute_batched_detailed_loss(self, model, data, omega, rot, tranz):
        batched_model = vmap(model, in_axes=(0, None))
        omega_pred, rot_pred, tranz_pred = batched_model(data, True)
        return jnp.mean((omega_pred - omega) ** 2) + jnp.mean((rot_pred - rot) ** 2) + jnp.mean((tranz_pred - tranz) ** 2)


    @eqx.filter_jit()
    def train_step(self, opt_state, transfrm, omega=None, rot=None, tranz=None, datailed_loss=False):
        """
        JIT-compiled training step using jax.example_libraries.optimizers.

        returns current value of loss and updated opt_state
        """
        def loss_fn(par):
            # param: the freshly adjusted trainable parameters
            # re-build/update model from trainable and static parameters
            model = eqx.combine(par, eqx.filter(self.model, lambda m: not eqx.is_array(m)))

            if datailed_loss:
                return self.compute_batched_detailed_loss(model, transfrm, omega, rot, tranz)
            else:
                return self.compute_batched_loss(model, transfrm)

        params = self.get_params(opt_state)  # get all trainable parameters
        loss, grads = jax.value_and_grad(loss_fn)(params)
        # update optimizer state
        opt_state = self.optimizer_update(self.step, grads, opt_state)
        return opt_state, loss

    def train_epoch(self, train_loader, omega_idx_=2, rot_idx=3, tranz_idx=4, detailed_loss=False):
        """
        Runs one epoch of training.
        """
        losses = []
        if detailed_loss:
            for batch in train_loader:
                data = jax.device_put(jnp.array(batch[self.loader_input_index].numpy()))

                omega_batch = jax.device_put(jnp.array(batch[omega_idx_].numpy()))
                rot_batch = jax.device_put(jnp.array(batch[rot_idx].numpy()))
                tranz_batch = jax.device_put(jnp.array(batch[tranz_idx].numpy()))

                self.opt_state, loss = self.train_step(self.opt_state, data,
                                                       omega_batch, rot_batch, tranz_batch, datailed_loss=True)
                losses.append(jax.device_get(loss))
                self.step += 1
        else:
            for batch in train_loader:
                batch = jax.device_put(jnp.array(batch[self.loader_input_index].numpy()))
                self.opt_state, loss = self.train_step(self.opt_state, batch)
                losses.append(jax.device_get(loss))
                self.step += 1
        avg_loss = np.mean(losses)
        return avg_loss

    def evaluate(self, test_model, test_loader):
        """
        Evaluates the model on all input of data_loader and return average loss.
        """
        total_loss = 0
        total_samples = 0
        for batch in test_loader:
            batch = jax.device_put(jnp.array(batch[self.loader_input_index].numpy()))
            batch_size = batch.shape[0]

            loss = self.compute_batched_loss(test_model, batch)
            total_loss += jax.device_get(loss) * batch_size
            total_samples += batch_size
        return total_loss / total_samples

    def train(self, train_loader, val_loader, args, nn_dict, epochs, save_every=10):
        """
        Full training loop on given number of epochs.
        """
        best_val_loss = float('inf')
        for epoch in tqdm(range(1, epochs + 1)):
            # update self.opt_state
            avg_loss = self.train_epoch(train_loader, detailed_loss=True)

            if epoch % save_every == 0:
                # get updated autoencoder model information
                params = self.get_params(self.opt_state)
                model_current = eqx.combine(params, eqx.filter(self.model, lambda m: not eqx.is_array(m)))
                val_loss = self.evaluate(model_current, val_loader)
                self.training_energy_tracker.append(val_loss)
                self.training_epoch.append(epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(args, nn_dict, epoch)

    def save_model(self, args, nn_dict, epoch):
        """
        Saves model parameters and optimizer state.
        """
        # get updated autoencoder model information
        params = self.get_params(self.opt_state)
        model_current = eqx.combine(params, eqx.filter(self.model, lambda m: not eqx.is_array(m)))

        # save the model parameters as Pytree: .eqx
        eqx.tree_serialise_leaves(os.path.join(self.log_dir, f"checkpoint_{epoch}_autoencoder.eqx"), model_current)

        # store the nn dict used to construct the autoencoder: .json
        with open(os.path.join(self.log_dir, f'checkpoint_{epoch}.json'), 'w') as json_file:
            json_file.write(json.dumps(nn_dict))

        # more information as .np
        np.save(
            os.path.join(self.log_dir, f"info_{epoch}"), {  # TODO all required info
                'system': args.system_name,
                'problem_name': args.problem_name,
                'activation': args.activation,
                'subspace_domain_type': args.subspace_domain_type,
                'subspace_dim': nn_dict['rot_latent_dim']+nn_dict['tranz_latent_dim'],
                't_schedule_final': self.lr,
                'model_type': args.model_type,
                'rot_subspace_dim': args.rot_subspace_dim,
                'tranz_subspace_dim': args.tranz_subspace_dim,
            })


    def load_model(self, epoch):
        """
        Loads model parameters and optimizer state.
        """
        # load autoencoder dictionary .json
        with open(os.path.join(self.log_dir, f'checkpoint_{epoch}.json'), 'r') as json_file:
            autoencoder_model_dict = json.loads(json_file.read())

        autoencoder = Autoencoder(autoencoder_model_dict)
        _, params_static = eqx.partition(autoencoder, eqx.is_array)

        # load autoencoder parameters (optimized weights and bias) .json
        model_params = eqx.tree_deserialise_leaves(os.path.join(self.log_dir, f"checkpoint_{epoch}_autoencoder.eqx"), autoencoder)

        return model_params, params_static


def evaluate_autoencoder(args, model, nn_dict, rng, train_loader, val_loader, test_loader, epochs, checkpoint_path, loader_input_index, pretrained=False):
    # Create a trainer module with specified hyperparameters
    trainer = TrainerModule(model, nn_dict, rng, loader_input_index, checkpoint_path)

    if not pretrained:
        trainer.train(train_loader, val_loader, args, nn_dict, epochs)

        # show loss behaviour during training
        plt.plot(trainer.training_epoch, trainer.training_energy_tracker, 'go--', label='energy vals')
        plt.xlabel('training iter')
        plt.ylabel('Energy')
        plt.yscale('log')

        plt.savefig(os.path.join(trainer.log_dir, "loss_while_training.png"))
        plt.close()

        model_params, model_static = eqx.partition(trainer.model, eqx.is_array)

    else:
        model_params, model_static = trainer.load_model(epochs)
    optimized_autoencoder = eqx.combine(model_params, model_static)

    test_loss = trainer.evaluate(optimized_autoencoder, test_loader)

    return optimized_autoencoder, test_loss