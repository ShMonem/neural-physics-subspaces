import jax.numpy as jnp
import jax.random as random
from jax import vmap
import jax
from jax import lax
import optax
import itertools
import equinox as eqx
import typing
from jax.example_libraries import optimizers
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm

import json
import os
from utils import ensure_dir_exists
import matplotlib.pyplot as plt

import argparse
from autoencoder_get_snapshots import read_snapshots
import autoencoder_config as config

class dof_snapshots(Dataset):
    """
    Data attributes for one body only
    """
    def __init__(self):
        super().__init__()
        self.dof = None
        self.act = None
        self.transform = None
        self.sampler = dof_sampler()
        self.len = -1

    def sampler_config(self, num_samples=100, angle_tol=1e-2, dist_tol=1e-1):
        self.sampler.set_num_samples_per_mast(num_samples)
        self.sampler.set_dist_tol(dist_tol)
        self.sampler.set_angle_tol(angle_tol)

    def dataFill(self, action_lables):
        dof, act = vmap(self.sampler.sample_pose_from_activation, in_axes=0)(action_lables)  # (num_samples*64, 6)
        dof, act = jnp.swapaxes(dof, 0, 1), jnp.swapaxes(act, 0, 1)
        q = vmap(vmap(dof_to_transformation, in_axes=0), in_axes=0)(dof)
        print("Dof samples/labels ", dof.shape)
        print("full transformation samples", q.shape)

        self.dof = dof
        self.act = act
        self.transform = q.reshape(act.shape[0], -1, 12)

        self.len = self.dof.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        dof = np.array(self.dof[idx], dtype=np.float32)     # 6
        act = np.array(self.act[idx], dtype=np.float32)     # 6
        transf = np.array(self.transform[idx], dtype=np.float32)  # 6
        return dof, act, transf



def skew_to_vec(omega_hat):
    """Convert skew-symmetric matrix to 3D vector (vee operator)."""
    return jnp.array([
        omega_hat[2, 1],
        omega_hat[0, 2],
        omega_hat[1, 0]
    ])


def extract_angular_velocity(T):
    """
    From a (4,3) transformation matrix, extract:
    - omega_hat: angular velocity matrix (3x3)
    - omega_vec: angular velocity vector (3,)
    Look modern robotics book
    From Lie theory, for any rotation matrix RâˆˆSO(3)RâˆˆSO(3), we have:
    omega = log(R)
          = (theta/ 2 sin(theta)) (R - R^T)
    theta = cos^-1 ((tr(r)-1)/2)
    """

    def log_SO3(R, eps=1e-6):
        """Compute the matrix logarithm of a SO(3) rotation matrix R."""
        trace = jnp.trace(R)
        # Prevent divide-by-zero
        theta = jnp.arccos(jnp.clip((trace - 1) / 2.0, -1.0 + eps, 1.0 - eps))

        def omega_hat_fn():
            return (theta / (2 * jnp.sin(theta))) * (R - R.T)

        omega_hat = jnp.where(jnp.abs(theta) < eps, jnp.zeros((3, 3)), omega_hat_fn())

        return omega_hat

    R = T[:3, :]
    omega_hat = log_SO3(R)
    omega_vec = skew_to_vec(omega_hat)
    return omega_hat


def dof_to_transformation(pos):

    roll, pitch, yaw, x, y, z = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
    # Rotation matrices for each axis
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll), jnp.cos(roll)]
    ])

    Ry = jnp.array([
        [jnp.cos(pitch), 0, jnp.sin(pitch)],
        [0, 1, 0],
        [-jnp.sin(pitch), 0, jnp.cos(pitch)]
    ])

    Rz = jnp.array([
        [jnp.cos(yaw), -jnp.sin(yaw), 0],
        [jnp.sin(yaw), jnp.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Compose rotation: R = Rz @ Ry @ Rx (ZYX order)
    R = Rz @ Ry @ Rx
    # assert_orthogonal(R)

    return jnp.vstack([R, jnp.array([[x, y, z]])]).reshape(-1)  # q: shape (4, 3)


def sample_random_poses(key, K, angle_tol=1e-2, dist_tol=1e-1):
    key_roll, key_pitch, key_yaw, key_xyz = random.split(key, 4)

    # Roll and yaw: Uniform(-pi, pi)
    roll = random.uniform(key_roll, shape=(K,), minval=-jnp.pi, maxval=jnp.pi)
    yaw = random.uniform(key_yaw, shape=(K,), minval=-jnp.pi, maxval=jnp.pi)
    # Pitch: Uniform(-pi/2, pi/2)
    pitch = random.uniform(key_pitch, shape=(K,), minval=-jnp.pi/2, maxval=jnp.pi/2)

    # check tolerance
    poses = jnp.stack([roll, pitch, yaw], axis=-1)
    angles_act = (jnp.abs(poses) > angle_tol).astype(int)

    # x, y, z: Uniform(-5, 5)
    translation = random.uniform(key_xyz, shape=(K, 3), minval=-2.0, maxval=2.0)
    translation_act = (jnp.abs(translation) > dist_tol).astype(int)

    # Stack all parts into (K, 6)
    poses = jnp.concatenate([poses, translation], axis=-1)
    act = jnp.concatenate([angles_act, translation_act], axis=-1)
    return poses, act


def generate_all_activation_combinations():
    # Use itertools.product to generate binary combinations
    combos = list(itertools.product([0, 1], repeat=6))
    return jnp.array(combos, dtype=jnp.int32)  # Shape: (64, 6)


class dof_sampler:

    def __init__(self):

        self.angle_tol = 0.05
        self.dist_tol = 0.05
        self.num_samples = 10

    def sampler_config(self, num_samples=100, angle_tol=1e-2, dist_tol=1e-1):
        self.set_num_samples_per_mast(num_samples)
        self.set_dist_tol(dist_tol)
        self.set_angle_tol(angle_tol)

    def set_angle_tol(self, val):
        self.angle_tol = val

    def set_dist_tol(self, val):
        self.dist_tol = val

    def set_num_samples_per_mast(self, val: int):
        self.num_samples = val

    def sample_pose_from_activation(self, activation_mask,  # jnp.array([0,1,1,0,0,1])
        key=jax.random.PRNGKey(0)        # Random key for initialization
    ):

        def sample_range(key, active, low, high, tol):
            key1, key2 = jax.random.split(key)

            def sample_positive(_):
                return jax.random.uniform(key2, minval=tol, maxval=high)

            def sample_negative(_):
                return jax.random.uniform(key2, minval=low, maxval=-tol)

            def sample_active(_):
                pick_side = jax.random.bernoulli(key1)
                return lax.cond(pick_side, sample_positive, sample_negative, operand=None)

            def sample_inactive(_):
                return jax.random.uniform(key2, minval=-tol, maxval=tol)

            return lax.cond(active, sample_active, sample_inactive, operand=None)

        angle_bounds = (-jnp.pi, jnp.pi)
        pitch_bounds = (-jnp.pi / 2, jnp.pi / 2)
        dist_bounds = (-2.0, 2.0)

        def sample_one(key):
            keys = jax.random.split(key, 6)
            roll = sample_range(keys[0], activation_mask[0], *angle_bounds, self.angle_tol)
            pitch = sample_range(keys[1], activation_mask[1], *pitch_bounds, self.angle_tol)
            yaw = sample_range(keys[2], activation_mask[2], *angle_bounds, self.angle_tol)

            x = sample_range(keys[3], activation_mask[3], *dist_bounds, self.dist_tol)
            y = sample_range(keys[4], activation_mask[4], *dist_bounds, self.dist_tol)
            z = sample_range(keys[5], activation_mask[5], *dist_bounds, self.dist_tol)
            return jnp.array([roll, pitch, yaw, x, y, z])

        K_keys = jax.random.split(key, self.num_samples)
        samples = vmap(sample_one)(K_keys)
        return samples, jnp.tile(activation_mask, (self.num_samples, 1))

class learn_dof(eqx.Module):
    q_dim: int
    dof_dim: int
    q2dof_layers: typing.List[eqx.nn.Linear]
    activation: typing.Callable

    def __init__(self, dict, key):
        # MLP layers
        self.activation = jax.nn.relu
        self.q_dim = 12  # flatten linear transformation
        self.dof_dim = 6  # poses (roll, pitch, yaw, x, y, z)
        self.q2dof_layers = []

        prev_width = self.q_dim  # first layer has full transformation dim
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have dof dim, otherwise a user pre-defined 'width'
            next_width = self.dof_dim if is_last else dict['MLP_hidden_layer_width']   # last layer 6 dof
            key, subkey = jax.random.split(key)
            self.q2dof_layers.append(eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

    def __call__(self, z):
        # get dof values
        dof = z
        for i_layer in range(len(self.q2dof_layers)):
            is_last = (i_layer + 1 == len(self.q2dof_layers))
            dof = self.q2dof_layers[i_layer](dof)
            if not is_last:
                dof = self.activation(dof)

        return dof


# class learn_action(eqx.Module):
#     q_dim: int
#     dof_dim: int
#     q2act_layers: typing.List[eqx.nn.Linear]
#     activation: typing.Callable
#
#     def __init__(self, dict, key):
#         # MLP layers
#         self.activation = jax.nn.relu
#         self.q_dim = 12  # flatten linear transformation
#         self.dof_dim = 6  # poses (roll, pitch, yaw, x, y, z)
#         self.q2act_layers = []
#
#         prev_width = self.dof_dim  # first layer coms from dof
#         for i_layer in range(dict['MLP_hidden_layers']):
#             is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
#             # last layer would have dof dim, otherwise a user pre-defined 'width'
#             next_width = self.dof_dim if is_last else dict['MLP_hidden_layer_width']  # last layer 6 dof_act
#             key, subkey = jax.random.split(key)
#             self.q2act_layers.append(eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
#             prev_width = next_width
#
#     def __call__(self, z):
#         # determine which dofs are active
#         action = z
#         for i_layer in range(len(self.q2act_layers)):
#             is_last = (i_layer + 1 == len(self.q2act_layers))
#             action = self.q2act_layers[i_layer](action)
#             if not is_last:
#                 action = self.activation(action)
#             # else:
#             #     action = jax.nn.sigmoid(action)
#
#         return action # (action > 0.5).astype(jnp.float32)



class learn_action(eqx.Module):
    q_dim: int
    dof_dim: int
    q2act_layers: typing.List[eqx.nn.Linear]
    activation: typing.Callable

    def __init__(self, dict, key):
        # MLP layers
        self.activation = jax.nn.relu
        self.q_dim = 12  # flatten linear transformation
        self.dof_dim = 6  # poses (roll, pitch, yaw, x, y, z)
        self.q2act_layers = []

        prev_width = self.dof_dim  # first layer coms from dof
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have dof dim, otherwise a user pre-defined 'width'
            next_width = self.dof_dim if is_last else dict['MLP_hidden_layer_width']  # last layer 6 dof_act
            key, subkey = jax.random.split(key)
            self.q2act_layers.append(eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

    def __call__(self, z):
        # determine which dofs are active
        action = z
        for i_layer in range(len(self.q2act_layers)):
            is_last = (i_layer + 1 == len(self.q2act_layers))
            action = self.q2act_layers[i_layer](action)
            if not is_last:
                action = self.activation(action)
            # else:
            #     action = jax.nn.sigmoid(action)

        return action # (action > 0.5).astype(jnp.float32)

class learn_link(eqx.Module):
    dof_branch: learn_dof
    action_branch: learn_action
    train: bool

    def __init__(self, dict, rngkey=None, training_phase=True):
        self.train = training_phase
        if rngkey is None:
            rngkey = jax.random.PRNGKey(0)

        dof_branch = learn_dof(dict, rngkey)
        rngkey, _ = random.split(rngkey)
        action_branch = learn_action(dict, rngkey)

        self.dof_branch = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, dof_branch)
        self.action_branch = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, action_branch)

    def __call__(self, y):

        if self.train:
            pred_q = vmap(vmap(self.dof_branch, in_axes=0), in_axes=0)(y)
            pred_action = vmap(vmap(self.action_branch, in_axes=0), in_axes=0)(pred_q)
        else:
            pred_q = vmap(self.dof_branch, in_axes=0)(y)
            pred_action = vmap(self.action_branch, in_axes=0)(pred_q)
        return pred_q, pred_action


def loss_fn(params, model, target_q, target_dof, target_act):
    # param: the freshly adjusted trainable parameters
    # re-build/update model from trainable and static parameters
    model = eqx.combine(params, eqx.filter(model, lambda m: not eqx.is_array(m)))
    # model_batched = vmap(vmap(model, in_axes=0), in_axes=0)
    pred_dof, pred_act = model(target_q)

    loss_dof = jnp.sum(optax.l2_loss(pred_dof - target_dof))/pred_dof.size    # how close to dof vals
    loss_sig = jnp.sum((vmap(vmap(optax.sigmoid_binary_cross_entropy, in_axes=(0, 0)), in_axes=(0, 0))(pred_act, target_act)))/pred_act.size

    # extra measure
    p_act = jax.nn.sigmoid(pred_act)
    p_act = (p_act > 0.5).astype(np.int32)
    loss_match = 1 - jnp.mean((p_act == target_act).astype(jnp.float32))
    # print and store
    # jax.debug.print("\nTraining loss in dof: {}, sigmoid_bin_act: {}, matching_act: {}", loss_dof, loss_sig, loss_match)
    loss_dict ={"dof": loss_dof, "sigmoid_bin_act": loss_sig, "matching_act": loss_match}
    return loss_dof + loss_sig , loss_dict


def train_dof_nn(save_every=10, epochs=200, num_train_samples_per_mask=100,
                 num_test_samples_per_mast=200, batchsize=500):
    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())
    # All possible dof activations
    action_lables = generate_all_activation_combinations()  # (64, 6)
    # dof values train snapshots
    dataset = dof_snapshots()
    dataset.sampler_config(num_samples=num_train_samples_per_mask, angle_tol=1e-2, dist_tol=1e-1)
    dataset.dataFill(action_lables)

    # loaders for each data set
    g1 = torch.Generator().manual_seed(24)
    train_dataloader = DataLoader(dataset, batch_size=batchsize, num_workers=10, pin_memory=True, shuffle=True, generator=g1)

    # test sample labels
    sampler = dof_sampler()
    sampler.sampler_config(num_samples=num_test_samples_per_mast, angle_tol=1e-2, dist_tol=1e-1)
    dof_batch, act_batch = vmap(sampler.sample_pose_from_activation, in_axes=(0, None))(action_lables, jax.random.PRNGKey(42))
    q_batch = vmap(dof_to_transformation, in_axes=0)(dof_batch.reshape(-1, 6)).reshape(action_lables.shape[0], -1, 12)

    def check_accuracy(model, dof, act, q, step):

        p_dof, p_act = model(q)

        loss__dof = np.linalg.norm(p_dof - dof)
        epo_test_dof.append(loss__dof)

        p_act = jax.nn.sigmoid(p_act)
        p_act = (p_act > 0.5).astype(np.int32)
        loss__act = 1 - np.mean((p_act == act).astype(jnp.float32))
        epo_test_act.append(loss__act)

        print("Step", step, "testing dof/act loss", loss__dof, "and loss in action", loss__act, "%")

    dict = {}
    dict['MLP_hidden_layers'] = 3
    dict['MLP_hidden_layer_width'] = 60

    # adam returns init_fun, update_fun, get_parameters
    optimizer_init, optimizer_update, get_params = optimizers.adam(1e-3)   # TODO: Try sgd
    # opt_state tracks model optimization
    model = learn_link(dict)
    params, _ = eqx.partition(model, eqx.is_array)
    opt_state = optimizer_init(eqx.filter(model, eqx.is_array))

    # Training step
    @eqx.filter_jit
    def train_eposh(step, model, train_loader, opt_state):
        losses = []
        loss_dof = []
        loss_act = []
        loss_matching = []
        for batch in train_loader:
            dof = jax.device_put(jnp.array(batch[0].numpy()))
            act = jax.device_put(jnp.array(batch[1].numpy()))
            q = jax.device_put(jnp.array(batch[2].numpy()))

            #train step
            param = get_params(opt_state)
            (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(param, model, q, dof, act)
            opt_state = optimizer_update(step, grads, opt_state)

            losses.append(jax.device_get(loss))
            loss_dof.append(jax.device_get(loss_dict["dof"]))
            loss_act.append(jax.device_get(loss_dict["sigmoid_bin_act"]))
            loss_matching.append(jax.device_get(loss_dict["matching_act"]))
            # re-build/update model from trainable and static parameters
            param = get_params(opt_state)
            model = eqx.combine(param, eqx.filter(model, lambda m: not eqx.is_array(m)))

        return model, opt_state, losses, loss_dof, loss_act, loss_matching

    step = 0

    epo_loss_dof = []
    epo_loss_act = []
    epo_loss_matching = []

    epo_test_dof = []
    epo_test_act = []
    path = "../output/dof_training"
    ensure_dir_exists(path)
    for epoch in tqdm(range(1, epochs + 1)):
        model, opt_state, losses, loss_dof, loss_sig, loss_matching = train_eposh(step, model, train_dataloader, opt_state)
        epo_loss_dof.append(np.mean(loss_dof))
        epo_loss_act.append(np.mean(loss_sig))
        epo_loss_matching.append(np.mean(loss_matching))

        print(f"\nTraining loss in dof: {np.mean(loss_dof)}, sigmoid_bin_act: {np.mean(loss_sig)}, matching_act: {np.mean(loss_matching)}" )
        check_accuracy(model, dof_batch, act_batch, q_batch, step)
        step += 1
        if epoch % save_every == 0:
            # save the model parameters as Pytree: .eqx
            eqx.tree_serialise_leaves(os.path.join(path, f"checkpoint_{epoch}_autoencoder.eqx"), model)
            # store the nn dict used to construct the autoencoder: .json
            with open(os.path.join(path, f'checkpoint_{epoch}.json'), 'w') as json_file:
                json_file.write(json.dumps(dict))

    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 8))
    x = range(1, epochs + 1)
    # Top-left
    axes[0, 0].plot(x, epo_loss_dof)
    axes[0, 0].set_title("train loss dof")

    # Top-mid
    axes[0, 1].plot(x, epo_loss_act, color='orange')
    axes[0, 1].set_title("train loss action")

    # Top-right
    axes[0, 2].plot(x, epo_loss_matching, color='blue')
    axes[0, 2].set_title("train loss matching")

    # Bottom-left
    axes[1, 0].plot(x, epo_test_dof, color='green')
    axes[1, 0].set_title("test dof error")

    # Bottom-right
    axes[1, 1].plot(x, epo_test_act, color='red')
    axes[1, 1].set_title("test action accuracy")

    # Add overall figure title (optional)
    fig.suptitle(f"DOF trainig chack {epochs}", fontsize=16)

    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # leave space for suptitle
    plt.savefig(os.path.join(path, f"accuracy_check{epochs}.png"))

def load_model(dir, epoch, nn_type):
    """
    Loads model parameters and optimizer state.
    """
    # load autoencoder dictionary .json
    with open(os.path.join(dir, f'checkpoint_{epoch}.json'), 'r') as json_file:
        model_dict = json.loads(json_file.read())

    model = nn_type(model_dict, training_phase=False)
    _, params_static = eqx.partition(model, eqx.is_array)

    # load autoencoder parameters (optimized weights and bias) .json
    model_params = eqx.tree_deserialise_leaves(os.path.join(dir, f"checkpoint_{epoch}_autoencoder.eqx"), model)

    return model


def main():

    pretrained = False
    num_itr = 200
    if not pretrained:
        # train a nn to learn active DOFs
        train_dof_nn(save_every=10,
                     epochs=num_itr,
                     num_train_samples_per_mask=800,
                     num_test_samples_per_mast=200,
                     batchsize=100)

    #  load nn from .eqx and test some FOM rigid bodies
    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_system_args(parser)
    config.add_learning_args(parser)  # Parse arguments
    config.add_case_specific_arguments(parser)
    args = parser.parse_args()

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)

    actor, nn_dict = read_snapshots(args)

    epochs = 750
    path = "../output/dof_training/"
    model = load_model(path, epochs, nn_type=learn_link)

    for bid in range(system.n_bodies):
        # print("actor body:" + str(bid), "\n vals range", np.min(abs(actor.bodies[bid].dof), axis=0),
        #       "\n", np.mean(abs(actor.bodies[bid].dof), axis=0),
        #       "\n", np.amax(abs(actor.bodies[bid].dof), axis=0))
        p_dof, p_act = model(actor.bodies[bid].fullT)

        probs = np.sum(abs(p_act), axis=0).astype(np.int32)

        print("\n step", probs)


if __name__ == '__main__':
    main()