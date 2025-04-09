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
    From Lie theory, for any rotation matrix R∈SO(3)R∈SO(3), we have:
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

    # Translation as the last row
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
        dist_bounds = (-5.0, 5.0)

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
            else:
                action = jax.nn.sigmoid(action)
        return action


class learn_link(eqx.Module):
    dof_branch: learn_dof
    action_branch: learn_action

    def __init__(self, dict, rngkey=None):

        if rngkey is None:
            rngkey = jax.random.PRNGKey(0)

        dof_branch = learn_dof(dict, rngkey)
        rngkey, _ = random.split(rngkey)
        action_branch = learn_action(dict, rngkey)

        self.dof_branch = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, dof_branch)
        self.action_branch = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, action_branch)

    def __call__(self, y, return_details=False):

        pred_q = self.dof_branch(y)
        pred_action = self.action_branch(pred_q)
        return pred_q, pred_action


def loss_fn(params, model, target_q, target_dof, target_act):
    # param: the freshly adjusted trainable parameters
    # re-build/update model from trainable and static parameters
    model = eqx.combine(params, eqx.filter(model, lambda m: not eqx.is_array(m)))
    model_batched = vmap(model, in_axes=0)
    pred_dof, pred_act = model_batched(target_q)  # (batch_size, dof_dim), (batch_size, act_dim)
    loss_dof = jnp.linalg.norm(pred_dof - target_dof)    # how close to dof vals
    loss_act = optax.sigmoid_binary_cross_entropy(pred_act, target_act).mean()    # which dofs are active
    return loss_dof + loss_act


def main():
    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())
    # All possible dof activations
    action_lables = generate_all_activation_combinations()  # (64, 6)
    print(action_lables.shape)
    # dof values sampler
    sampler = dof_sampler()
    sampler.set_num_samples_per_mast(10)
    sampler.set_dist_tol(1e-1)
    sampler.set_angle_tol(1e-2)

    dof_samples, act_samples = vmap(sampler.sample_pose_from_activation, in_axes=0)(action_lables)  # (num_samples*64, 6)
    dof_samples, act_samples = dof_samples.reshape(-1, 6), act_samples.reshape(-1, 6)
    print(dof_samples.shape)
    q_samples = vmap(dof_to_transformation, in_axes=0)(dof_samples)
    print(q_samples.shape)

    dict = {}
    dict['MLP_hidden_layers'] = 3
    dict['MLP_hidden_layer_width'] = 20

    # adam returns init_fun, update_fun, get_parameters
    optimizer_init, optimizer_update, get_params = optimizers.adam(1e-3)
    # opt_state tracks model optimization
    model = learn_link(dict)
    params, _ = eqx.partition(model, eqx.is_array)
    opt_state = optimizer_init(eqx.filter(model, eqx.is_array))

    # Training step
    @eqx.filter_jit
    def train_step(step, model, opt_state, q, dof, act):
        param = get_params(opt_state)
        loss, grads = jax.value_and_grad(loss_fn)(param, model, q, dof, act)
        opt_state = optimizer_update(step, grads, opt_state)
        # re-build/update model from trainable and static parameters
        model = eqx.combine(params, eqx.filter(model, lambda m: not eqx.is_array(m)))
        return model, opt_state, loss

    step = 0
    for epoch in range(500):
        model, opt_state, loss = train_step(step, model, opt_state, q_samples, dof_samples, act_samples)
        print(step, loss)
        step += 1

    activation = jnp.array([1, 0, 1, 1, 0, 0])  # Roll, Yaw, X are active

    dof_batch, act_batch = sampler.sample_pose_from_activation(activation)
    q_bach = vmap(dof_to_transformation, in_axes=0)(dof_batch)
    p_dof, p_act = vmap(model, in_axes=0)(q_bach)

    print((p_act.sum(axis=0)/p_act.shape[0] > 0.5).astype(jnp.int32))
    print(p_act.sum(axis=0)/p_act.shape[0])


if __name__ == '__main__':
    main()