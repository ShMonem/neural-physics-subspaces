
import jax
import jax.numpy as jnp
import equinox as eqx
from jax.scipy.linalg import expm
import typing


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

    if domain_name == 'V2latent2T':

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

        encoder = Pos_omega_latent_encode(spec_dict, rngkey)
        decoder = latent_rot_decode(spec_dict, rngkey)

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

# Define the feedforward neural network
class Pos_omega_latent_encode(eqx.Module):
    pos2omega_layers: typing.List[eqx.nn.Linear]
    omega2latent_layers: typing.List[eqx.nn.Linear]
    activation: typing.Callable

    def __init__(self, dict, rngkey):
        # MLP layers
        self.activation = str_to_act(dict['activation'])

        self.pos2omega_layers = []
        self.omega2latent_layers = []
        prev_width = dict['pos_dim']  # first layer has position dim
        # hidden_layer_with = dict['MLP_hidden_layer_width']

        # building the layers
        # First from positions to angular velocity mat
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have output dim, otherwise 'width'
            next_width = dict['omega_dim'] if is_last else dict['MLP_hidden_layer_width']

            rngkey, subkey = jax.random.split(rngkey)
            self.pos2omega_layers.append(eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

        # second from angular velocity to laten space
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            next_width = dict['latent_dim'] if is_last else dict['MLP_hidden_layer_width']
            rngkey, subkey = jax.random.split(rngkey)
            self.omega2latent_layers.append(eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width
        k =0
    def __call__(self, x):
        # MLP layes
        omega, latent = None, None
        for i_layer in range(len(self.pos2omega_layers)):
            # is_last = (i_layer + 1 == len(self.pos2omega_layers))
            x = self.pos2omega_layers[i_layer].weight @ x
            if self.pos2omega_layers[i_layer].bias is not None:
                x = x + self.pos2omega_layers[i_layer].bias.reshape(-1, 1)

            x = self.activation(x)
        omega = x

        for i_layer in range(len(self.omega2latent_layers)):
            # is_last = (i_layer + 1 == len(self.omega2latent_layers))
            x = self.omega2latent_layers[i_layer].weight @ x
            if self.omega2latent_layers[i_layer].bias is not None:
                x = x + self.omega2latent_layers[i_layer].bias.reshape(-1, 1)

            x = self.activation(x)
        latent = x
        return omega, latent


class latent_rot_decode(eqx.Module):
    activation: callable
    latent2omega_layers: typing.List[eqx.nn.Linear]
    latent2tranz_layers: typing.List[eqx.nn.Linear]
    def __init__(self, dict, rngkey):

        self.activation = jax.nn.relu
        self.latent2omega_layers = []
        self.latent2tranz_layers = []
        prev_width = dict['latent_dim']
        # first, from latent to omega layers
        for i_layer in range(dict['MLP_hidden_layers']):
            is_last = (i_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have output dim, otherwise 'width'
            next_width = dict['omega_dim'] if is_last else dict['MLP_hidden_layer_width']

            rngkey, subkey = jax.random.split(rngkey)
            self.latent2omega_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=False, key=subkey))
            prev_width = next_width

        # second, latent to translation layers
        prev_width = dict['latent_dim']
        for t_layer in range(dict['MLP_hidden_layers']):
            is_last = (t_layer + 1 == dict['MLP_hidden_layers'])
            # last layer would have output dim, otherwise 'width'
            next_width = dict['tranz_dim'] if is_last else dict['MLP_hidden_layer_width']

            rngkey, subkey = jax.random.split(rngkey)
            self.latent2tranz_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=False, key=subkey))
            prev_width = next_width

    def __call__(self, x_init):
        omega = None
        # rotation layers
        x = x_init
        for i_layer in range(len(self.latent2omega_layers)):
            is_last = (i_layer + 1 == len(self.latent2omega_layers))

            x = self.latent2omega_layers[i_layer].weight @ x
            if self.latent2omega_layers[i_layer].bias is not None:
                x = x + self.latent2omega_layers[i_layer].bias.reshape(-1, 1)

            if not is_last:
                x = self.activation(x)
        if self.latent2omega_layers[-1].bias is not None:
            omega = x + self.latent2omega_layers[-1].bias.reshape(-1, 1)
        else:
            omega = x
        # one exponential layer
        rotations = exp_omega(omega).T

        # translation layers
        x_t = x_init
        for i_layer in range(len(self.latent2tranz_layers)):
            is_last = (i_layer + 1 == len(self.latent2tranz_layers))
            x_t = self.latent2tranz_layers[i_layer].weight @ x_t
            if self.latent2tranz_layers[i_layer].bias is not None:
                x_t = x_t + self.latent2tranz_layers[i_layer].bias.reshape(-1, 1)
            if not is_last:
                x_t = self.activation(x_t)
        if self.latent2tranz_layers[-1].bias is not None:
            translations = x_t + self.latent2tranz_layers[-1].bias.reshape(-1, 1)
        else:
            translations = x_t

        # last layer combines between rot and translation into complete stacked linear transformations
        transf = form_transformations(rotations, translations)
        return omega, transf, rotations, translations

def exp_omega(omega_flatten):
    def compute_rotation_for_one_character(omega_flat_column):

        # Reshape into (n, 3, 3) structure
        n = omega_flat_column.shape[0] // 9
        omega_matrices = omega_flat_column.reshape(n, 3, 3)

        # Apply expm to each matrix
        expm_matrices = jax.vmap(expm)(omega_matrices)

        # Flatten back to (9,)
        return expm_matrices.reshape(-1)

    # Apply the function to each column using vmap
    if omega_flatten.ndim == 1:
        return compute_rotation_for_one_character(omega_flatten)
    elif omega_flatten.ndim == 2:
        return jax.vmap(compute_rotation_for_one_character, in_axes=1)(omega_flatten)
    else:
        raise ValueError("Input must be of shape (9n,) or (9n, K)")


def form_transformations(rotations, translations):
    """
    Form stacked flatten transformations (12n) from stacked flatten rotations (9n)
    and stacked translations (3n).

    Parameters:
        rotations: jax.numpy.ndarray
            - Shape (9n,) or (9n, K), where each 9 entries represent a flattened 3x3 rotation matrix.
        translations: jax.numpy.ndarray
            - Shape (3n,) or (3n, K), where each 3 entries represent a translation vector.

    Returns:
        jax.numpy.ndarray:
            - Shape (12n,) or (12n, K), where each 12 entries represent a flattened 4x3 transformation matrix.
    """
    if rotations.ndim == 1 and translations.ndim == 1:  # Case: Single transformation (9n) and (3n)
        n = rotations.shape[0] // 9
        # Reshape to (n, 3, 3) for rotations and (n, 3) for translations
        rotations_reshaped = rotations.reshape(n, 3, 3)
        translations_reshaped = translations.reshape(n, 3, 1)

        # Concatenate rotations and translations into (n, 4, 3)
        transformations = jnp.concatenate([rotations_reshaped, translations_reshaped], axis=2)

        # Flatten back to (12n,)
        return transformations.reshape(-1)

    elif rotations.ndim == 2 and translations.ndim == 2:  # Case: Snapshots (9n, K) and (3n, K)
        n = rotations.shape[0] // 9
        k = rotations.shape[1]

        # Reshape to (n, 3, 3, K) for rotations and (n, 3, 1, K) for translations
        rotations_reshaped = rotations.reshape(n, 3, 3, k)
        translations_reshaped = translations.reshape(n, 1, 3, k)

        # Concatenate rotations and translations into (n, 4, 3, K)
        transformations = jnp.concatenate([rotations_reshaped, translations_reshaped], axis=1)

        # Flatten back to (12n, K)
        return transformations.reshape(-1, k)

    else:
        raise ValueError("Input shapes must be (9n,) and (3n,) or (9n, K) and (3n, K).")
