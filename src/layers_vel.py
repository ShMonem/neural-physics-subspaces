
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


def model_spec_from_args(args, in_dim, out_dim):
    """
    Build the dictionary which we feed into the more general create_network
    (this dictionary can be saved to recreate the same network later)
    """

    spec_dict = {}

    spec_dict['in_dim'] = in_dim
    spec_dict['out_dim'] = out_dim

    spec_dict['model_type'] = args.model_type

    # Add MLP-specific args
    if spec_dict['model_type'] in ["SubspaceMLP"]:

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
        decoder = latent_rot_decode(spec_dict, rngkey, spec_dict['depth'], spec_dict['hidden_dim'])

    else:
        raise ValueError(f"unrecognized model_type {spec_dict['model_type']}")

    # Always create the model in 32-bit, even if the system is defaulting to 64 bit.
    # Otherwise serialization things fail. Passing 64-bit inputs to the model should give
    # the expected up-conversion at evaluation time.
    model_encoder = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, encoder)
    model_decoder = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x, decoder)
    print(f"\n== Created encoding network ({spec_dict['model_type']}):")
    print(model_encoder)
    print(f"\n== Created decoding network ({spec_dict['model_type']}):")
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

    def __init__(self, dict, rngkey, depth=3, hidden_dim=40):

        self.activation = jax.nn.relu
        self.latent2omega_layers = []
        prev_width = dict['latent_dim']
        # first, from latent to omega layers
        for i_layer in range(depth):
            is_last = (i_layer + 1 == depth)
            # last layer would have output dim, otherwise 'width'
            next_width = dict['omega_dim'] if is_last else hidden_dim

            rngkey, subkey = jax.random.split(rngkey)
            self.latent2omega_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

        # second, latent to translation layers
        prev_width = dict['latent_dim']
        for t_layer in range(depth):
            is_last = (t_layer + 1 == depth)
            # last layer would have output dim, otherwise 'width'
            next_width = dict['tranz_dim'] if is_last else hidden_dim

            rngkey, subkey = jax.random.split(rngkey)
            self.latent2omega_layers.append(
                eqx.nn.Linear(prev_width, next_width, use_bias=True, key=subkey))
            prev_width = next_width

    def __call__(self, x_init):
        omega = None
        # rotation layers
        x = x_init
        for i_layer in range(len(self.latent2omega_layers)//2):
            # is_last = (i_layer + 1 == len(self.latent2omega_layers))

            x = self.latent2omega_layers[i_layer].weight @ x
            if self.latent2omega_layers[i_layer].bias is not None:
                x = x + self.latent2omega_layers[i_layer].bias.reshape(-1, 1)

            x = self.activation(x)
        omega = x
        rotation = exp_omega(omega).T
        # translation layers
        x_t = x_init
        for i_layer in range(len(self.latent2omega_layers)//2, len(self.latent2omega_layers)):
            # is_last = (i_layer + 1 == len(self.latent2omega_layers))
            x_t = self.latent2omega_layers[i_layer].weight @ x_t
            if self.latent2omega_layers[i_layer].bias is not None:
                x_t = x_t + self.latent2omega_layers[i_layer].bias.reshape(-1, 1)

            x_t = self.activation(x_t)
        translation = x_t
        return omega, rotation, translation

def exp_omega(omega_flatten):
    """
       Compute rotation matrices from flattened 3x3 skew-symmetric angular velocity matrices.

       Parameters:
           omega_flat (jax.numpy.ndarray): Flattened angular velocity matrices of shape (9, K).

       Returns:
           jax.numpy.ndarray: Flattened rotation matrices of shape (9, K).
       """

    def compute_single_rotation(omega_flat_column):
        # Reshape to (3, 3) matrix
        omega_matrix = omega_flat_column.reshape(3, 3)

        # Compute rotation matrix using matrix exponential
        rotation_matrix = expm(omega_matrix)

        # Flatten back to (9,)
        return rotation_matrix.flatten()

    # Apply the function to each column using vmap
    rotations = jax.vmap(compute_single_rotation, in_axes=1)(omega_flatten)

    return rotations
