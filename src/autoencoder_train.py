## Standard libraries
import os
import json
import math
import numpy as np
from scipy import spatial

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0

## For statistical data visualization
import seaborn as sns
sns.reset_orig()
sns.set()

## Progress bar shows smart progress meter
from tqdm.auto import tqdm

## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
from jax import random, vmap, pmap
import equinox as eqx
# Seeding for random operations
main_rng = random.PRNGKey(24)

## Flax (NN in JAX)
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

## Optax (Optimizers in JAX)
import optax

## PyTorch Data Loading
import torch
import torch.utils.data as data
# from torch.utils.tensorboard import SummaryWriter  # launch/visualize logs
# import torchvision
# from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader

import autoencoder_layers as layers

print("Device:", jax.devices())

import argparse
from autoencoder_get_snapshots import read_snapshots
import autoencoder_config as config
from utils import ensure_dir_exists

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
def main():
    # Build command line arguments
    parser = argparse.ArgumentParser()

    # Shared arguments
    config.add_system_args(parser)
    config.add_learning_args(parser)
    # config.add_training_args(parser)
    config.add_jax_args(parser)

    #  Arguments specific to this program
    config.add_case_specific_arguments(parser)

    # Parse arguments
    args = parser.parse_args()
    dataset, nn_dict = read_snapshots(args) # TODO: make loading option possible
    print("mean vals for dof", np.mean(np.abs(dataset.dof), axis=0))
    print("Training snapshots have been loaded, network dict was constructed:\n", nn_dict)

    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)

    # Force jax to initialize itself so errors get thrown early
    _ = jnp.zeros(())
    # split the data into train-validation-test

    batchSize = 500

    g = torch.Generator().manual_seed(1)
    train_set, val_set, test_set = data.random_split(dataset, [11111, 5555, 5556], generator=g)

    # loaders for each data set
    g1 = torch.Generator().manual_seed(24)
    train_dataloader = DataLoader(train_set, batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True, generator=g1)
    g2 = torch.Generator().manual_seed(15)
    val_dataloader = DataLoader(val_set, batch_size=batchSize, num_workers=4, shuffle=False, generator=g2)
    g3 = torch.Generator().manual_seed(77)
    test_dataloader = DataLoader(test_set, batch_size=batchSize, num_workers=4, shuffle=True, generator=g3)

    ## 1. Test encoder implementation
    # Random key for initialization
    # rng = random.PRNGKey(0)
    # # Example full transformations as input
    # transfors = next(iter(train_dataloader))[5]  # [5] is the index of transformations in the data set
    # transfors = jax.device_put(jnp.array(transfors.numpy()))
    # encoder = layers.transf2Latent_Encoder(nn_dict, rng)
    # # model1 is the NN learning angular velocity from positional velocity, and finds latent space
    # encoder_params, encoder_static = eqx.partition(encoder, eqx.is_array)
    # # print(jax.tree_map(lambda x: x.shape, encoder_params))
    #
    # # Test decoder on one image and on a batch
    # out = encoder(transfors[0])
    # print(transfors[0].shape)
    # print(out.shape)
    # batched_encoder = vmap(encoder, in_axes=0)
    # out2 = batched_encoder(transfors)
    # print(transfors.shape)
    # print(out2.shape)

    ## 2, Test decoder implementation
    # Example latents as input
    # rng, lat_rng = random.split(rng)
    # latents = random.normal(lat_rng, (500, nn_dict['rot_latent_dim']+nn_dict['tranz_latent_dim']))
    # decoder = layers.latent2Transf_Decoder(nn_dict, rng)
    # # model1 is the NN learning angular velocity from positional velocity, and finds latent space
    # decoder_params, decoder_static = eqx.partition(decoder, eqx.is_array)
    # # print(jax.tree_map(lambda x: x.shape, decoder_params))
    # # Test decoder on one image and on a batch
    # out = decoder(latents[0])
    # print(latents[0].shape)
    # print(out.shape)
    # batched_decoder = vmap(decoder, in_axes=0)
    # out2 = batched_decoder(latents)
    # print(latents.shape)
    # print(out2.shape)
    #
    # rng, auto_rng = random.split(rng)
    # autoencoder = layers.Autoencoder(nn_dict, auto_rng)
    # autoeecoder_params, autoeecoder_static = eqx.partition(autoencoder, eqx.is_array)
    # # print(jax.tree_map(lambda x: x.shape, autoeecoder_params))
    # # Example full transformations as input
    # transfors = next(iter(train_dataloader))[5]  # [5] is the index of transformations in the data set
    # transfors = jax.device_put(jnp.array(transfors.numpy()))
    # out = autoencoder(transfors[0])
    # print(transfors[0].shape)
    # print(out.shape)
    #
    # batched_autoencoder = vmap(autoencoder, in_axes=0)
    # out2 = batched_autoencoder(transfors)
    # print(transfors.shape)
    # print(out2.shape)

    # Random key for initialization
    rng = random.PRNGKey(0)
    # Train the autoencoder
    model_train_dict = {}
    epochs = 500

    loader_input_index = 5  # for full transformations
    network_filename_dir = os.path.join(args.output_dir, args.problem_name, args.output_nn_dir)
    rot_reduction_loss = []

    start, end = 1 , 9
    args.output_prefix = 'Omerrnrm_p_Rerrnrm_p_terrnrm_'
    for rot_dim in range(start, end + 1):
        # modify rot latent dim in both args and dict
        args.rot_subspace_dim, nn_dict['rot_latent_dim'] = rot_dim, rot_dim
        # Build an informative output name
        network_filename_base = f"{args.output_prefix}_{args.activation}_epochs_{epochs}" \
                                f"_rot_latent_dim_{args.rot_subspace_dim}_tranz_latent_dim_{args.tranz_subspace_dim}"
        network_filename_pre = os.path.join(network_filename_dir, network_filename_base)

        ensure_dir_exists(network_filename_pre)
        print(f"Saving result to {network_filename_pre}")

        test_loss = layers.evaluate_autoencoder(system, system_def, args, nn_dict, rng,
                                                    train_dataloader, val_dataloader, test_dataloader,
                                                    epochs, network_filename_pre, int(loader_input_index),
                                                    pretrained=False)

        rot_reduction_loss.append(test_loss)

    plt.figure(figsize=(6, 4))
    plt.plot(range(start, end  + 1), rot_reduction_loss,
             '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    plt.title("Reconstruction error over rot latent dimensionality", fontsize=14)
    plt.minorticks_off()
    plt.xlabel('rot latent dim')
    plt.ylabel('Energy')
    plt.yscale('log')

    plt.savefig(os.path.join(network_filename_dir, args.output_prefix + "training_tranz_latent_3_diff_rot_latent_dim.png"))
    plt.show()


if __name__ == '__main__':
    main()