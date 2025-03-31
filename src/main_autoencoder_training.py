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
from jax import random, vmap
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


import layers_geomSubspace as layers

print("Device:", jax.devices()[0])

import argparse
from main_arrange_snapshots import read_snapshots
import config_geomSubspace as config
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
    parser.add_argument("--output_dir", type=str, default="../output")
    parser.add_argument("--output_nn_dir", type=str, default="pretrained_models")
    parser.add_argument("--snapshots_input_dir", type=str, default="snapshots")

    parser.add_argument("--output_prefix", type=str, default="")

    # network defaults
    parser.add_argument("--subspace_domain_type", type=str, default='normal')
    parser.add_argument("--model_type", type=str, default='learnGeometricalAwareSolver')
    parser.add_argument("--activation", type=str, default='ReLU')
    parser.add_argument("--rot_subspace_dim", type=int, default=9)
    parser.add_argument("--tranz_subspace_dim", type=int, default=3)
    parser.add_argument("--numSnapshots", type=int, default=9999)

    # Parse arguments
    args = parser.parse_args()
    train_dataset, nn_dict = read_snapshots(args) # TODO: make loading option possible
    print("Training snapshots have been loaded, network dict was constructed:\n", nn_dict)

    # split the data into train-validation-test
    train_batch = 5000
    val_batch = 1999
    test_batch = 3000
    batchSize = 500

    train_set, val_set, test_set = data.random_split(train_dataset, [5000, 1999, 3000], generator=torch.Generator().manual_seed(0))


    # loaders for each data set
    train_dataloader = DataLoader(train_set, batch_size=batchSize, num_workers=4, pin_memory=True, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batchSize, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batchSize, num_workers=4, shuffle=True)

    ## 1. Test encoder implementation
    # Random key for initialization
    rng = random.PRNGKey(0)
    # Example full transformations as input
    transfors = next(iter(train_dataloader))[5]  # [5] is the index of transformations in the data set
    transfors = jax.device_put(jnp.array(transfors.numpy()))
    encoder = layers.transf2Latent_Encoder(nn_dict, rng)
    # model1 is the NN learning angular velocity from positional velocity, and finds latent space
    encoder_params, encoder_static = eqx.partition(encoder, eqx.is_array)
    # print(jax.tree_map(lambda x: x.shape, encoder_params))

    # Test decoder on one image and on a batch
    out = encoder(transfors[0])
    print(transfors[0].shape)
    print(out.shape)
    batched_encoder = vmap(encoder, in_axes=0)
    out2 = batched_encoder(transfors)
    print(transfors.shape)
    print(out2.shape)

    ## 2, Test decoder implementation
    # Example latents as input
    rng, lat_rng = random.split(rng)
    latents = random.normal(lat_rng, (500, nn_dict['rot_latent_dim']+nn_dict['tranz_latent_dim']))
    decoder = layers.latent2Transf_Decoder(nn_dict, rng)
    # model1 is the NN learning angular velocity from positional velocity, and finds latent space
    decoder_params, decoder_static = eqx.partition(decoder, eqx.is_array)
    # print(jax.tree_map(lambda x: x.shape, decoder_params))
    # Test decoder on one image and on a batch
    out = decoder(latents[0])
    print(latents[0].shape)
    print(out.shape)
    batched_decoder = vmap(decoder, in_axes=0)
    out2 = batched_decoder(latents)
    print(latents.shape)
    print(out2.shape)

    rng, auto_rng = random.split(rng)
    autoencoder = layers.Autoencoder(nn_dict, auto_rng)
    autoeecoder_params, autoeecoder_static = eqx.partition(autoencoder, eqx.is_array)
    # print(jax.tree_map(lambda x: x.shape, autoeecoder_params))
    # Example full transformations as input
    transfors = next(iter(train_dataloader))[5]  # [5] is the index of transformations in the data set
    transfors = jax.device_put(jnp.array(transfors.numpy()))
    out = autoencoder(transfors[0])
    print(transfors[0].shape)
    print(out.shape)

    batched_autoencoder = vmap(autoencoder, in_axes=0)
    out2 = batched_autoencoder(transfors)
    print(transfors.shape)
    print(out2.shape)

    # Train the autoencoder
    model_train_dict = {}
    epochs = 100

    loader_input_index = 5  # for full transformations
    rot_reduction_loss = []
    network_filename_dir = os.path.join(args.output_dir, args.problem_name, args.output_nn_dir)

    # Build an informative output name
    network_filename_base = f"{args.output_prefix}_{args.activation}_epochs_{epochs}" \
                            f"_rot_latent_dim_{args.rot_subspace_dim}_tranz_latent_dim_{args.tranz_subspace_dim}"
    network_filename_pre = os.path.join(network_filename_dir, network_filename_base)

    ensure_dir_exists(network_filename_pre)
    print(f"Saving result to {network_filename_pre}")

    _ , test_loss = layers.evaluate_autoencoder(args, autoencoder, nn_dict, rng,
                                                train_dataloader, val_dataloader, test_dataloader,
                                                epochs, network_filename_pre, int(loader_input_index),
                                                pretrained=False)

    rot_reduction_loss.append(test_loss)

    # plt.figure(figsize=(6, 4))
    # plt.plot(range(1, 9 + 1), rot_reduction_loss,
    #          '--', color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16)
    # plt.title("Reconstruction error over rot latent dimensionality", fontsize=14)
    # plt.minorticks_off()
    # plt.xlabel('rot latent dim')
    # plt.ylabel('Energy')
    # plt.yscale('log')
    # plt.savefig(os.path.join(network_filename_dir, "loss_while_training_tranz_latent_3_diff_rot_latent_dim.png"))
    # plt.show()


if __name__ == '__main__':
    main()