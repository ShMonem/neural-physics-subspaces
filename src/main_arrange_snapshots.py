import numpy as np
import config_geomSubspace as config
import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

import jax.numpy as jnp
import pickle

import torch
from torch.utils.data import Dataset, DataLoader


class characterDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.linear = None
        self.pos = None
        # self.angular = None
        self.omega = None
        self.rot = None
        self.tranz = None
        self.fullT = None
        self.len = -1


    def dataFill(self, args, numSnaps, file_patern_p1="snap_", file_patern_ext=".npz"):
        snap_posV = []
        snap_pos = []
        snap_omega = []
        snap_rot = []
        snap_tranz = []
        snap_fullT = []
        self.len = numSnaps
        count = 0
        for s in range(1, numSnaps+1):

                data = np.load(os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                                            file_patern_p1 + f"{s:05d}" + file_patern_ext))
                snap_posV.append(data['vel'])
                snap_pos.append(data['pos'])
                snap_omega.append(data['omega_mat'])
                snap_rot.append(data['rot'])
                snap_tranz.append(data['tranz'])
                snap_fullT.append(data['full_transform'])

                count += 1

        # K number of snapshots to read
        # n number of bodies in the charecter
        # V num of verts in character
        self.linear = jnp.array(snap_posV).reshape(count, -1)  # (K, 3V)  
        self.pos = jnp.array(snap_pos).reshape(count, -1)  # (K, 3V)
        self.rot = jnp.array(snap_rot).reshape(count, -1)  # (K, 9n)
        self.omega = jnp.array(snap_omega).reshape(count, -1)  # (K, 9n)
        self.tranz = jnp.array(snap_tranz).reshape(count, -1)  # (K, 3n)
        self.fullT = jnp.array(snap_fullT).reshape(count, -1)  # (K, 12n)
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        linear = np.array(self.linear[idx], dtype=np.float32)  # 3V
        pos = np.array(self.pos[idx], dtype=np.float32)        # 3V
        omega = np.array(self.omega[idx], dtype=np.float32)    # 9
        rot = np.array( self.rot[idx], dtype=np.float32)        # 9
        tranz = np.array(self.tranz[idx], dtype=np.float32)    # 3
        fullT = np.array(self.fullT[idx], dtype=np.float32)    # 12
        return linear, pos, omega, rot, tranz, fullT

def read_snapshots(args, store_npy=False):

    number_snapshots = args.numSnapshots
    case_extension= "numsnapshots_" + f"{number_snapshots:05d}"

    character = characterDataset()
    character.dataFill(args, number_snapshots)
    # Construct the learned subspace operator dictionary
    nn_dict = {}
    nn_dict['model_type'] = args.model_type
    nn_dict['activation'] = args.activation
    nn_dict['MLP_hidden_layers'] = args.MLP_hidden_layers  # 3
    nn_dict['MLP_hidden_layer_width'] = args.MLP_hidden_layer_width
    nn_dict['vel_dim'] = character.linear.shape[1]  # 3V
    nn_dict['pos_dim'] = character.pos.shape[1]  # 3V
    nn_dict['rot_dim'] = character.rot.shape[1]  # 9
    nn_dict['omega_dim'] = character.omega.shape[1]  # 9
    nn_dict['tranz_dim'] = character.tranz.shape[1]  # 9
    nn_dict['fullT_dim'] = character.fullT.shape[1]  # 12
    nn_dict['rot_latent_dim'] = args.rot_subspace_dim
    nn_dict['tranz_latent_dim'] = args.tranz_subspace_dim

    if store_npy:
        # Store them as .npy matrices to be called later
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_velos.npy"), character.linear)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_pos.npy"), character.pos)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_rot.npy"), character.rot)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_omega.npy"), character.omega)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_tranz.npy"), character.tranz)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_fullT.npy"), character.fullT)

        with open(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "_nn_dict"), 'wb') as f:
            pickle.dump(nn_dict, f)
    return character, nn_dict

def load_snapshots(args, shift, jump):
    number_snapshots = args.numSnapshots
    case_extension= "numsnapshots_" + f"{number_snapshots:05d}"+ "_jump_"+f"{jump:05d}" + "_shift_" + f"{shift:05d}"
    character = characterDataset()
    character.linear = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_velos.npy"))
    character.pos = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_pos.npy"))
    character.rot = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_rot.npy"))
    character.omega = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_omega.npy"))
    character.tranz = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_tranz.npy"))
    character.fullT = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_fullT.npy"))

    return character

