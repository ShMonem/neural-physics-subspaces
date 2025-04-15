import numpy as np
import autoencoder_config as config
import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

import jax.numpy as jnp
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from jax import random, vmap, lax

class body(Dataset):
    """
    Data attributes for one body only
    """
    def __init__(self):
        super().__init__()
        self.linear = None
        self.pos = None
        # self.angular = None
        self.omega = None
        self.rot = None
        self.tranz = None
        self.fullT = None
        self.dof = None
        self.len = -1

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        linear = np.array(self.linear[idx], dtype=np.float32)  # 3V
        pos = np.array(self.pos[idx], dtype=np.float32)        # 3V
        omega = np.array(self.omega[idx], dtype=np.float32)    # 9
        rot = np.array( self.rot[idx], dtype=np.float32)        # 9
        tranz = np.array(self.tranz[idx], dtype=np.float32)    # 3
        fullT = np.array(self.fullT[idx], dtype=np.float32)    # 12
        dof = np.array(self.dof[idx], dtype=np.float32)  # 6
        return linear, pos, omega, rot, tranz, fullT, dof
    
    def one_body_dataFill(self, args, numSnaps, file_patern_p1="snap_", file_patern_ext=".npz"):
        snap_posV = []
        snap_pos = []
        snap_omega = []
        snap_rot = []
        snap_tranz = []
        snap_fullT = []
        snap_dof = []
        self.len = numSnaps
        count = 0
        for s in range(1, numSnaps+1):
            data = np.load(os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                                        file_patern_p1 + f"{s:05d}" + file_patern_ext))
            snap_posV.append(data['vel'])  # (V, 3)
            snap_pos.append(data['pos'])   # (V, 3)
            snap_omega.append(data['omega_mat'])   # (3, 3)
            snap_rot.append(data['rot'])           # (3, 3)
            snap_tranz.append(data['tranz'])       # (3,)
            snap_fullT.append(data['full_transform'])  # (4, 3)
            snap_dof.append(transformation_to_dof(data['full_transform']))  # (6,)
            count += 1

        # K number of snapshots to read
        # n number of bodies in the charecter
        # V num of verts in link
        self.linear = jnp.array(snap_posV).reshape(count, -1)  # (K, 3V)  
        self.pos = jnp.array(snap_pos).reshape(count, -1)  # (K, 3V)
        self.rot = jnp.array(snap_rot).reshape(count, -1)  # (K, 9n)
        self.omega = jnp.array(snap_omega).reshape(count, -1)  # (K, 9n)
        self.tranz = jnp.array(snap_tranz).reshape(count, -1)  # (K, 3n)
        self.fullT = jnp.array(snap_fullT).reshape(count, -1)  # (K, 12n)
        self.dof = jnp.array(snap_dof).reshape(count, -1)  # (K, 6)   # TODO: this works yet only on one body
    
class character:
    """
        Data attributes for complete character with multi-body struct
        """

    def __init__(self):
        super().__init__()
        self.bodies = []
        self.num_bodies = 0


def read_snapshots(args, store_npy=False):
    # Build the system object
    system, system_def = config.construct_system_from_name(args.system_name, args.problem_name)
    case_extension= "numsnapshots_" + f"{args.numSnapshots:05d}"

    actor = character()
    for bid in range(system.n_bodies):

        link = body()
        link.one_body_dataFill(args, args.numSnapshots, file_patern_ext="_body_"+str(bid)+".npz")
        # Construct the learned subspace operator dictionary
        actor.bodies.append(link)
        actor.num_bodies +=1

    nn_dict = {}
    nn_dict['model_type'] = args.model_type
    nn_dict['activation'] = args.activation
    nn_dict['MLP_hidden_layers'] = args.MLP_hidden_layers  # 3
    nn_dict['MLP_hidden_layer_width'] = args.MLP_hidden_layer_width
    nn_dict['vel_dim'] = link.linear.shape[1]  # 3V
    nn_dict['pos_dim'] = link.pos.shape[1]  # 3V
    nn_dict['rot_dim'] = link.rot.shape[1]  # 9
    nn_dict['omega_dim'] = link.omega.shape[1]  # 9
    nn_dict['tranz_dim'] = link.tranz.shape[1]  # 9
    nn_dict['fullT_dim'] = link.fullT.shape[1]  # 12
    nn_dict['rot_latent_dim'] = args.rot_subspace_dim
    nn_dict['tranz_latent_dim'] = args.tranz_subspace_dim

    # if store_npy:
    #     # Store them as .npy matrices to be called later
    #     np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_velos.npy"), link.linear)
    #     np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_pos.npy"), link.pos)
    #     np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_rot.npy"), link.rot)
    #     np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_omega.npy"), link.omega)
    #     np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_tranz.npy"), link.tranz)
    #     np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_fullT.npy"), link.fullT)
    #
    #     with open(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "_nn_dict"), 'wb') as f:
    #         pickle.dump(nn_dict, f)
    return actor, nn_dict

def load_snapshots(args, shift, jump):
    number_snapshots = args.numSnapshots
    case_extension= "numsnapshots_" + f"{number_snapshots:05d}"+ "_jump_"+f"{jump:05d}" + "_shift_" + f"{shift:05d}"
    link = body()
    link.linear = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_velos.npy"))
    link.pos = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_pos.npy"))
    link.rot = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_rot.npy"))
    link.omega = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_omega.npy"))
    link.tranz = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_tranz.npy"))
    link.fullT = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_fullT.npy"))

    return link

def transformation_to_dof(T, retur_values=True, return_dof=False):
    
    """
    Converts a 4x3 homogeneous transformation matrix to 6D pose (yaw, pitch, roll, x, y, z)
    using ZYX (yaw-pitch-roll) Euler angle convention.

    Args:
        T: jnp.ndarray of shape (4, 3)

    Returns:
        array([yaw, pitch, roll, x, y, z])
    """

    # the function returens either the values or the degrees of freedom
    assert (retur_values or return_dof) and not (retur_values and return_dof)  # TODO: dof part
    assert T.shape[-2:] == (4, 3), "Input must be a (n, 4x3) transformation matrix"
    
    def one_body_transform_to_dof(T):
        # Extract rotation and translation
        R = T[:3, :]
        t = T[3, :]
    
        # Compute pitch
        pitch = -jnp.arcsin(jnp.clip(R[2, 0], -1.0, 1.0))
        cos_pitch = jnp.cos(pitch)
        tol = 1e-6
        # Handle gimbal lock
        def handle_regular(_):
            roll = jnp.arctan2(R[2, 1], R[2, 2])
            yaw = jnp.arctan2(R[1, 0], R[0, 0])
            return roll, pitch, yaw

        def handle_gimbal_lock(_):
            roll = jnp.arctan2(-R[1, 2], R[1, 1])
            yaw = 0.0
            return roll, pitch, yaw

        roll, pitch, yaw = lax.cond(
            jnp.abs(cos_pitch) > tol,
            handle_regular,
            handle_gimbal_lock,
            operand=None
        )
    
        return np.array([yaw, pitch, roll, t[0], t[1], t[2]])

    return one_body_transform_to_dof(T)