import numpy as np
import config_geomSubspace as config
import os

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")

import jax.numpy as jnp
import pickle


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


def read_train_and_test_snapshots(args, shift=25, jump=50,
                                  file_patern_p1="snapshot", file_patern_ext="_.npz", store_npy=True):
    character = body()
    snap_posV = []
    snap_pos = []
    snap_omega = []
    snap_rot = []
    test_snap_posV = []
    test_snap_pos = []
    test_snap_omega = []
    test_snap_rot = []
    snap_tranz = []
    snap_fullT = []
    test_snap_tranz = []
    test_snap_fullT = []

    count = 0
    number_snapshots = args.numSnapshots
    case_extension= "numsnapshots_" + f"{number_snapshots:05d}"+ "_jump_"+f"{jump:05d}" + "_shift_" + f"{shift:05d}"
    for s in range(1, number_snapshots):
        if s % jump == 0:
            data = np.load(os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                                        file_patern_p1 + f"{s:05d}" + file_patern_ext))
            snap_posV.append(data['vel'])
            snap_pos.append(data['pos'])
            snap_omega.append(data['omega_mat'])
            snap_rot.append(data['rot'])
            snap_tranz.append(data['tranz'])
            snap_fullT.append(data['full_transform'])

            test_data = np.load(os.path.join(args.output_dir, args.problem_name, args.snapshots_input_dir,
                                             file_patern_p1 + f"{s + shift:05d}" + file_patern_ext))
            test_snap_posV.append(test_data['vel'])
            test_snap_pos.append(test_data['pos'])
            test_snap_omega.append(test_data['omega_mat'])
            test_snap_rot.append(test_data['rot'])
            test_snap_tranz.append(data['tranz'])
            test_snap_fullT.append(data['full_transform'])
            count += 1

    character.linear = jnp.array(snap_posV).reshape(count, -1).T  # (Sum 3V_i , K)  (i <= n bodies)
    character.pos = jnp.array(snap_pos).reshape(count, -1).T  # (Sum 3V_i , K)
    character.rot = jnp.array(snap_rot).reshape(count, -1).T  # (9n, K)
    character.omega = jnp.array(snap_omega).reshape(count, -1).T  # (9n, K)
    character.tranz = jnp.array(snap_tranz).reshape(count, -1).T  # (3n, K)
    character.fullT = jnp.array(snap_fullT).reshape(count, -1).T  # (12n, K)

    character.linear_test = jnp.array(test_snap_posV).reshape(count, -1).T  # (Sum 3V_i , K)
    character.pos_test = jnp.array(test_snap_pos).reshape(count, -1).T  # (Sum 3V_i , K)
    character.rot_test = jnp.array(test_snap_rot).reshape(count, -1).T
    character.omega_test = jnp.array(test_snap_omega).reshape(count, -1).T
    character.tranz_test = jnp.array(test_snap_tranz).reshape(count, -1).T  # (3n, K)
    character.fullT_test = jnp.array(test_snap_fullT).reshape(count, -1).T  # (9n, K)

    # Construct the learned subspace operator dictionary
    nn_dict = {}
    nn_dict['model_type'] = args.model_type
    nn_dict['activation'] = args.activation
    nn_dict['MLP_hidden_layers'] = args.MLP_hidden_layers  # 3
    nn_dict['MLP_hidden_layer_width'] = args.MLP_hidden_layer_width
    nn_dict['vel_dim'] = character.linear.shape[0]  # 3V
    nn_dict['pos_dim'] = character.pos.shape[0]  # 3V
    nn_dict['rot_dim'] = character.rot.shape[0]  # 9
    nn_dict['omega_dim'] = character.omega.shape[0]  # 9
    nn_dict['tranz_dim'] = character.tranz.shape[0]  # 9
    nn_dict['latent_dim'] = args.subspace_dim

    if store_npy:
        # Store them as .npy matrices to be called later
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_velos.npy"), character.linear)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_pos.npy"), character.pos)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_rot.npy"), character.rot)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_omega.npy"), character.omega)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_linear_tranz.npy"), character.tranz)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "train_fullT.npy"), character.fullT)

        # Store test snapshots as .npy matrices to be called later
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_linear_velos.npy"), character.linear_test)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_pos.npy"), character.pos_test)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_linear_rot.npy"), character.rot_test)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_omega.npy"), character.omega_test)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_linear_tranz.npy"), character.tranz_test)
        np.save(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_fullT.npy"), character.fullT_test)

        with open(os.path.join( args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "_nn_dict"), 'wb') as f:
            pickle.dump(nn_dict, f)
    return character, nn_dict

def read_train_snapshots(args, shift, jump):
    number_snapshots = args.numSnapshots
    case_extension= "numsnapshots_" + f"{number_snapshots:05d}"+ "_jump_"+f"{jump:05d}" + "_shift_" + f"{shift:05d}"
    character = body()
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

def read_test_snapshots(args, shift, jump):
    number_snapshots = args.numSnapshots
    case_extension= "numsnapshots_" + f"{number_snapshots:05d}"+ "_jump_"+f"{jump:05d}" + "_shift_" + f"{shift:05d}"
    character = body()
    character.linear_test = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_linear_velos.npy"))
    character.pos_test = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_pos.npy"))
    character.rot_test = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_linear_rot.npy"))
    character.omega_test = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_omega.npy"))
    character.tranz_test = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_linear_tranz.npy"))
    character.fullT_test = np.load(
        os.path.join(args.output_dir, args.problem_name, args.output_nn_dir, case_extension + "test_fullT.npy"))
