import jax

# imports from this project
from fem_model import FEMSystem
from rigid3d_model import Rigid3DSystem


## Argument parser

def add_system_args(parser):
    # TODO: arguments are fixed for now only for debugging purpose!
    # parser.add_argument("--system_name", type=str, required=True)
    # parser.add_argument("--problem_name", type=str, required=True)
    parser.add_argument("--system_name", type=str, default="rigid3d")   # adjust --
    parser.add_argument("--problem_name", type=str, default="bar")    # adjust --
    parser.add_argument("--timestep_h", type=float, default=0.05)


def add_learning_args(parser):
    parser.add_argument("--MLP_hidden_layers", type=int, default=3)     # adjust --
    parser.add_argument("--MLP_hidden_layer_width", type=int, default=40)    # adjust --


def add_training_args(parser):
    parser.add_argument("--run_name", type=str, default="training")
    parser.add_argument("--n_train_iters", type=int, default=1000000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--report_every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_every", type=int, default=250000)
    parser.add_argument("--lr_decay_frac", type=float, default=.5)


def add_jax_args(parser):
    parser.add_argument("--log_compiles", action='store_true')
    parser.add_argument("--disable_jit", action='store_true')
    parser.add_argument("--debug-nans", action='store_true')
    parser.add_argument("--enable_double_precision", action='store_true')


def process_jax_args(args):

    if args.log_compiles:
        jax.config.update("jax_log_compiles", 1)
    if args.disable_jit:
        jax.config.update('jax_disable_jit', True)
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
    if args.enable_double_precision:
        jax.config.update("jax_enable_x64", True)

system_class_registry = {
        'fem' : FEMSystem,
        'rigid3d' : Rigid3DSystem,
    }


def construct_system_from_name(system_name, problem_name):

    if system_name not in system_class_registry:
        raise ValueError(
        f"""
        System name {system_name} not found in system registry. Either the name is wrong, or the system needs to be registered with the system_class_registry dictionary after the class is defined.

        Current registered systems are {system_class_registry.keys()}.
        """)

    system_class = system_class_registry[system_name]
    system, system_def = system_class.construct(problem_name)

    return system, system_def