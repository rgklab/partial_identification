from collections import OrderedDict
import numpy as np

####################
# Collection of directed acyclic graphs for testing the quality
####################


class DAG:
    r"""Describes the DAG structure, the variables in the DAG, and their dimensions"""
    def __init__(self, name: str, graph: OrderedDict, latent_dim: int,  n_latent: int,
                 do_var: int, target_var: int, var_dims: np.ndarray, binary_keys: list):
        self.name = name
        self.graph = graph
        self.var_dims = var_dims
        self.latent_dim = latent_dim
        self.n_latent = n_latent
        self.do_var = do_var
        self.target_var = target_var

        self.binary_keys = binary_keys  # list of binary variables

    def __str__(self):
        return self.name


# T ----> Y
def gen_2d(var_dims: np.ndarray, binary_keys: list):
    name = '2d'
    graph = OrderedDict([
        (0, []),  # T
        (1, [0, -1]),  # Y
    ])
    do_var = 0
    target_var = 1
    n_latent = 1  # Total number of latent variables
    latent_dim = 1  # The dimension of latent noise variables
    return DAG(name, graph, latent_dim, n_latent, do_var, target_var, var_dims, binary_keys)


# X -> T -> Y. Identifiable
def gen_backdoor(var_dims: np.ndarray, binary_keys: list):
    name = 'backdoor'
    graph = OrderedDict([
        (0, []),  # X
        (1, [0, -1]),  # T
        (2, [0, 1, -2]),  # Y
    ])
    do_var = 1
    target_var = 2
    n_latent = 2  # Total number of latent variables
    latent_dim = 1  # The dimension of latent noise variables
    return DAG(name, graph, latent_dim, n_latent, do_var, target_var, var_dims, binary_keys)


#      - - - -
#    /        \
#   T -> M -> Y     Identifiable
def gen_frontdoor(var_dims: np.ndarray, binary_keys: list):
    name = 'frontdoor'
    graph = OrderedDict([
        (0, [-1, -2]),
        (1, [0, -3]),
        (2, [1, -1, -4]),
    ])
    do_var = 0
    target_var = 2
    n_latent = 4
    latent_dim = 1
    return DAG(name, graph, latent_dim, n_latent, do_var, target_var, var_dims, binary_keys)


#           --
#         /   \
#   X -> T -> Y
def gen_iv(var_dims: np.ndarray, binary_keys: list):
    name = 'iv'
    graph = OrderedDict([
        (0, []),
        (1, [0, -1, -2]),
        (2, [1, -2, -3]),
    ])
    do_var = 1
    target_var = 2
    n_latent = 3
    latent_dim = 1
    return DAG(name, graph, latent_dim, n_latent, do_var, target_var, var_dims, binary_keys)


#       -------
#      /    -  \
#    /     / \ \
#   T -> X ->  Y    Non-identifiable
def gen_leaky(var_dims: np.ndarray, binary_keys: list):
    name = 'leaky'
    graph = OrderedDict([
        (0, [-1, -2]),
        (1, [0, -3, -4]),
        (2, [1, -1, -3, -5])
    ])

    do_var = 0
    target_var = 2
    n_latent = 5
    latent_dim = 1
    return DAG(name, graph, latent_dim, n_latent, do_var, target_var, var_dims, binary_keys)


# Non-identifiable
def gen_bow(var_dims: np.ndarray, binary_keys: list):
    name = 'bow'
    graph = OrderedDict([
        (0, [-1, -3]),
        (1, [0, -1, -2]),
    ])
    do_var = 0
    target_var = 1
    n_latent = 3
    latent_dim = 1
    return DAG(name, graph, latent_dim, n_latent, do_var, target_var, var_dims, binary_keys)


def gen_dags(key: str, var_dims: np.ndarray, binary_keys: list):
    return {
        'backdoor': gen_backdoor,
        'frontdoor': gen_frontdoor,
        'bow': gen_bow,
        'leaky': gen_leaky,
        'iv': gen_iv,
        '2d': gen_2d
    }[key](var_dims, binary_keys)
