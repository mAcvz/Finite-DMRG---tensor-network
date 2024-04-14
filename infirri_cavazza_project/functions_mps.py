# functions_mps.py>

import tensornetwork as tn
import numpy as np
import sys


# ==================================================================
# DESCRIPTION:
# ------------------------------------------------------------------
# This Python script provides functions for manipulating Matrix Product
# States (MPS) and evaluating physical quantities such as total energy
# It includes functions for creating random normalized MPS, evaluating
# the norm of MPS, splitting MPS tensors, and computing the total energy
# of the MPS under the influence of a given Hamiltonian operator
# ------------------------------------------------------------------
# Functions:
# ------------------------------------------------------------------
# 1. create_random_MPS_normalized(num_sites, physical_dim, bond_dim,
#                                 debug=False, external_leg=False):
#    - Generates a random normalized MPS as an array of tensors
# 2. create_random_MPS_normalized_PBC(num_sites, physical_dim, bond_dim,
#                                     debug=False, external_leg=False):
#    - Generates a random normalized MPS with periodic boundary conditions (PBC)
# 3. create_dual_MPS_tensor(MPS):
#    - Creates the dual of a tensor MPS
# 4. normalize_tensor(tensor):
#    - Normalizes a tensor
# 5. norm_only_MPS(MPS, external_leg=True, debug=False, PBC=False):
#    - Evaluates the norm of an MPS
# 6. split_two_right(tensor, limit, shape_left, shape_right, renormalize=False):
#    - Splits a tensor into two tensors, keeping the right tensor
# 7. split_two_left(tensor, limit, shape_left, shape_right, renormalize=False):
#    - Splits a tensor into two tensors, keeping the left tensor
# 8. total_energy(MPS_tens, MPO, dual_MPS, external_leg=False,
#                 PBC=False, MPO_first=None):
#    - Computes the total energy of the MPS under a given Hamiltonian
# ==================================================================


# =============================================================================
# Important quantities
# =============================================================================

sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])

id = np.eye(2)
oo = np.diag([0, 0])



# ==========================================================================================
# Create MPS (randomly)
# ==========================================================================================

def create_random_MPS_normalized(num_sites, phisical_dim, bond_dim, debug=False, external_leg=False):
    '''
    create_random_MPS_normalized(num_sites, physical_dim, bond_dim, debug=False, external_leg=False):
    DESCRIPTION:
       - Generates a random normalized Matrix Product State (MPS) as an array of tensors
    PROPERTIES:
       - Creates a random MPS with specified parameters
    INPUT PARAMETERS:
       - num_sites (int): Number of sites in the MPS
       - physical_dim (int): Physical dimension of each site
       - bond_dim (int): Bond dimension of the MPS
       - debug (bool): Optional parameter for debugging (default: False)
       - external_leg (bool): Indicates if the MPS has an external leg (default: False)
    OUTPUT:
       - List of numpy.ndarray: Array of tensors representing the MPS
       - Note: It only works for even num_sites
    '''

    nodes = []
    inf_ind, sup_ind = 1, 1

    for ii in range(num_sites):
        
        inf_ind = phisical_dim**(ii)
        sup_ind = phisical_dim**(num_sites - 1 - ii)

        if ii == 0:
            shape = (1, phisical_dim, int(inf_ind*phisical_dim))

        elif ii == num_sites-1:
            shape = (int(sup_ind*phisical_dim), phisical_dim, 1)

        elif inf_ind < bond_dim and ii < num_sites/2:
            new = min(int(inf_ind*phisical_dim), bond_dim)
            shape = (inf_ind, phisical_dim, new)

        elif sup_ind < bond_dim and ii >= num_sites/2:
            new = min(int(sup_ind*phisical_dim), bond_dim)
            shape = (new, phisical_dim, sup_ind)

        else:
            shape = (bond_dim, phisical_dim, bond_dim)

        tensor = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        if debug == True:
            print("added tensor of dimensions (", *shape, ")")

        # To prevent overflow
        nodes.append(normalize_tensor(tensor))

    if external_leg == False:
        nodes[0] = np.tensordot(nodes[0], np.array([1]), axes = ([0], [0]))
        nodes[-1] = np.tensordot(nodes[-1], np.array([1]), axes = ([2], [0]))

    norma = norm_only_MPS(nodes, external_leg=external_leg, PBC=False)
    new_nodes = [nodes[ii]*np.power(norma, -1./num_sites) for ii in range(len(nodes))]

    return new_nodes


def create_random_MPS_normalized_PBC(num_sites, phisical_dim, bond_dim, debug=False, external_leg=False):

    '''
    2. create_random_MPS_normalized_PBC(num_sites, physical_dim, bond_dim, debug=False, external_leg=False):
    DESCRIPTION:
       - Generates a random normalized Matrix Product State (MPS) with periodic boundary conditions (PBC)
    PROPERTIES:
       - Creates a random MPS with PBC and specified parameters
    INPUT PARAMETERS:
       - num_sites (int): Number of sites in the MPS
       - physical_dim (int): Physical dimension of each site
       - bond_dim (int): Bond dimension of the MPS
       - debug (bool): Optional parameter for debugging (default: False)
       - external_leg (bool): Indicates if the MPS has an external leg (default: False)
    OUTPUT:
       - List of numpy.ndarray: Array of tensors representing the MPS
       - Note: It only works for even num_sites
       - Note: PBC require external legs; if external_leg is False, it raises an error
    '''

    if external_leg == False:
        print("PBC require external legs")
        sys.error(1)

    nodes = []

    for ii in range(num_sites):
        
        shape = (bond_dim, phisical_dim, bond_dim)

        tensor = np.random.rand(*shape) + 1j * np.random.rand(*shape)
        if debug == True:
            print("added tensor of dimensions (", *shape, ")")

        # To prevent overflow
        nodes.append(normalize_tensor(tensor))

    norma = norm_only_MPS(nodes, external_leg=external_leg, PBC=False)
    new_nodes = [nodes[ii]*np.power(norma, -1./num_sites) for ii in range(len(nodes))]

    return new_nodes


# ==========================================================================================
# Create the DUAL
# ==========================================================================================

def create_dual_MPS_tensor(MPS):

    '''
    3. create_dual_MPS_tensor(MPS):
    DESCRIPTION:
       - Creates the dual of a Matrix Product State (MPS) by taking the conjugate of each tensor
    PROPERTIES:
       - Generates the dual MPS from the given MPS
    INPUT PARAMETERS:
       - MPS (List of numpy.ndarray): Array of tensors representing the MPS
    OUTPUT:
       - List of numpy.ndarray: Array of tensors representing the dual MPS
       - Note: Indexes are kept in the correct direction
       - Note: No need to specify an external leg
    '''

    dual_MPS = []

    for tensor in MPS:
        dual_MPS.append(np.conjugate(tensor))

    return dual_MPS


# ==========================================================================================
# Evaluate the NORM
# ==========================================================================================


def normalize_tensor(tensor):

    '''
    4. normalize_tensor(tensor):
    DESCRIPTION:
       - Normalizes a given tensor using singular value decomposition (SVD)
    PROPERTIES:
       - Computes the SVD of the tensor and normalizes it using singular values
    INPUT PARAMETERS:
       - tensor (numpy.ndarray): Tensor to be normalized
    OUTPUT:
       - numpy.ndarray: Normalized tensor
       - Note: The normalization is performed using SVD
    '''

    U, S, Vh = np.linalg.svd(tensor.reshape(-1, tensor.shape[-1]), full_matrices=False)

    S_normalized = S / np.sqrt(np.sum(S**2))

    tensor_normalized = np.dot(U, np.dot(np.diag(S_normalized), Vh)).reshape(tensor.shape)

    return tensor_normalized


def norm_only_MPS(MPS, external_leg=True, debug=False, PBC=False):

    '''
    norm_only_MPS(MPS, external_leg=True, debug=False, PBC=False):
    DESCRIPTION:
       - Computes the norm of a Matrix Product State (MPS) using tensor network methods
    PROPERTIES:
       - Constructs the dual MPS and connects tensors to compute the norm
       - Supports periodic boundary conditions (PBC) and external legs
    INPUT PARAMETERS:
       - MPS (list of numpy.ndarray): List of tensors representing the MPS
       - external_leg (bool): Flag indicating whether the MPS has an external leg (default: True)
       - debug (bool): Flag for enabling debug mode (default: False)
       - PBC (bool): Flag indicating periodic boundary conditions (default: False)
    OUTPUT:
       - float: Norm of the MPS
       - Note: The norm is computed using tensor network contraction
    '''

    dual_MPS = create_dual_MPS_tensor(MPS)

    MPS_nodes = [tn.Node(tensor) for tensor in MPS]
    dual_MPS_nodes = [tn.Node(tensor) for tensor in dual_MPS]

    if PBC == False:

        if external_leg == True:

            for ii in range(len(MPS_nodes)):
                tn.connect(MPS_nodes[ii][1], dual_MPS_nodes[ii][1])

            for ii in range(1,len(MPS_nodes)):
                tn.connect(MPS_nodes[ii-1][2], MPS_nodes[ii][0])
                tn.connect(dual_MPS_nodes[ii-1][2], dual_MPS_nodes[ii][0])
                
            tn.connect(MPS_nodes[0][0], dual_MPS_nodes[0][0])
            tn.connect(MPS_nodes[-1][2], dual_MPS_nodes[-1][2])

        else:

            tn.connect(MPS_nodes[0][0], dual_MPS_nodes[0][0])
            for ii in range(1, len(MPS_nodes)-1):
                tn.connect(MPS_nodes[ii][1], dual_MPS_nodes[ii][1])
            tn.connect(MPS_nodes[-1][1], dual_MPS_nodes[-1][1])

            tn.connect(MPS_nodes[0][1], MPS_nodes[1][0])
            tn.connect(dual_MPS_nodes[0][1], dual_MPS_nodes[1][0])
            for ii in range(2, len(MPS_nodes)):
                tn.connect(MPS_nodes[ii-1][2], MPS_nodes[ii][0])
                tn.connect(dual_MPS_nodes[ii-1][2], dual_MPS_nodes[ii][0])

    else:

        for ii in range(len(MPS_nodes)):
            # Vertical
            tn.connect(MPS_nodes[ii][1], dual_MPS_nodes[ii][1])
            # Horizontal
            tn.connect(MPS_nodes[ii-1][2], MPS_nodes[ii][0])
            tn.connect(dual_MPS_nodes[ii-1][2], dual_MPS_nodes[ii][0])

    inner_product = tn.contractors.greedy(MPS_nodes + dual_MPS_nodes)
    norma = np.sqrt(inner_product.tensor.item())
    if debug == True:
        print("Norm =", norma)

    return norma.real


def split_two_right(tensore, limitig, shape_left, shape_right, renormalize=False):

    '''
    split_two_right(tensore, limitig, shape_left, shape_right, renormalize=False):
    DESCRIPTION:
       - Splits a tensor into two tensors, keeping the right part
    PROPERTIES:
       - Performs a singular value decomposition (SVD) to split the tensor
       - Supports optional renormalization of singular values
    INPUT PARAMETERS:
       - tensore (numpy.ndarray): Tensor to be split
       - limitig (int): Number of singular values to keep
       - shape_left (tuple): Shape of the left tensor after splitting
       - shape_right (tuple): Shape of the right tensor after splitting
       - renormalize (bool): Flag indicating whether to renormalize singular values (default: False)
    OUTPUT:
       - tuple: Two tensors resulting from the split operation (left_tens, right_tens)
    '''

    U, S, Vh = np.linalg.svd(tensore, full_matrices=False)
    U = U[:, :limitig]
    S = S[:limitig]
    Vh = Vh[:limitig, :]

    if renormalize == True:
        S = S / np.sqrt((np.sum(S**2)))

    left_tens = U.reshape(shape_left)
    following_tens = np.tensordot(np.diag(S[:]), Vh, axes=([1], [0]))
    right_tens = following_tens.reshape(shape_right)

    return left_tens, right_tens


def split_two_left(tensore, limitig, shape_left, shape_right, renormalize=False):

    '''
    11. split_two_left(tensore, limitig, shape_left, shape_right, renormalize=False):
    DESCRIPTION:
       - Splits a tensor into two tensors, keeping the left part
    PROPERTIES:
       - Performs a singular value decomposition (SVD) to split the tensor
       - Supports optional renormalization of singular values
    INPUT PARAMETERS:
       - tensore (numpy.ndarray): Tensor to be split
       - limitig (int): Number of singular values to keep
       - shape_left (tuple): Shape of the left tensor after splitting
       - shape_right (tuple): Shape of the right tensor after splitting
       - renormalize (bool): Flag indicating whether to renormalize singular values (default: False)
    OUTPUT:
       - tuple: Two tensors resulting from the split operation (left_tens, right_tens)
    '''

    U, S, Vh = np.linalg.svd(tensore, full_matrices=False)
    U = U[:, :limitig]
    S = S[:limitig]
    Vh = Vh[:limitig, :]

    if renormalize == True:
        S = S / np.sqrt((np.sum(S**2)))

    following_tens = np.tensordot(U, np.diag(S[:]), axes=([1], [0]))
    left_tens = following_tens.reshape(shape_left)
    right_tens = Vh.reshape(shape_right)

    return left_tens, right_tens


# ==========================================================================================
# Evaluate the ENERGY
# ==========================================================================================

def total_energy (MPS_tens, MPO, dual_MPS, external_leg=False, PBC=False, MPO_first=None):

    '''
    12. total_energy(MPS_tens, MPO, dual_MPS, external_leg=False, PBC=False, MPO_first=None):
    DESCRIPTION:
       - Computes the total energy of the system
    PROPERTIES:
       - Computes the contraction of tensors representing the MPS, MPO, and its dual
       - Supports periodic boundary conditions (PBC)
       - Connects tensors according to the network structure
    INPUT PARAMETERS:
       - MPS_tens (list of numpy.ndarray): List of tensors representing the MPS
       - MPO (numpy.ndarray): Tensor representing the MPO
       - dual_MPS (list of numpy.ndarray): List of tensors representing the dual of the MPS
       - external_leg (bool): Flag indicating the presence of an external leg (default: False)
       - PBC (bool): Flag indicating periodic boundary conditions (default: False)
       - MPO_first (numpy.ndarray): Tensor representing the first MPO tensor for PBC (default: None)
    OUTPUT:
       - float: Total energy of the system
    '''

    if PBC == False:

        MPO_tens = [MPO for ii in range(len(MPS_tens))]
        MPO_tens[0] = np.tensordot(np.array([1, 0, 0]), MPO_tens[0], axes=([0], [0]))
        MPO_tens[-1] = np.tensordot(np.array([0, 0, 1]), MPO_tens[-1], axes=([0], [1]))

        MPS_nodes = [tn.Node(tensor) for tensor in MPS_tens]
        MPO_nodes = [tn.Node(tensor) for tensor in MPO_tens]
        dual_MPS_nodes = [tn.Node(tensor) for tensor in dual_MPS]

        # Vertical
        if external_leg == False:
            tn.connect(MPS_nodes[0][0], MPO_nodes[0][2])
            tn.connect(dual_MPS_nodes[0][0], MPO_nodes[0][1])
        else:
            tn.connect(MPS_nodes[0][1], MPO_nodes[0][2])
            tn.connect(dual_MPS_nodes[0][1], MPO_nodes[0][1])
            tn.connect(MPS_nodes[0][0], dual_MPS_nodes[0][0])

        for ii in range(1, len(MPS_nodes)-1):
            tn.connect(MPS_nodes[ii][1], MPO_nodes[ii][3])
            tn.connect(dual_MPS_nodes[ii][1], MPO_nodes[ii][2])
        
        if external_leg == False:
            tn.connect(MPS_nodes[-1][1], MPO_nodes[-1][2])
            tn.connect(dual_MPS_nodes[-1][1], MPO_nodes[-1][1])
        else:
            tn.connect(MPS_nodes[-1][1], MPO_nodes[-1][2])
            tn.connect(dual_MPS_nodes[-1][1], MPO_nodes[-1][1])
            tn.connect(MPS_nodes[-1][2], dual_MPS_nodes[-1][2])

        # Horizontal
        if external_leg == False:
            tn.connect(MPS_nodes[0][1], MPS_nodes[1][0])
            tn.connect(dual_MPS_nodes[0][1], dual_MPS_nodes[1][0])
            tn.connect(MPO_nodes[0][0], MPO_nodes[1][0])
        else:
            tn.connect(MPS_nodes[0][2], MPS_nodes[1][0])
            tn.connect(dual_MPS_nodes[0][2], dual_MPS_nodes[1][0])
            tn.connect(MPO_nodes[0][0], MPO_nodes[1][0])

        for ii in range(2, len(MPS_nodes)):
            tn.connect(MPS_nodes[ii-1][2], MPS_nodes[ii][0])
            tn.connect(dual_MPS_nodes[ii-1][2], dual_MPS_nodes[ii][0])
            tn.connect(MPO_nodes[ii-1][1], MPO_nodes[ii][0])

    else:
    
        MPO_tens = [MPO for ii in range(len(MPS_tens))]
        MPO_tens[0] = MPO_first

        MPS_nodes = [tn.Node(tensor) for tensor in MPS_tens]
        MPO_nodes = [tn.Node(tensor) for tensor in MPO_tens]
        dual_MPS_nodes = [tn.Node(tensor) for tensor in dual_MPS]

        # Vertical
        for ii in range(len(MPS_nodes)):
            tn.connect(MPS_nodes[ii][1], MPO_nodes[ii][3])
            tn.connect(dual_MPS_nodes[ii][1], MPO_nodes[ii][2])

        # Horizontal
        for ii in range(len(MPS_nodes)):
            tn.connect(MPS_nodes[ii-1][2], MPS_nodes[ii][0])
            tn.connect(dual_MPS_nodes[ii-1][2], dual_MPS_nodes[ii][0])
            tn.connect(MPO_nodes[ii-1][1], MPO_nodes[ii][0])

    energy = tn.contractors.greedy(MPS_nodes + dual_MPS_nodes + MPO_nodes)

    return energy.tensor.item()

