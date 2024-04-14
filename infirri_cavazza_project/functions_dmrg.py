# functions_dmrg.py>

import tensornetwork as tn
import numpy as np
import functions_mps


# ==================================================================
# DESCRIPTION:
# ------------------------------------------------------------------
# This Python script provides functions for manipulating Matrix Product
# States (MPS) and evaluating physical quantities such as total energy
# It includes functions for creating random normalized MPS, evaluating
# the norm of MPS, splitting MPS tensors, and computing the total energy
# of the MPS under the influence of a given Hamiltonian operator
# Additionally, it contains functions for performing environment
# contractions and implementing the Lanczos algorithm for DMRG
# ------------------------------------------------------------------
# Functions:
# ------------------------------------------------------------------
# 1. Left_starting_block(MPS, DUAL, MPO, PBC=False):
#    - Computes the left starting block tensor for the DMRG algorithm
# 2. Left_env_contraction(L_MPS, L_MPS_D, MPO_site):
#    - Computes the left environment contraction in DMRG algorithm
# 3. Right_starting_block(MPS, DUAL, MPO):
#    - Computes the right starting block tensor for the DMRG algorithm
# 4. Right_env_contraction(R_MPS, R_MPS_D, MPO_site):
#    - Computes the right environment contraction in DMRG algorithm
# 5. Effective_Ham(L_MPS, L_MPS_D, R_MPS, R_MPS_D, MPO_site):
#    - Computes the effective Hamiltonian for the DMRG algorithm
# 6. Lanczos_first_term(psi_present, Eff_Hamiltonian):
#    - Computes the first term in the Lanczos algorithm
# 7. Lanczos_alpha(psi_present, FT):
#    - Computes alpha in the Lanczos algorithm
# 8. norm_two_sites(MPS, external_leg=True, debug=False):
#    - Computes the norm of two sites in the MPS
# 9. iteration_sweep(ll, MPS_tens, MPO_site, right_movement=True,
#                    debug=False, ext_leg=False, PBC=False, MPO_first=None):
#    - Performs one iteration sweep in the DMRG algorithm
# ==================================================================


# ==========================================================================================
# Environment contractions (Left and Right)
# ==========================================================================================

def Left_starting_block(MPS, DUAL, MPO, PBC=False):
    '''
    DESCRIPTION:
       - Computes the left starting block tensor for the DMRG algorithm
    PROPERTIES:
       - Contracts tensors to obtain the left starting block
    INPUT PARAMETERS:
       - MPS (list of numpy.ndarray): List of tensors representing the MPS
       - DUAL (list of numpy.ndarray): List of tensors representing the dual MPS
       - MPO (numpy.ndarray): Tensor representing the MPO
       - PBC (bool): Flag indicating periodic boundary conditions (default: False)
    OUTPUT:
       - numpy.ndarray: Left starting block tensor
    '''

    node_A = tn.Node(np.array(DUAL), name = "A")
    node_B = tn.Node(np.array(MPO), name = "B")
    node_C = tn.Node(np.array(MPS), name = "C")

    A_B_phy = node_A[1] ^ node_B[1]
    B_C_phy = node_B[2] ^ node_C[1]
    A_C_imm = node_A[0] ^ node_C[0]

    network_ABC = tn.reachable(node_A) | tn.reachable(node_B) | tn.reachable(node_C)
    block_ABC = tn.contractors.greedy(network_ABC, output_edge_order = [node_A[2],node_B[0],node_C[2]])
    
    return block_ABC.tensor


def Left_env_contraction(L_MPS, L_MPS_D, MPO_site):
    '''
    DESCRIPTION:
       - Computes the left environment contraction in DMRG algorithm
    PROPERTIES:
       - Contracts tensors to compute the left environment
    INPUT PARAMETERS:
       - L_MPS (list of numpy.ndarray): List of tensors representing the left MPS
       - L_MPS_D (list of numpy.ndarray): List of tensors representing the dual left MPS
       - MPO_site (numpy.ndarray): Tensor representing the MPO for a site
    OUTPUT:
       - numpy.ndarray: Result of the left environment contraction
    '''

    v1 = np.array([1] + [0 for ii in range(MPO_site.shape[0] - 1)])
    left_MPO = np.tensordot(v1, MPO_site, axes=([0], [0]))
    Left_env = Left_starting_block(L_MPS[0], L_MPS_D[0], left_MPO)

    if len(L_MPS) > 1: 
        for ii in range(1,len(L_MPS)): 

            node_ABC = tn.Node(Left_env, name = "ABC")
            node_D = tn.Node(np.array(L_MPS_D[ii]), name = "D")
            node_E = tn.Node(np.array(MPO_site), name = "E")
            node_F = tn.Node(np.array(L_MPS[ii]), name = "F")

            # bond edges
            ABC_D_bond = node_ABC[0] ^ node_D[0]
            ABC_E_bond = node_ABC[1] ^ node_E[0]
            ABC_F_bond = node_ABC[2] ^ node_F[0]

            # physical edges
            D_E_phy =  node_D[1] ^ node_E[2]
            E_F_phy =  node_E[3] ^ node_F[1]

            network_ABC_DEF = tn.reachable(node_ABC) | tn.reachable(node_D) | tn.reachable(node_E) | tn.reachable(node_F) 
            Left_env = tn.contractors.greedy(network_ABC_DEF, output_edge_order = [node_D[2],node_E[1],node_F[2]])
    
        result = Left_env.tensor
    else: 
        result = Left_env

    return result 


def Right_starting_block(MPS, DUAL, MPO):
    '''
    DESCRIPTION:
       - Computes the right starting block tensor for the DMRG algorithm
    PROPERTIES:
       - Contracts tensors to obtain the right starting block
    INPUT PARAMETERS:
       - MPS (list of numpy.ndarray): List of tensors representing the MPS
       - DUAL (list of numpy.ndarray): List of tensors representing the dual MPS
       - MPO (numpy.ndarray): Tensor representing the MPO
    OUTPUT:
       - numpy.ndarray: Right starting block tensor
    '''

    node_A = tn.Node(np.array(DUAL), name = "A")
    node_B = tn.Node(np.array(MPO), name = "B") 
    node_C = tn.Node(np.array(MPS), name = "C") 

    A_B_phy = node_A[1] ^ node_B[1]
    B_C_phy = node_B[2] ^ node_C[1]
    A_C_imm = node_A[2] ^ node_C[2]

    network_ABC = tn.reachable(node_A) | tn.reachable(node_B) | tn.reachable(node_C)
    block_ABC = tn.contractors.greedy(network_ABC, output_edge_order = [node_A[0],node_B[0],node_C[0]])
    
    return block_ABC.tensor


def Right_env_contraction(R_MPS, R_MPS_D, MPO_site):
    '''
    DESCRIPTION:
       - Computes the right environment contraction in DMRG algorithm
    PROPERTIES:
       - Contracts tensors to compute the right environment
    INPUT PARAMETERS:
       - R_MPS (list of numpy.ndarray): List of tensors representing the right MPS
       - R_MPS_D (list of numpy.ndarray): List of tensors representing the dual right MPS
       - MPO_site (numpy.ndarray): Tensor representing the MPO for a site
    OUTPUT:
       - numpy.ndarray: Result of the right environment contraction
    '''

    v1 = np.array([0 for ii in range(MPO_site.shape[0] - 1)] + [1])
    right_MPO = np.tensordot(v1, MPO_site, axes=([0], [1]))
    Right_env = Right_starting_block(R_MPS[-1], R_MPS_D[-1], right_MPO)

    if len(R_MPS) > 1:

        for ii in range(len(R_MPS)-2, -1, -1): 

            node_ABC = tn.Node(Right_env, name = "ABC")
            node_D = tn.Node(np.array(R_MPS_D[ii]), name = "D")
            node_E = tn.Node(np.array(MPO_site), name = "E")
            node_F = tn.Node(np.array(R_MPS[ii]), name = "F")

            # bond edges
            ABC_D_bond = node_ABC[0] ^ node_D[2]
            ABC_E_bond = node_ABC[1] ^ node_E[1]
            ABC_F_bond = node_ABC[2] ^ node_F[2]

            # physical edges
            D_E_phy =  node_D[1] ^ node_E[2]
            E_F_phy =  node_E[3] ^ node_F[1]

            network_ABC_DEF = tn.reachable(node_ABC) | tn.reachable(node_D) | tn.reachable(node_E) | tn.reachable(node_F) 
            Right_env = tn.contractors.greedy(network_ABC_DEF, output_edge_order = [node_D[0],node_E[0],node_F[0]])

        result = Right_env.tensor
    else: 
        result = Right_env

    return result 



# ==========================================================================================
# Lanczos Algorithm
# ==========================================================================================


def Effective_Ham(L_MPS, L_MPS_D, R_MPS, R_MPS_D, MPO_site):
    '''
    DESCRIPTION:
       - Computes the effective Hamiltonian for the DMRG algorithm
    PROPERTIES:
       - Contracts tensors to compute the effective Hamiltonian
    INPUT PARAMETERS:
       - L_MPS (list of numpy.ndarray): List of tensors representing the left MPS
       - L_MPS_D (list of numpy.ndarray): List of tensors representing the dual left MPS
       - R_MPS (list of numpy.ndarray): List of tensors representing the right MPS
       - R_MPS_D (list of numpy.ndarray): List of tensors representing the dual right MPS
       - MPO_site (numpy.ndarray): Tensor representing the MPO for a site
    OUTPUT:
       - numpy.ndarray: Resulting effective Hamiltonian tensor
    '''
    
    Left_env = Left_env_contraction(L_MPS, L_MPS_D, MPO_site)
    Right_env = Right_env_contraction(R_MPS, R_MPS_D, MPO_site)

    node_L = tn.Node(Left_env, name = "Left")   # (2,3,3)
    node_R = tn.Node(Right_env, name = "Right") # (2,3,2)
    node_w1 = tn.Node(MPO_site, name = "w1")    # (3,3,2,2)
    node_w2 = tn.Node(MPO_site, name = "w2")    # (3,3,2,2)

    L_w1_bond = node_L[1] ^ node_w1[0] 
    w1_w2_bond = node_w1[1] ^ node_w2[0]
    w2_R_bond = node_w2[1] ^ node_R[1]

    network_Ham = tn.reachable(node_L) | tn.reachable(node_R) | tn.reachable(node_w1) | tn.reachable(node_w2) 
    # I kept consistence the order of the index: (bond_up, bond_up, bond_down, bond_down, phy_up, phy_up, phy_down, phy_down)
    Effective_Ham = tn.contractors.greedy(network_Ham, output_edge_order = [node_L[0], node_w1[2], node_w2[2], node_R[0], 
                                                                            node_L[2], node_w1[3], node_w2[3] ,node_R[2] ])


    return Effective_Ham.tensor



def Lanczos_first_term(psi_present, Eff_Hamiltonian):
    '''
    DESCRIPTION:
       - Computes the first term in the Lanczos algorithm
    PROPERTIES:
       - Contracts tensors to compute the first term
    INPUT PARAMETERS:
       - psi_present (numpy.ndarray): Tensor representing the present wavefunction
       - Eff_Hamiltonian (numpy.ndarray): Tensor representing the effective Hamiltonian
    OUTPUT:
       - numpy.ndarray: Resulting first term tensor
    '''
    
    node_Ham = tn.Node(Eff_Hamiltonian, name = "Effective_H")
    node_Psi_present = tn.Node(psi_present, name = "Pso_present")


    Ham_psi_bond = node_Ham[4] ^ node_Psi_present[0]
    psi_Ham_bond = node_Psi_present[3] ^ node_Ham[7]

    Ham_psi_phy = node_Ham[5] ^ node_Psi_present[1]
    Ham_psi_phy = node_Ham[6] ^ node_Psi_present[2]

    network_first_term = tn.reachable(node_Ham) | tn.reachable(node_Psi_present) 
    lanczos_first_term = tn.contractors.greedy(network_first_term, output_edge_order = [node_Ham[0], node_Ham[1], node_Ham[2], node_Ham[3]])

    return lanczos_first_term.tensor


def Lanczos_alpha(psi_present, FT):
    '''
    DESCRIPTION:
       - Computes alpha in the Lanczos algorithm
    PROPERTIES:
       - Contracts tensors to compute alpha
    INPUT PARAMETERS:
       - psi_present (numpy.ndarray): Tensor representing the present wavefunction
       - FT (numpy.ndarray): Tensor representing the first term
    OUTPUT:
       - numpy.ndarray: Resulting alpha
    '''

    dual_psi_present = np.conjugate(psi_present)

    node_FT = tn.Node(FT, name = "FT") 
    node_d_psi_p = tn.Node(dual_psi_present, name = "Tensor_1") # dual psi present
    

    FT_d_psi_bond = node_FT[0] ^ node_d_psi_p[0] # bond 
    FT_d_psi_bond = node_FT[1] ^ node_d_psi_p[1] # phy 
    FT_d_psi_bond = node_FT[2] ^ node_d_psi_p[2] # phy 
    FT_d_psi_bond = node_FT[3] ^ node_d_psi_p[3] # bond


    network_alpha = tn.reachable(node_FT) | tn.reachable(node_d_psi_p) 
    lanczos_alpha = tn.contractors.greedy(network_alpha)

    return lanczos_alpha.tensor


def norm_two_sites(MPS, external_leg=True, debug=False):
    '''
    DESCRIPTION:
       - Computes the norm of two sites in the MPS
    PROPERTIES:
       - Contracts tensors to compute the norm
    INPUT PARAMETERS:
       - MPS (list of numpy.ndarray): List of tensors representing the MPS
       - external_leg (bool): Flag indicating the presence of an external leg (default: True)
       - debug (bool): Flag indicating debug mode (default: False)
    OUTPUT:
       - float: Resulting norm
    '''

    dual_MPS = functions_mps.create_dual_MPS_tensor(MPS)

    node_MPS = tn.Node(MPS)
    node_d_MPS = tn.Node(np.conjugate(MPS)) 

    bond_ext_L = node_MPS[0] ^ node_d_MPS[0] # bond
    bond_ext_R = node_MPS[1] ^  node_d_MPS[1] # bind
    bond_int_L = node_MPS[2] ^  node_d_MPS[2] # phy
    bond_int_R = node_MPS[3] ^  node_d_MPS[3]  # phy

    network_psi_old = tn.reachable(node_MPS) | tn.reachable(node_d_MPS) 
    inner_product = tn.contractors.greedy(network_psi_old)

    norm = np.sqrt(inner_product.tensor.item())

    if debug == True:
        print("Norm =", norm)
        print("tensor 1  =", MPS[0].shape)
        print("tensor 2  =", MPS[1].shape)
        print("dual tensor 1  =", dual_MPS[0].shape)
        print("dual tensor 2  =", dual_MPS[1].shape)

    return norm.real


def iteration_sweep(ll, MPS_tens, MPO_site, right_movement=True, debug=False, ext_leg=False, PBC=False, MPO_first=None):
    '''
    DESCRIPTION:
       - Performs one iteration sweep in the DMRG algorithm
    PROPERTIES:
       - Computes tensors in the DMRG algorithm
    INPUT PARAMETERS:
       - ll (int): Lower limit of the iteration sweep
       - ul (int): Upper limit of the iteration sweep
       - MPS_L (list of numpy.ndarray): List of tensors representing the left MPS
       - DUAL_L (list of numpy.ndarray): List of tensors representing the dual left MPS
       - MPS_R (list of numpy.ndarray): List of tensors representing the right MPS
       - DUAL_R (list of numpy.ndarray): List of tensors representing the dual right MPS
       - MPO (numpy.ndarray): Tensor representing the MPO
       - Env (numpy.ndarray): Tensor representing the environment
       - Env_d (numpy.ndarray): Tensor representing the dual environment
       - debug (bool): Flag indicating debug mode (default: False)
    OUTPUT:
       - list of numpy.ndarray: Updated left MPS
       - list of numpy.ndarray: Updated dual left MPS
       - list of numpy.ndarray: Updated right MPS
       - list of numpy.ndarray: Updated dual right MPS
       - numpy.ndarray: Updated environment
       - numpy.ndarray: Updated dual environment
    '''

    left = ll - 1
    right = ll + 1

    MPS_tens.canonicalize()

    # Back in ordered form (OUR):
    ordered = [MPS_tens.tensors[ii] for ii in range(len(MPS_tens))]
    ordered_dual = functions_mps.create_dual_MPS_tensor(ordered)
    
    if debug == True:
        print("NORM left ", functions_mps.norm_only_MPS(ordered, external_leg=ext_leg, debug=False, PBC=False))
        print("Energy ", functions_mps.total_energy(ordered, MPO_site, ordered_dual, external_leg=ext_leg, PBC=False))
        print("norm 2 sites ", functions_mps.norm_only_MPS([ordered[ll-1], ordered[ll]], external_leg=True, PBC=False))

    H_eff = Effective_Ham(ordered[:left], ordered_dual[:left],
                                            ordered[right:], ordered_dual[right:], MPO_site)

    psi_1 = np.tensordot(ordered[ll-1], ordered[ll], axes=([2], [0]))
    psi_1 = psi_1 / norm_two_sites(psi_1)

    Hpsi1 = Lanczos_first_term(psi_1, H_eff)
    alpha1 = Lanczos_alpha(psi_1, Hpsi1)

    psi_2 = Hpsi1 - alpha1*psi_1
    beta2 = norm_two_sites(psi_2)

    if ( beta2 > 1e-10):

        psi_2 = psi_2 / beta2
        Hpsi2 = Lanczos_first_term(psi_2, H_eff)

        alpha2 = Lanczos_alpha(psi_2, Hpsi2)
        M = [[alpha1, beta2], [beta2, alpha2]] 
        D,U = np.linalg.eigh(M)

        energy = D[0]
        out_psi = np.conjugate(U[0,0])*psi_1 + np.conjugate(U[1,0])*psi_2

    else: 

        energy = alpha1
        out_psi = psi_1


    # split psi_present with svd and update:
    limitig = ordered[ll-1].shape[2]
    matrix = out_psi.reshape(out_psi.shape[0]*out_psi.shape[1], out_psi.shape[2]*out_psi.shape[3])

    if right_movement == True:
        ordered[ll-1], ordered[ll] = functions_mps.split_two_right(matrix, limitig, ordered[ll-1].shape,
                                                                    ordered[ll].shape, renormalize=True)
    else:
        ordered[ll-1], ordered[ll] = functions_mps.split_two_left(matrix, limitig, ordered[ll-1].shape,
                                                                ordered[ll].shape, renormalize=True)

    ordered_dual = functions_mps.create_dual_MPS_tensor(ordered)
    norma = functions_mps.norm_only_MPS(ordered, external_leg=ext_leg, debug=False, PBC=False)
    energia = functions_mps.total_energy(ordered, MPO_site, ordered_dual, external_leg=ext_leg, PBC=False, MPO_first=MPO_first)

    if debug == True:
        print("NORMA post Lanczos", norma)
        print("Energy", energia)

    # Move orthogonality center
    if right_movement == True:
        MPS_tens.position(ll)
    else:
        MPS_tens.position(ll-1)

    return ordered[ll-1], ordered[ll], norma, energia
