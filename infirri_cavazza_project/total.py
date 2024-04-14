import tensorflow as tf
import tensornetwork as tn
import numpy as np
import matplotlib.pyplot as plt

import time
import functions_mps
import functions_dmrg
import sys

from itertools import permutations
from tensornetwork import FiniteMPS


# =============================================================================
# Important quantities
# =============================================================================

sz = np.array([[1, 0], [0, -1]])
sx = np.array([[0, 1], [1, 0]])

id = np.eye(2)
oo = np.diag([0, 0])


# =============================================================================
# Function
# =============================================================================

def DMRG_lanczos(num_sites, physical_dim, bond_dim, MPO, ext_leg, eps, HH, JJ, name_folder, PBC = False, debug=False):

    if MPO == "ising_energy":
        # To retrieve energy
        MPO_site = np.array([
            [id, sz, -HH*sx],
            [oo, oo, -JJ*sz],
            [oo, oo, id]
        ])
        MPO_first = np.array([
            [oo, oo, oo],
            [-JJ*sz, oo, oo],
            [id, sz, -HH*sx]
        ])
    elif MPO == "ising_density":
        # To retrieve the energy desity
        MPO_site = np.array([
            [id, sz, -HH*sx],
            [oo, oo, -JJ*sz],
            [oo, oo, id]
        ])/(num_sites**(1/num_sites))
        # MPO_first = MPO_site
        MPO_first = np.array([
            [oo, oo, oo],
            [-JJ*sz, oo, oo],
            [id, sz, -HH*sx]
        ])/(num_sites**(1/num_sites))
    elif MPO == "identity":
        # For Debugging reasons (to test if the energy is coherent)
        MPO_site = np.array([
            [id, oo, id],
            [oo, oo, oo],
            [oo, oo, id]
        ])/(num_sites**(1/num_sites))
        MPO_first = MPO_site
    else:
        NameError("No MPO found for string inserted")

    if PBC == False:
        MPS_tens_init = functions_mps.create_random_MPS_normalized(num_sites, physical_dim, bond_dim, debug=False, external_leg=ext_leg)
    else:
        MPS_tens_init = functions_mps.create_random_MPS_normalized_PBC(num_sites, physical_dim, bond_dim, debug=False, external_leg=ext_leg)

    energia = []
    norme = []
    energia_PBC = []

    converged_PBC = False
    counter_PBC = 0
    energy_last_PBC = 1000
    split = 0

    while not converged_PBC:

        converged = False
        counter = 0
        energy_last = 1000

        init_center = int(num_sites/2)-1
        MPS_tens = tn.FiniteMPS(tensors=MPS_tens_init, canonicalize=True, center_position=init_center)

        time_init = time.time()
        time_iter = time_init

        # while condition on the sweep
        while not converged:

            # sweep left
            for ll in range(init_center, 1, -1):

                AA, BB, NN, EE = functions_dmrg.iteration_sweep(ll, MPS_tens, MPO_site, right_movement=False, ext_leg=ext_leg, PBC=False)

                # Update norm and energy
                norme.append(NN)
                energia.append(EE)

                # Back to OUR form:
                MPS_tens.tensors[ll-1] = AA
                MPS_tens.tensors[ll] = BB

            # sweep right
            for ll in range(2, num_sites-1, 1):

                AA, BB, NN, EE = functions_dmrg.iteration_sweep(ll, MPS_tens, MPO_site, right_movement=True, ext_leg=ext_leg, PBC=False)

                # Update norm and energy
                norme.append(NN)
                energia.append(EE)

                # Back to OUR form:
                MPS_tens.tensors[ll-1] = AA
                MPS_tens.tensors[ll] = BB

            # sweep left
            for ll in range(num_sites-2, init_center, -1):
            
                AA, BB, NN, EE = functions_dmrg.iteration_sweep(ll, MPS_tens, MPO_site, right_movement=False, ext_leg=ext_leg, PBC=False)

                # Update norm and energy
                norme.append(NN)
                energia.append(EE)

                # Back to OUR form:
                MPS_tens.tensors[ll-1] = AA
                MPS_tens.tensors[ll] = BB

            if abs((energy_last - EE)/EE) < eps:
                converged = True

            energy_last = EE
            counter += 1

            if debug == True:
                print(f"Iteration {counter} completed at time {time.time() - time_iter} with energy {energy_last} and norm {NN}")
            time_iter = time.time()

        if PBC == False:

            converged_PBC = True

        else:
            
            norma_PBC = functions_mps.norm_only_MPS(MPS_tens.tensors, external_leg=True, debug=False, PBC=True)
            MPS_tens.tensors = [MPS_tens.tensors[ii]*np.power(norma_PBC, -1./num_sites) for ii in range(len(MPS_tens.tensors))]
            energy_PBC = functions_mps.total_energy(MPS_tens.tensors, MPO_site, functions_mps.create_dual_MPS_tensor(MPS_tens.tensors),
                                                    external_leg=True, PBC=True, MPO_first=MPO_first) # change me
            energia_PBC.append(energy_PBC)


            if abs((energy_last_PBC - energy_PBC)/energy_PBC) < eps:
                converged_PBC = True

            energy_last_PBC = energy_PBC
            counter_PBC += 1

            random_pos = np.random.randint(0, num_sites, 1)
            while random_pos == split:
                random_pos = np.random.randint(0, num_sites, 1)
            split = random_pos

            new_split = [(ii + int(split))%num_sites for ii in range(num_sites)]
            MPS_tens_init = [MPS_tens.tensors[ii] for ii in new_split]

            if debug == True:
                print(f"Completed splitting at cut edge {new_split[0]} with energy {energy_PBC} and norm {norma_PBC}")

    if debug == True:
        print(f"Converged after {counter_PBC} splits")

    if PBC == True:

        name_e = name_folder + "_energy.txt"
        np.savetxt(name_e, energia_PBC)

    else:

        name_e = name_folder + "_energy.txt"
        np.savetxt(name_e, energia)

        name_n = name_folder + "_norm.txt"
        np.savetxt(name_n, norme)
