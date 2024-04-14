import total
import time
import numpy as np
import csv


# ==================================================================
# DESCRIPTION:
# ------------------------------------------------------------------
# This Python program conducts a series of calculations using the
# Density Matrix Renormalization Group (DMRG) algorithm with Lanczos
# iteration for generic MPO. It iterates over different bond
# dimensions, computes the total energy, and records the time taken
# for each calculation
# ------------------------------------------------------------------
# Libraries Used:
# ------------------------------------------------------------------
# - import total: Contains the DMRG_lanczos function for performing
#   DMRG calculations
# - import time: Used for measuring the execution time of each
#   alculation
# - import numpy as np: Utilized for numerical computations and
#   data manipulation
# - import csv: Used for writing information about the calculations
#   to a CSV file
# ------------------------------------------------------------------
# Parameters:
# ------------------------------------------------------------------
# - change (str): Path to the directory where the data will be saved
# - bond_dims (list of int): List of bond dimensions to iterate over
# - num_sites (int): Number of sites in the Ising model
# - physical_dim (int): Physical dimension of the Ising model
# - MPO (str): Type of Matrix Product Operator (MPO) used in the
#   calculation (e.g., "ising_density", "ising_energy")
# - lanczos_iter (int): Number of Lanczos iterations to perform
# - ext_leg (bool): Flag indicating the presence of an external leg
#   in the calculation
# - eps (float): Tolerance parameter for convergence criteria
# - JJ (float): Coupling constant parameter of the Ising model
# - HH (float): External magnetic field parameter of the Ising model
# - debug (bool): Flag indicating whether to print debug information
# - tempi (list of float): List to store the execution times of
#   each calculation
# ------------------------------------------------------------------
# Output:
# ------------------------------------------------------------------
# - A CSV file containing information about each calculation,
#   including parameters such as number of sites, physical dimension,
#   bond dimension, MPO type, Lanczos iterations, presence of
#   external leg, coupling constant, and external magnetic field
# - Separate folders for each calculation containing the computed data
# - A text file containing the execution times of each calculation
# ------------------------------------------------------------------


change = "create_data/data/ising/change_b"
bond_dims = [ii+1 for ii in range(20)]


num_sites = 20
physical_dim = 2
MPO = "ising_density"
lanczos_iter = 2
ext_leg = True
eps = 1e-3
JJ = 1
HH = 0
debug = False
tempi = []

dictionary = {"num_sites" : num_sites,
                "physical_dim" : physical_dim,
                "bond_dim" : bond_dims,
                "MPO" : MPO,
                "lanczos_iter" : lanczos_iter,
                "ext_leg" : ext_leg,
                "H" : HH,
                "J" : JJ}

name_info = change + "/info_run"
with open(name_info, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
    writer.writeheader()
    writer.writerow(dictionary)

    iteration = 0
    for bond_dim in bond_dims:

        time0 = time.time()
        name_folder = change + "/" + str(iteration)
        name = "N" + str(num_sites) + "_pd" + str(physical_dim) + "_bd" + str(bond_dim) + "_MPO" + str(MPO) + "_lan" + str(lanczos_iter) + "_ext" + str(ext_leg)

        total.DMRG_lanczos(num_sites, physical_dim, bond_dim, MPO, ext_leg, eps, HH, JJ, name_folder, PBC = True)
        print("Completed  calculation", name, "in time", time.time()-time0)

        iteration += 1
        tempi.append(time.time() - time0)


    name_times = change + "/time_run"
    np.savetxt(name_times, tempi, delimiter=',')
