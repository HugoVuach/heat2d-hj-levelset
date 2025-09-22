import numpy

import resolution_equation_chaleur
import resolution_equation_hamilton
import setup_resolution_equation
import sauvegarde_graphiques
import calcul_normes
import sauvegarde_iteration
import creation_maillage_prefractal
import _env

from numba import set_num_threads

def optimisation(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, pas_temps, terme_source, tableau_des_normales, crank_nicolson, start_iter=0, energies=[], vols=[]):
    """
    La fonction qui permet d'optimiser la forme du domaine Omega_+
    
    :param maillage: La matrice du maillage
    :param coeff_diffusion_chaud: Le coefficient de diffusion dans le milieu chaud
    :param coeff_diffusion_froid: Le coefficient de diffusion dans le milieu froid
    :param noeuds_lambda_condition: Les noeuds qui possède une condition au bord avec lambda
    :param nombre_iterations: Le nombre d'itérations que l'on souhaite faire
    :param multiplicateur_pas_temps: Toujours garder < 1, permet de réduire le pas de temps
    :param terme_source: Matrice avec le terme source dans l'équation (= 0 pour le problème principal)
    :param tableau_des_normales: Tableau avec les normales de chaque noeuds

    :return U(Omega_opti) et Omega_opti:

    """
    nombre_iterations_boucle_opti = 5
    energie = numpy.zeros(nombre_iterations_boucle_opti+1)
    volumes = numpy.zeros(nombre_iterations_boucle_opti+1)
    
    maillage0 = maillage.copy() # On réalise une copie du maillage initiale
    u0 = resolution_equation_chaleur.resolution(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, terme_source, False) if not crank_nicolson else resolution_equation_chaleur.resolution_crank_nicolson(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, pas_temps, terme_source, False)
    sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u0, nombre_iterations-1, "fig_u0_tf")

    N = maillage.shape[0]
    # Maillage carré donc même pas d'espace selon x ou y
    pas_espace = 1/N

    # Il faut respecter la condition CFL
    if not crank_nicolson:
        pas_temps = multiplicateur_pas_temps * (0.4) * ((pas_espace**2) / (2*max([coeff_diffusion_chaud, coeff_diffusion_froid]))) # On met 0.4 au lieu de 0.5 pour bien être strictement inférieur à dxdy/2max(D+,D-)
    
    if (start_iter > 0):
        for k in range(start_iter):
            volumes[k] = vols[k]
            energie[k] = energies[k]

    # On calcule d'abord l'énergie pour u_0 (pour avoir une référence)
    energie[start_iter] = calcul_energie(u0, pas_espace, pas_temps, maillage, maillage0)
    volumes[start_iter] = volume(pas_espace, maillage)

    print("Energie initiale :", energie[start_iter])
    print("Volume initial :", volumes[start_iter])

    k = start_iter # Itération de la boucle
    ene = 0
    old_psi = resolution_equation_hamilton.initialisation_psi(maillage, noeuds_lambda_condition)

    while k < nombre_iterations_boucle_opti:
        
        print("======[ Itération", (k+1) ,"]======")
        print("1. Résolution de l'équation de la chaleur")
        u = resolution_equation_chaleur.resolution(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, terme_source, False) if not crank_nicolson else resolution_equation_chaleur.resolution_crank_nicolson(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, pas_temps, terme_source, False)

        #sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u, nombre_iterations-1, "fig_u_final_tf_" + str(k))

        print("2. Résolution du problème adjoint")
        f_adj = (2*u) #Terme source pour le problème adjoint
        print(f_adj.shape)
        p = resolution_equation_chaleur.resolution(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, f_adj, False) if not crank_nicolson else resolution_equation_chaleur.resolution_crank_nicolson(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, pas_temps, f_adj, False)

        print("3. Calcul de l'énergie")
        ene = calcul_energie(u, pas_espace, pas_temps, maillage, maillage0)

        print("4. Descente de gradient, méthode level-set")
        mu = min([N*1e-4, 1])
        ene = calcul_energie(u, pas_espace, pas_temps, maillage, maillage0)
        vol = volume(pas_espace, maillage)

        while ene > energie[k] and (mu < 1 and mu > 1e-6):

            print("\t a- Résolution de l'équation de Hamilton Jacobi avec mu=" + str(mu))
            psi = resolution_equation_hamilton.resolution(noeuds_lambda_condition, maillage, maillage0, nombre_iterations, pas_espace, pas_temps, coeff_diffusion_chaud, coeff_diffusion_froid, u, p, tableau_des_normales, mu)

            # Mise à jour du maillage
            print("\t b- Mise à jour du maillage")
            maillage, noeuds_lambda_condition, tableau_des_normales  = setup_resolution_equation.mise_a_jour_maillage(psi)
            
            u = resolution_equation_chaleur.resolution(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, terme_source, False) if not crank_nicolson else resolution_equation_chaleur.resolution_crank_nicolson(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, pas_temps, terme_source, False)

            sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u, nombre_iterations-1, "fig_u"+str(k+1)+"_tf")
            ene = calcul_energie(u, pas_espace, pas_temps, maillage, maillage0)
            vol = volume(pas_espace, maillage)

            print("Energie :", ene)
            if ene <= energie[k]:
                # The step is increased if the energy decreased
                mu = mu * 1.1
                maillage, noeuds_lambda_condition, tableau_des_normales  = setup_resolution_equation.mise_a_jour_maillage(old_psi)
            else:
                maillage, noeuds_lambda_condition, tableau_des_normales  = setup_resolution_equation.mise_a_jour_maillage(old_psi)
                mu = mu / 2
            

        old_psi = psi
        k += 1
        energie[k] = ene
        volumes[k] = vol
        print("Energie trouvée :", ene)
        print("Volume actuel :", vol)
        print("Sauvegarde de l'itération en cours (...)")
        sauvegarde_iteration.sauvegarde(u, maillage, vol, ene, "save_iteration_"+str(k)+".txt", k)
        print("Sauvegarde terminée")
        
        if ene == energie[k-1]:
            print("Energie finale :", ene)
            print("Volume final :", vol)
            break
       
    return maillage0, u0, maillage, u, energie, volumes


def volume(pas_espace, maillage):

    return (numpy.sum(maillage))*(pas_espace**2)
    


def calcul_energie(u, pas_espace, pas_temps, maillage, maillage0): 
    """
    La fonction qui permet de calculer l'énergie pour le problème d'optimisation
    
    :param u: La fonction de température
    :param pas_espace: Le pas de discrétisation en espace
    :param pas_temps: Le pas de discrétisation en temps

    :return La valeur de la fonction d'énergie:

    """

    return calcul_normes.normeL2(u, pas_espace, pas_temps, maillage)

if __name__ == '__main__':

    set_num_threads(16)

    N=5*30
    coeff_diffusion_chaud = 2
    coeff_diffusion_froid = 1
    nombre_iterations = 50
    multiplicateur_pas_temps = 0.2
    pas_temps = 1e-4
    crank_nicolson = True
    start_from_file = False
    file_name = "save_iteration_20.txt"
    start_iter = 0
    energies = []
    volumes = []

    maillage, noeuds_lambda_condition, tableau_des_normales = creation_maillage_prefractal.creation_maillage_koch(N)

    if start_from_file:
        maillage = numpy.zeros((N, N))

        print("Lecture du fichier...")
        with open(file_name, 'r') as file:
            # Read each line in the file
            for line in file:
                args = line.strip().split(",")
                if args[0].lower() == "iteration":
                    start_iter = int(args[1])+1
                    energies = numpy.zeros(start_iter)
                    volumes = numpy.zeros(start_iter)
                if args[0].lower() == "energie":
                    k = int(args[1])
                    val = float(args[2])
                    energies[k] = val
                if args[0].lower() == "volume":
                    k = int(args[1])
                    val = float(args[2])
                    volumes[k] = val
                if args[0].lower() == "maillage":
                    x = int(args[1])
                    y = int(args[2])
                    value = float(args[3])
                    maillage[x,y] = value


        noeuds_lambda_condition, tableau_des_normales = setup_resolution_equation.trouver_normales(maillage)
        print("Lecture terminée... Commence à l'itération", start_iter)


    terme_source = numpy.zeros((nombre_iterations,N,N))

    maillage0, u0, maillage, u, energies, volumes = optimisation(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, pas_temps, terme_source, tableau_des_normales, crank_nicolson, start_iter, energies, volumes)
    
    sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u0,  nombre_iterations-1, "fig_u0_tf")
    sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u, nombre_iterations-1, "fig_u_final_tf")
    sauvegarde_graphiques.sauvegarde_plot_energie(energies)
    sauvegarde_graphiques.sauvegarde_plot_volume(volumes)