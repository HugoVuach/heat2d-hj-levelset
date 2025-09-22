import numpy

import resolution_equation_chaleur
import setup_resolution_equation
import sauvegarde_graphiques


# Ce code permet de tester la simulation
if __name__ == '__main__':

    N=500
    coeff_diffusion_chaud = 1
    coeff_diffusion_froid = 1
    nombre_iterations = 2
    pas_temps = 0.005
   
    terme_source = numpy.zeros((nombre_iterations,N,N))

    maillage, noeuds_lambda_condition, tableau_des_normales = setup_resolution_equation.creation_maillage_carre(N)

    u = resolution_equation_chaleur.resolution_crank_nicolson(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, pas_temps, terme_source, False)
    #u = resolution_equation_chaleur.resolution(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, 1, terme_source, False)
    
    print(pas_temps)
    sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u, 0, "fig_u0")
    sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u, nombre_iterations-1, "fig_u_final")
