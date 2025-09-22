import numpy

import _env
import resolution_equation_chaleur
import setup_resolution_equation
import sauvegarde_graphiques
import calcul_normes


# Ce code permet de tester la simulation
if __name__ == '__main__':

    coeff_diffusion_chaud = 2
    coeff_diffusion_froid = 1
    nombre_iterations = 25
    multiplicateur_pas_temps = 1

    values = [50,60,70,80,90,100,200,300,400]
    normes_L2_euler = numpy.zeros(len(values))
    normes_L1_euler = numpy.zeros(len(values))
    normes_H1_euler = numpy.zeros(len(values))
    normes_H10_euler = numpy.zeros(len(values))

    normes_L2_cn = numpy.zeros(len(values))
    normes_L1_cn = numpy.zeros(len(values))
    normes_H1_cn = numpy.zeros(len(values))
    normes_H10_cn = numpy.zeros(len(values))

    for k in range(len(values)):

        N = values[k]

        print("Itération:",k)

        maillage, noeuds_lambda_condition, tableau_des_normales = setup_resolution_equation.creation_maillage_carre(N)

        terme_source = numpy.zeros((nombre_iterations,N,N))
        solution_analytique = numpy.zeros((nombre_iterations,N,N))

        
        dx= 1/N
        dt= multiplicateur_pas_temps * (0.4) * ((dx**2) / (2*max([coeff_diffusion_chaud, coeff_diffusion_froid])))
        dy=dx
        for i in range(N):
            for j in range(N):
                for n in range(nombre_iterations):
                    t = n*dt
                    x=i*dx
                    y=j*dy
                    D = coeff_diffusion_chaud if maillage[i,j] == _env.NODE_HOT_SIDE else coeff_diffusion_froid
                    terme_source[n, i, j] = numpy.exp(-t)*(numpy.pi**2 / N**2)*numpy.sin((numpy.pi*x)/(N*dx))*numpy.sin((numpy.pi*y)/(N*dy))*(1-D*((1/dx)**2 + (1/dy)**2))
                    solution_analytique[n, i, j] = numpy.exp(-t)*numpy.sin((numpy.pi*x)/(N*dx))*numpy.sin((numpy.pi*y)/(N*dy))

        maillage, noeuds_lambda_condition, tableau_des_normales = setup_resolution_equation.creation_maillage_carre(N)

        u_euler = resolution_equation_chaleur.resolution(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, terme_source, True)
        u_cn = resolution_equation_chaleur.resolution_crank_nicolson(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, dt, terme_source, True)
        
        sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(u_cn, nombre_iterations-1, "fig_finale_num_"+str(k))
        sauvegarde_graphiques.sauvegarde_plot_temperature_instant_fixe(solution_analytique, nombre_iterations-1, "fig_finale_an_"+str(k))
        sauvegarde_graphiques.sauvegarde_plot_erreur_num_vs_analytique(u_cn, solution_analytique, nombre_iterations-1)

        ecart_euler = u_euler - solution_analytique
        ecart_cn = u_cn - solution_analytique

        normes_L1_euler[k] = calcul_normes.normeL1(ecart_euler, dx, dt, maillage)
        normes_L2_euler[k] = calcul_normes.normeL2(ecart_euler, dx, dt, maillage)
        normes_H1_euler[k] = calcul_normes.normeH1(ecart_euler, dx, dt, maillage)
        normes_H10_euler[k] = calcul_normes.normeH10(ecart_euler, dx, dt, maillage)

        normes_L1_cn[k] = calcul_normes.normeL1(ecart_cn, dx, dt, maillage)
        normes_L2_cn[k] = calcul_normes.normeL2(ecart_cn, dx, dt, maillage)
        normes_H1_cn[k] = calcul_normes.normeH1(ecart_cn, dx, dt, maillage)
        normes_H10_cn[k] = calcul_normes.normeH10(ecart_cn, dx, dt, maillage)
        
    sauvegarde_graphiques.sauvegarde_plot_comparaison_L2(values, normes_L2_euler, normes_L2_cn)
    sauvegarde_graphiques.sauvegarde_plot_comparaison(values, normes_L1_euler, normes_L2_euler, normes_H1_euler, normes_H10_euler, normes_L1_cn, normes_L2_cn, normes_H1_cn, normes_H10_cn)