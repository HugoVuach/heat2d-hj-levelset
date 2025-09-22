import numpy
import matplotlib.pyplot as plt

def sauvegarde_plot_temperature_instant_fixe(U, k, title):

    vmin = 0
    vmax = 1

    plt.matshow(U[k, :, :], cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()

    nom_fichier = title + '.jpg'

    plt.savefig(nom_fichier, dpi=600)
    plt.close()

def sauvegarde_plot_volume(vol):

    plt.title("Volume de $\\Omega_+$ pour chaque itération de la boucle d'optimisation")
    plt.grid()
    plt.ylim(0,0.2)
    plt.plot([_ for _ in range(len(vol))], vol, color="r")

    nom_fichier = "volume.jpg"

    plt.savefig(nom_fichier, dpi=600)
    plt.close()

def sauvegarde_plot_comparaison(values, normeL1_euler, normeL2_euler, normeH1_euler, normeH10_euler, normeL1_cn, normeL2_cn, normeH1_cn, normeH10_cn):

    plt.title("Erreur entre $u_{an}$ et $u_{num}$ pour différentes normes")
    plt.grid()
    plt.plot(values, normeL1_euler, ls="-.", marker="x", label="Explicit $||u_{num}-u_{an}||_{L^1(\\Omega)}$")
    plt.plot(values, normeL2_euler, ls="-.", marker="o", label="Explicit $||u_{num}-u_{an}||_{L^2(\\Omega)}$")
    plt.plot(values, normeH1_euler, ls="-.", marker="+", label="Explicit $||u_{num}-u_{an}||_{H^1(\\Omega)}$")
    plt.plot(values, normeH10_euler, ls="-.", marker="h", label="Explicit $||u_{num}-u_{an}||_{H^1_0(\\Omega)}$")

    plt.plot(values, normeL1_cn, ls="--", marker="x", label="Crank-Nicolson $||u_{num}-u_{an}||_{L^1(\\Omega)}$")
    plt.plot(values, normeL2_cn, ls="--", marker="o", label="Crank-Nicolson $||u_{num}-u_{an}||_{L^2(\\Omega)}$")
    plt.plot(values, normeH1_cn, ls="--", marker="+", label="Crank-Nicolson $||u_{num}-u_{an}||_{H^1(\\Omega)}$")
    plt.plot(values, normeH10_cn, ls="--", marker="h", label="Crank-Nicolson $||u_{num}-u_{an}||_{H^1_0(\\Omega)}$")

    plt.legend()

    plt.xlabel("maillage $N\\times N$")
    plt.ylabel("Erreur pour chaque norme")

    nom_fichier = "comparaison.jpg"

    plt.savefig(nom_fichier, dpi=600)
    plt.close()

def sauvegarde_plot_comparaison_L2(values, normeL2_euler, normeL2_cn):

    plt.title("Erreur entre $u_{an}$ et $u_{num}$ pour la norme $L^2$")
    plt.grid()
    plt.plot(values, normeL2_euler, ls="-.", marker="o", label="Explicit $||u_{num}-u_{an}||_{L^2(\\Omega)}$")
    plt.plot(values, normeL2_cn, ls="--", marker="o", label="Crank-Nicolson $||u_{num}-u_{an}||_{L^2(\\Omega)}$")

    plt.legend()

    plt.xlabel("maillage $N\\times N$")
    plt.ylabel("Erreur pour la norme $L^2$")

    nom_fichier = "comparaison_L2.jpg"

    plt.savefig(nom_fichier, dpi=600)
    plt.close()

def sauvegarde_plot_erreur_num_vs_analytique(u_numerique, u_analytique, k):

    plt.title("$|u_{num}-u_{analytique}|$")

    plt.matshow(numpy.abs(u_numerique[k, :, :]-u_analytique[k, :, :]), cmap='viridis')
    plt.colorbar()

    nom_fichier = 'erreur.jpg'

    plt.savefig(nom_fichier, dpi=600)
    plt.close()

    

def sauvegarde_plot_energie(ene):

    plt.title("Norme $L^2$ (énergie) de $u(\\Omega)$ pour chaque itération de la boucle d'optimisation")
    plt.grid()
    
    plt.plot([_ for _ in range(len(ene))], ene, color="r")

    nom_fichier = "energie.jpg"

    plt.savefig(nom_fichier, dpi=600)
    plt.close()


def sauvegarde_plot_psi_0(psi, title):

    vmin = -10
    vmax = +10

    plt.matshow(psi, vmin=vmin, vmax=vmax, cmap="Spectral")
    plt.colorbar()

    nom_fichier = title + '.jpg'

    plt.savefig(nom_fichier, dpi=600)
    plt.close()