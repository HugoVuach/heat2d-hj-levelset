import numpy
from numba import njit, prange, jit

import _env
import setup_resolution_equation

@njit(parallel=True) 
def resolution(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, multiplicateur_pas_temps, terme_source, essai_convergence):
    """
    La fonction qui permet de résoudre l'équation de la chaleur
    
    :param maillage: La matrice du maillage
    :param coeff_diffusion_chaud: Le coefficient de diffusion dans le milieu cpyhthaud
    :param coeff_diffusion_froid: Le coefficient de diffusion dans le milieu froid
    :param noeuds_lambda_condition: Les noeuds qui possède une condition au bord avec lambda
    :param nombre_iterations: Le nombre d'itérations que l'on souhaite faire
    :param multiplicateur_pas_temps: Toujours garder < 1, permet de réduire le pas de temps
    :param terme_source: Matrice avec le terme source dans l'équation (= 0 pour le problème principal)
    :param essai_convergence: Pour la partie comparaison entre solution numérique et solution analytique

    :return La solution en température U(x,y) sous forme de matrice:

    """


    N = maillage.shape[0]
    U = numpy.zeros((nombre_iterations, N,N)) # Création d'une liste

    
    # Initialisation de U
    for i in range(N):
        for j in range(N):
            U[0, i, j] = maillage[i,j]
    
   

    # Maillage carré donc même pas d'espace selon x ou y
    pas_espace = 1/N

    # Il faut respecter la condition CFL
    pas_temps = multiplicateur_pas_temps * (0.4) * ((pas_espace**2) / (2*max([coeff_diffusion_chaud, coeff_diffusion_froid]))) # On met 0.4 au lieu de 0.5 pour bien être strictement inférieur à dxdy/2max(D+,D-)
    
    # Dans le cas où l'on veut tester la convergence du maillage
    if essai_convergence:
        for i in range(N):
            for j in range(N):
                x=i*pas_espace
                y=j*pas_espace
                U[0, i, j] = numpy.sin((numpy.pi*x)/(N*pas_espace))*numpy.sin((numpy.pi*y)/(N*pas_espace))
    
    
    for k in range(nombre_iterations-1):
        # En utilisant un range(1, N-1), les bords extérieurs du maillages agissent comme des sources froides
        for x in prange(1, N-1):
            for y in prange(1, N-1):
                if (x,y) in noeuds_lambda_condition:
                    U[k+1, x,y] = calcul_avec_condition(maillage, U, k, x, y, coeff_diffusion_chaud, coeff_diffusion_froid, pas_espace, pas_temps, terme_source)
                else:

                    U[k+1, x,y] = calcul_sans_condition(maillage, U, k, x, y, coeff_diffusion_chaud, coeff_diffusion_froid, pas_espace, pas_temps, terme_source)

    return U
            
@jit
def calcul_sans_condition(maillage, U, k, x, y,coeff_diffusion_chaud,coeff_diffusion_froid, pas_espace, pas_temps, terme_source):
    """
    Lorsque le noeud ne possède pas de conditions aux bords,
    
    :param maillage: La matrice du maillage
    :param U: La matrice des températures
    :param k: L'itération actuelle
    :param x: Coordonnée x sur le maillage
    :param y: Coordonnée y sur le maillage
    :param coeff_diffusion_chaud: Le coefficient de diffusion dans le milieu chaud
    :param coeff_diffusion_froid: Le coefficient de diffusion dans le milieu froid
    :param terme_source: Matrice avec le terme source dans l'équation (= 0 pour le problème principal)
    
    :return La solution U(x,y) sous forme de matrice:

    """

    # On choisit le coefficient de diffusion adapté
    coeff_diffusion = coeff_diffusion_chaud if maillage[x,y] == _env.NODE_HOT_SIDE else coeff_diffusion_froid


    #u = U(x,y)+du à l'itération k+1
    du = (((coeff_diffusion / (pas_espace**2)) * (U[k,x,y+1]+U[k,x,y-1]-2*U[k,x,y])) + ((coeff_diffusion / (pas_espace**2)) * (U[k,x+1,y]+U[k,x-1,y]-2*U[k,x,y]) + terme_source[k,x,y])) * pas_temps

    return (U[k,x,y]+du)

@jit
def calcul_avec_condition(maillage, U, k, x, y,coeff_diffusion_chaud,coeff_diffusion_froid, pas_espace, pas_temps, terme_source):
    
    """
    Lorsque le noeud possède une condition aux bords,
    
    :param maillage: La matrice du maillage
    :param U: La matrice des températures
    :param k: L'itération actuelle
    :param x: Coordonnée x sur le maillage
    :param y: Coordonnée y sur le maillage
    :param coeff_diffusion_chaud: Le coefficient de diffusion dans le milieu chaud
    :param coeff_diffusion_froid: Le coefficient de diffusion dans le milieu froid
    :param terme_source: Matrice avec le terme source dans l'équation (= 0 pour le problème principal)
    
    :return La solution U(x,y) sous forme de matrice:
    """

    du = terme_source[k,x,y] # On initialise avec le terme source

    # On choisit le coefficient de diffusion adapté
    coeff_diffusion = coeff_diffusion_chaud if maillage[x,y] == _env.NODE_HOT_SIDE else coeff_diffusion_froid

    # Ensuite, on somme les contributions de chaque case selon s'il est sont du même côté (froid ou chaud) ou pas comme un sorte de puzzle

    # Case au dessus
    if maillage[x,y] != maillage[x, y-1]: 
        du -= (1/ (4 * pas_espace))*(setup_resolution_equation.lambda_condition(x,y-1)+setup_resolution_equation.lambda_condition(x,y))*(U[k,x,y]-U[k,x,y-1])
    else: 
        du -= (coeff_diffusion / pas_espace**2)*(U[k,x,y]-U[k,x,y-1])

    # Case à droite
    if maillage[x,y] != maillage[x+1, y]: 
        du += (1/ (4 * pas_espace))*(setup_resolution_equation.lambda_condition(x+1,y)+setup_resolution_equation.lambda_condition(x,y))*(U[k,x+1,y]-U[k,x,y])
    else: 
        du += (coeff_diffusion / pas_espace**2)*(U[k,x+1,y]-U[k,x,y])

    # Case en dessous
    if maillage[x,y] != maillage[x, y+1]: 
        du += (1/ (4 * pas_espace))*(setup_resolution_equation.lambda_condition(x,y+1)+setup_resolution_equation.lambda_condition(x,y))*(U[k,x,y+1]-U[k,x,y])
    else: 
        du += (coeff_diffusion / pas_espace**2)*(U[k,x,y+1]-U[k,x,y])

    # Case à gauche
    if maillage[x,y] != maillage[x-1, y]: 
        du-= (1/ (4 * pas_espace))*(setup_resolution_equation.lambda_condition(x-1,y)+setup_resolution_equation.lambda_condition(x,y))*(U[k,x,y]-U[k,x-1,y])
    else: 
        du -= (coeff_diffusion / pas_espace**2)*(U[k,x,y]-U[k,x-1,y])     

    du*=pas_temps

    return U[k,x,y]+du


import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve


def resolution_crank_nicolson(maillage, coeff_diffusion_chaud, coeff_diffusion_froid, noeuds_lambda_condition, nombre_iterations, pas_temps, terme_source, essai_convergence):
    """
    La fonction qui permet de résoudre l'équation de la chaleur
    
    :param maillage: La matrice du maillage
    :param coeff_diffusion_chaud: Le coefficient de diffusion dans le milieu cpyhthaud
    :param coeff_diffusion_froid: Le coefficient de diffusion dans le milieu froid
    :param noeuds_lambda_condition: Les noeuds qui possède une condition au bord avec lambda
    :param nombre_iterations: Le nombre d'itérations que l'on souhaite faire
    :param pas_temps: Le pas de temps
    :param terme_source: Matrice avec le terme source dans l'équation (= 0 pour le problème principal)
    :param essai_convergence: Pour la partie comparaison entre solution numérique et solution analytique

    :return La solution en température U(x,y) sous forme de matrice:

    """


    N = maillage.shape[0]
    U = numpy.zeros((nombre_iterations, N**2)) # Création d'une liste
    S = numpy.zeros_like(U)

    # Maillage carré donc même pas d'espace selon x ou y
    pas_espace = 1/N
    
    # Initialisation de U et de S
    for i in prange(N):
        for j in prange(N):
            U[0, i+j*N] = maillage[i,j]

    for k in prange(nombre_iterations-1):
        for i in prange(N):
            for j in prange(N):
                S[k, i+j*N]  = terme_source[k,i,j]+terme_source[k+1,i,j]
    
    # Création des matrices A_1, A_2, A_3, A_4
    A1 = sp.lil_matrix((N**2, N**2))
    A2 = sp.lil_matrix((N**2, N**2))
    A3 = sp.lil_matrix((N**2, N**2))
    A4 = sp.lil_matrix((N**2, N**2))

    for k in prange(N, N*(N-1)):
        i, j = k%N, int(numpy.floor(k/N))
        if (i == 0 or i == N-1): continue

        

        D = coeff_diffusion_chaud if maillage[i,j] == _env.NODE_HOT_SIDE else coeff_diffusion_froid
        # Case en haut
        if (maillage[i,j]!=maillage[i,j-1]):
            lambda_1_k = setup_resolution_equation.lambda_condition(i,j-1)+setup_resolution_equation.lambda_condition(i,j)
            A1[k,k-N] = (1/(4*pas_espace))*lambda_1_k
            A1[k,k]= -(1/(4*pas_espace))*lambda_1_k
        else:
            A1[k,k-N]= D/(pas_espace**2)
            A1[k,k]= -D/(pas_espace**2)
        
        # Case à droite
        if (maillage[i+1,j]!=maillage[i,j]):
            lambda_2_k = setup_resolution_equation.lambda_condition(i+1,j)+setup_resolution_equation.lambda_condition(i,j)
            A2[k,k+1]= (1/(4*pas_espace))*lambda_2_k
            A2[k,k]= -(1/(4*pas_espace))*lambda_2_k
        else:
            A2[k,k+1]= D/(pas_espace**2)
            A2[k,k]= -D/(pas_espace**2)

        # Case en bas
        if (maillage[i,j]!=maillage[i,j+1]):
            lambda_3_k = setup_resolution_equation.lambda_condition(i,j+1)+setup_resolution_equation.lambda_condition(i,j)
            A3[k,k+N] = (1/(4*pas_espace))*lambda_3_k
            A3[k,k]= -(1/(4*pas_espace))*lambda_3_k
        else:
            A3[k,k+N]= D/(pas_espace**2)
            A3[k,k]= -D/(pas_espace**2)

        # Case à gauche
        if (maillage[i-1,j]!=maillage[i,j]):
            lambda_4_k = setup_resolution_equation.lambda_condition(i-1,j)+setup_resolution_equation.lambda_condition(i,j)
            A4[k,k-1]= (1/(4*pas_espace))*lambda_4_k
            A4[k,k]= -(1/(4*pas_espace))*lambda_4_k
        else:
            A4[k,k-1]= D/(pas_espace**2)
            A4[k,k]= -D/(pas_espace**2)

    A1 = A1.tocsr()
    A2 = A2.tocsr()
    A3 = A3.tocsr()
    A4 = A4.tocsr()

    A = A1 + A2 + A3 + A4
    I = sp.eye(N**2, format="csr")  # Matrice identité creuse
    

#################Version qui explose en mémoire ################""    
    # Calcul de l'inverse de la matrice creuse
    # inv_matrix = inv(I - (pas_temps / 2) * A)
    # Produit matriciel entre matrices creuses
    #matrice_rigidite = inv_matrix @ (I + (pas_temps / 2) * A)
    #print(matrice_rigidite)
###############################################################""""""

    M = I - (pas_temps / 2) * A  # Matrice du système
    B = I + (pas_temps / 2) * A  # Matrice de droite

    # Fonction pour résoudre le système M * X = B sans inversion explicite
    def solve_M(B):
        return spsolve(M, B)

    def apply_matrice_rigidite(V):
        return spsolve(M, B @ V)  # Résout M * X = B * V

    # Dans le cas où l'on veut tester la convergence du maillage
    if essai_convergence:
        for i in prange(N):
            for j in prange(N):
                x=i*pas_espace
                y=j*pas_espace
                U[0, i+j*N] = numpy.sin((numpy.pi*x)/(N*pas_espace))*numpy.sin((numpy.pi*y)/(N*pas_espace))
    
    for k in range(nombre_iterations-1):
        U[k+1,::] = apply_matrice_rigidite(U[k, :].reshape(-1, 1)).flatten() + spsolve(M, (pas_temps/2) * S[k, :])
    
    """
    for k in range(nombre_iterations-1):
        U[k+1,::] = numpy.matmul(matrice_rigidite, U[k, ::]) + numpy.matmul(inv_matrix, (pas_temps/2)*S[k, ::])
    """
    
    # On redonne V sous forme d'une carte de température
    V = numpy.zeros((nombre_iterations, N, N))
    for k in range(nombre_iterations):
        for i in prange(N):
            for j in prange(N):
                V[k,i,j] = U[k, i+j*N]
    
    return V