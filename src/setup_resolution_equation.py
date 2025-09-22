import _env
import numpy
from numba import jit

@jit
def lambda_condition(x,y):
    """
    La fonction qui sera utilisé dans la condition au bord
    
    :param x: La coordonnée x
    :param y: La coordonnée y

    :return lambda(x,y):

    """

    return 10#*abs(numpy.sin(x*y))**30

def trouver_normales(mat_maillage):
    N = mat_maillage.shape[0]
    
    noeuds_lambda_condition = []
    tableau_des_normales = numpy.array([N*[(0,0)] for _ in range(N)])

    # On repère les bords
    for x in range(1, N-1):
        for y in range(1, N-1):
            nx,ny = 0,0 # Coordonnées x,y du vecteur normal

            if mat_maillage[x,y-1] != mat_maillage[x,y]:
                ny += 1
                
            if mat_maillage[x,y+1] != mat_maillage[x,y]:
                ny += -1

            if mat_maillage[x+1,y] != mat_maillage[x,y]: 
                nx += 1

            if mat_maillage[x-1,y] != mat_maillage[x,y]:
                nx += -1

            norme = numpy.sqrt(nx ** 2 + ny ** 2) if (nx,ny) != (0,0) else 1
            nx, ny = nx/norme, ny/norme

            tableau_des_normales[x][y] = (nx,ny)
            # Si le noeud a une condition au bord, alors sa normale est non nulle
            if (nx,ny) != (0,0): 
                noeuds_lambda_condition.append((x,y))

    return noeuds_lambda_condition, tableau_des_normales

def creation_maillage_carre(N):
    """
    La fonction qui permet de créer un maillage avec un domaine carré

    :param  N: Permet de créer un maillage NxN

    :return:

    """
    mat_maillage = _env.NODE_COLD_SIDE * numpy.ones((N,N))

    # On définit un carré plus petit où l'on a Omega_+
    for x in range(N//3, 2*N//3, 1):
        for y in range(N//3, 2*N//3, 1):
            mat_maillage[x,y] = _env.NODE_HOT_SIDE
    
    noeuds_lambda_condition, tableau_des_normales = trouver_normales(mat_maillage)

    return mat_maillage, noeuds_lambda_condition, tableau_des_normales



def creation_maillage_circ(N, rayon):
    """
    La fonction qui permet de créer un maillage avec un domaine circulaire

    :param  N: Permet de créer un maillage NxN

    :return:

    """
    mat_maillage = _env.NODE_COLD_SIDE * numpy.ones((N,N))

    # On définit un cercle plus petit où l'on a Omega_+
    for x in range(1, N):
        for y in range(1, N):
            if (numpy.sqrt((x - (N//2))**2 + (y - (N//2))**2) <= rayon):
                mat_maillage[x,y] = _env.NODE_HOT_SIDE
    
    noeuds_lambda_condition, tableau_des_normales = trouver_normales(mat_maillage)

    return mat_maillage, noeuds_lambda_condition, tableau_des_normales

def mise_a_jour_maillage(psi):

    N = 5*psi.shape[0]
    # On choisit un maillage 5x plus grossiers pour psi, donc ici on 5 fait un maillage 5x plus fin
    mat_maillage = numpy.zeros((N,N))

    for x in range(N):
        for y in range(N):

            if (psi[x//5, y//5] <= 0):
                mat_maillage[x,y] = _env.NODE_HOT_SIDE
            else:
                mat_maillage[x,y] = _env.NODE_COLD_SIDE

    noeuds_lambda_condition, tableau_des_normales = trouver_normales(mat_maillage)

    return mat_maillage, noeuds_lambda_condition, tableau_des_normales


