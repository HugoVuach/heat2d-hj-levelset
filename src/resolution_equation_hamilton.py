import numpy
import setup_resolution_equation
import sauvegarde_graphiques
import _env
from numba import jit, njit, prange
from scipy.ndimage import label

@njit(parallel=True)
def initialisation_psi(maillage, noeuds_lambda_condition):
    """
    La fonction qui permet d'initialiser la fonction psi
    
    :param maillage: La matrice du maillage
    :param noeuds_lambda_condition: Les noeuds qui possède une condition au bord avec lambda
    
    :return psi_0 définit comme il faut:

    """    
    
    N = maillage.shape[0]//5
    psi = numpy.zeros((N,N))

    for i in prange(N):
        for j in prange(N):
            psi[i, j] = distance_au_bord(maillage, noeuds_lambda_condition, 5*i+1, 5*j+1) # On décale de 1 pour ne pas coïncider avec le maillage plus gros (sinon dès qu'on a une partie qui fait 5x5 alors on ne détecte plus correctement le bord)

    return psi

@jit
def distance_au_bord(maillage, noeuds_lambda_condition, x, y):
    """
    La fonction qui permet de calculer la distance (algébrique) entre un noeud et le bord de Omega_+
    
    :param maillage: La matrice du maillage
    :param noeuds_lambda_condition: Les noeuds qui possède une condition au bord avec lambda
    :param x,y: Les coordonnées du noeud d'intérêt
    
    :return dist(noeud, bord):

    """
    distances_noeud_bord = []

    for (w,z) in noeuds_lambda_condition:
        dist = numpy.sqrt((x-w)**2 + (y-z)**2)
        distances_noeud_bord.append(dist if maillage[x,y] == 0 else -dist)

    # Il faut prendre en compte que si les distances sont < 0, alors la distance au bord est le max de la liste (car le on prend la valeur absolue pour la distance géométrique)
    return min(distances_noeud_bord) if distances_noeud_bord[0] >= 0 else max(distances_noeud_bord)

@njit(parallel=True)
def resolution(noeuds_lambda_condition, maillage, maillage0, nombre_iterations, pas_espace, pas_temps, coeff_diffusion_chaud, coeff_diffusion_froid, u, p, tableau_des_normales, mu):

    M = maillage.shape[0] // 5
    pas_espace_psi = 1/M

    psi = initialisation_psi(maillage, noeuds_lambda_condition)

    #sauvegarde_graphiques.sauvegarde_plot_psi_0(psi, "psi_0")

    # On calcul V dans toute la boite D
    V = numpy.zeros((M,M))
    for i in prange(1, M-1):
        for j in prange(1, M-1):
            x=5*i
            y=5*j

            cote = maillage[x,y]
            coeff_diffusion = coeff_diffusion_chaud if cote==_env.NODE_HOT_SIDE else coeff_diffusion_froid 
            divn = calcul_div_n(i, j, psi, pas_espace_psi)
            V[i,j] = calcul_de_V(x,y, divn, pas_espace, pas_temps, nombre_iterations, coeff_diffusion, cote, u, p, tableau_des_normales, maillage, maillage0)
    
    maxV_abs = numpy.max(numpy.abs(V)) # Le maximum de Vij pour le calcul du pas de temps

    # Le pas de temps nécessaire pour respecter la condition du schéma "upwind"
    pas_temps_psi = 0.4*mu*pas_espace_psi / (numpy.sqrt(2)*maxV_abs)

    psi_new = psi.copy()

    k = 0
    max_iter = 1e6
    print("\t\t i) Résolution de l'équation d'Hamilton Jacobi jusqu'à stationnarité")
    ene = numpy.sum(psi)*(pas_espace_psi**2)
    new_ene = numpy.inf
    while abs(ene-new_ene) > 1e-7 and k < max_iter:
        ene = new_ene
        k += 1
        for i in prange(M//6, 5*M//6):
            for j in prange(M//6, 5*M//6):
                """
                Les différents calculs pour le schéma << upwind >>
                """
                # Calcul de D_ij^(+x) 
                D_ij_px = (psi[i+1,j]-psi[i,j])/pas_espace_psi
                # Calcul de D_ij^(-x) 
                D_ij_mx = (psi[i,j]-psi[i-1,j])/pas_espace_psi
                # Calcul de D_ij^(+y) 
                D_ij_py = (psi[i,j-1]-psi[i,j])/pas_espace_psi
                # Calcul de D_ij^(-y) 
                D_ij_my = (psi[i,j]-psi[i,j+1])/pas_espace_psi
                # Calcul de nabla^+
                nabla_p = numpy.sqrt((max([0, D_ij_mx])**2) + (min([0, D_ij_px])**2) + (max([0, D_ij_my])**2) + (min([0, D_ij_py])**2))
                # Calcul de nabla^-
                nabla_m = numpy.sqrt((max([0, D_ij_px])**2) + (min([0, D_ij_mx])**2) + (max([0, D_ij_py])**2) + (min([0, D_ij_my])**2))

                # Ajout de la pénalisation
                pen = penalisation(maillage0, maillage, psi, pas_espace, pas_espace_psi)
                Vij = V[i,j]+pen

                # Recalcul du pas de temps
                pas_temps_psi = 0.5*mu*pas_espace_psi / (numpy.sqrt(2)*(maxV_abs+abs(pen)))

                psi_new[i, j] = psi[i, j] - pas_temps_psi*(max([0, Vij])*nabla_p + min([0, Vij])*nabla_m)
        
        psi = psi_new.copy()
        new_ene = numpy.sum(psi)*(pas_espace_psi**2)
    print("\t\t ii) Solution stationnaire trouvée")
    return psi_new


    ...

@jit
def detect_holes(phi):
    """
    Détecte les trous dans un maillage carré 2D.
    
    Args:
        phi (numpy.ndarray): Une matrice 2D représentant le domaine (phi < 0 = intérieur).

    Returns:
        int: Le nombre de trous détectés.
        list: Les indices des trous (facultatif, pour diagnostic).
    """
    # Étape 1 : Construire le masque binaire
    mask = (phi <= 0)
    
    # Étape 2 : Labeling des composantes connexes
    labeled_array, num_features = custom_label_CCL(mask)
    
    # Étape 3 : Identifier les trous
    holes = []
    for feature in range(1, num_features + 1):
        # Extraire la composante courante
        component = (labeled_array == feature)
        
        # Vérifier si la composante touche les bords
        touches_border = (
            numpy.any(component[0, :]) or        # Haut
            numpy.any(component[-1, :]) or      # Bas
            numpy.any(component[:, 0]) or       # Gauche
            numpy.any(component[:, -1])         # Droite
        )
        
        if not touches_border:
            # Si elle ne touche pas les bords, c'est un trou
            holes.append(feature)
    
    return len(holes)

@jit
def connexite(phi):
    """
    Détecte les trous dans un maillage carré 2D.
    
    Args:
        phi (numpy.ndarray): Une matrice 2D représentant le domaine (phi < 0 = intérieur).

    Returns:
        int: Le nombre de trous détectés.
        list: Les indices des trous (facultatif, pour diagnostic).
    """
    # Étape 1 : Construire le masque binaire
    mask = (phi == 1)
    
    # Étape 2 : Labeling des composantes connexes
    labeled_array, num_features = custom_label_CCL(mask)
    
    # Étape 3 : Identifier les trous
    connexe = []
    for feature in range(1, num_features + 1):
        # Extraire la composante courante
        connexe.append(feature)
    
    return len(connexe)

@jit
def custom_label(grid):
    """
    Fonction pour détecter et étiqueter les composantes connexes dans une grille 2D binaire.
    Remplace scipy.ndimage.label.

    Args:
        grid (numpy.ndarray): Une grille binaire 2D (0 pour vide, 1 pour le matériau).

    Returns:
        labeled_array (numpy.ndarray): Grille avec des étiquettes uniques pour chaque composante connexe.
        num_features (int): Nombre total de composantes connexes.
    """
    rows, cols = grid.shape
    labeled_array = numpy.zeros((rows, cols))  # Grille étiquetée
    label = 0  # Compteur pour les étiquettes



    # Parcours de la grille pour trouver les composantes connexes
    for x in range(rows):
        for y in range(cols):
            if grid[x, y] == 1 and labeled_array[x, y] == 0:
                label += 1  # Nouvelle composante trouvée

                # Fonction de parcours en profondeur (DFS)
                stack = numpy.array([(x,y)])
                
                while len(stack)>0:
                    i, j = stack[-1]
                    stack = stack[:-1:]

                    if 0 <= i < rows and 0 <= j < cols and grid[i, j] == 1 and labeled_array[i, j] == 0:
                        labeled_array[i, j] = label
                        # Ajouter les 4 voisins (connectivité 4)
                        stack = numpy.concatenate((stack, numpy.array([(i-1, j), (i+1, j), (i, j-1), (i, j+1)])))
                        
    return labeled_array, label

@njit
def find(x, parent):
    # Recherche avec compression de chemin
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

@njit
def union(x, y, parent):
    # Fusionner deux ensembles en utilisant l'union par minimum
    xroot = find(x, parent)
    yroot = find(y, parent)
    if xroot != yroot:
        if xroot < yroot:
            parent[yroot] = xroot
        else:
            parent[xroot] = yroot

@njit
def custom_label_CCL(grid):
    """
    Étiquetage des composantes connexes en deux passes avec union-find.
    
    Arguments :
        grid (numpy.ndarray) : Tableau binaire 2D (1 pour le premier plan, 0 pour l'arrière-plan).
        
    Renvoie :
        labels (numpy.ndarray) : Tableau 2D étiqueté.
        num_features (int) : Nombre total de composantes connexes.
    """
    rows, cols = grid.shape
    labels = numpy.zeros((rows, cols), dtype=numpy.int32)
    
    # Allouer un tableau pour les parents
    parent = numpy.empty(rows * cols + 1, dtype=numpy.int32)
    for i in range(rows * cols + 1):
        parent[i] = i

    next_label = 1

    # Première passe
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                left = labels[i, j - 1] if j > 0 else 0
                top = labels[i - 1, j] if i > 0 else 0
                
                if left == 0 and top == 0:
                    labels[i, j] = next_label
                    next_label += 1
                elif left != 0 and top != 0:
                    # Les deux voisins sont étiquetés : attribuer le label minimum et les unir.
                    min_label = left if left < top else top
                    labels[i, j] = min_label
                    union(left, top, parent)
                else:
                    # Un seul voisin est étiqueté.
                    labels[i, j] = left if left != 0 else top

    # Deuxième passe
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                labels[i, j] = find(labels[i, j], parent)

    # Compter le nombre d'étiquettes utilisées
    used = numpy.zeros(next_label, dtype=numpy.uint8)
    for i in range(rows):
        for j in range(cols):
            if labels[i, j] != 0:
                used[labels[i, j]] = 1

    num_features = 0
    for i in range(1, next_label):
        if used[i] == 1:
            num_features += 1

    return labels, num_features
  
@jit
def calcul_de_V(x, y, divn, pas_espace, pas_temps, nombre_iterations, coeff_diffusion, cote, u, p, tableau_des_normales, maillage, maillage0):
    """
    La fonction qui permet de calculer V
    
    :param x,y: Les coordonnées du noeud d'intérêt
    :param divn: La courbure
    :param pas_espace: Le pas de discretisation en espace /!\ pour le maillage de u
    :param pas_temps: Le pas de discretisation en temps  /!\ pour la discrétisation de u
    :param nombre_iterations: Le nombre d'itération lorsque l'on résout l'équation de la chaleur
    :param coeff_diffusion: La coeff de diffusion bien choisi
    :param cote: Le côté où se trouve le noeud (chaud = 1, froid = 0)
    :param u: La solution du problème de la chaleur
    :param p: La solution du problème adjoint
    :param tableau_des_normales: Les normales de chacune des cellules
    :param maillage: Le maillage courant
    :param maillage0: Le maillage initial
    
    :return V(x,y):

    """
    
    list = [vt(k, x, y, divn, pas_espace, pas_temps, coeff_diffusion, cote, u, p, tableau_des_normales) for k in range(nombre_iterations-1)]
    return pas_temps*sum(list)

@jit
# On définit une sous fonction vt qui a le rôle d'intégrande
def vt(k, x, y, divn, pas_espace, pas_temps, coeff_diffusion, cote, u, p, tableau_des_normales):

        #Dans la dérivée, il y a un moins devant certains termes si on est dans Omega_- et un plus si on est dans Omega_+
        sens = 2*cote - 1

        # A|u+|² - B|u-|², avec A=1, B=1
        up_um = sens*u[k, x, y]**2

        D_gradu_gradp = ((sens * coeff_diffusion) / (4*(pas_espace**2))) * ((u[k, x, y-1] - u[k, x,y+1])*(p[k, x, y-1] - p[k, x,y+1]) + (u[k, x+1, y] + u[k, x-1,y])*(p[k, x+1, y] - p[k, x-1,y]))

        # du/dt * p
        dtup = (sens*(u[k+1, x, y]-u[k, x, y])/(pas_temps))*p[k, x, y]

        # Normale sortante au bord (par convention on a nx,ny = 0,0 si le noeud n'est pas sur le bord)
        nx,ny = tableau_des_normales[x][y]

        #H (lambda *(u- - u+)(p- - p+))
        H_lambda_diffu_diffp = divn*setup_resolution_equation.lambda_condition(x,y)*(u[k,x+nx,y+ny]-u[k,x,y])*(p[k,x+nx,y+ny]-p[k,x,y])

        # Le coeff en d/dn
        dn_lambda = (nx/(2*pas_espace))*(setup_resolution_equation.lambda_condition(x+1,y)*(u[k,x+nx+1,y+ny] - u[k,x+1,y])*(p[k,x+nx+1,y+ny] - p[k,x+1,y]) - setup_resolution_equation.lambda_condition(x-1,y)*(u[k,x+nx-1,y+ny] - u[k,x-1,y])*(p[k,x+nx-1,y+ny] - p[k,x-1,y])) + (ny/(2*pas_espace))*(setup_resolution_equation.lambda_condition(x,y-1)*(u[k,x+nx,y+ny-1]-u[k, x, y-1])*(p[k,x+nx,y+ny-1]-p[k, x, y-1]) - setup_resolution_equation.lambda_condition(x,y+1)*(u[k,x+nx,y+ny+1]-u[k, x, y+1])*(p[k,x+nx,y+ny+1]-p[k, x, y+1]))

        # On a un signe - devant up_um car on minimise -J
        return up_um + D_gradu_gradp + dtup + H_lambda_diffu_diffp + dn_lambda
    
@jit
def penalisation(maillage0, maillage, psi, pas_espace, pas_espace_psi):
    
    new_maillage = numpy.where(psi <= 0, 1, 0)

    nbr_trous = detect_holes(new_maillage)
    composantes_connexes = connexite(new_maillage)


    vol = numpy.sum(new_maillage)*(pas_espace_psi**2)

    return -(2e3)*(vol-numpy.sum(maillage0)*(pas_espace**2)) - 2e3*nbr_trous*(pas_espace_psi**2) - 2e3*(abs(composantes_connexes-1))

@jit
def calcul_div_n(x,y,psi, pas_espace_psi):
    
    dx = (psi[x+1, y]-psi[x, y])/(pas_espace_psi)
    dy = (psi[x, y]-psi[x, y+1])/(pas_espace_psi)

    if (dx == 0): dx = (psi[x+1, y]-psi[x-1, y])/(2*pas_espace_psi)
    if (dy == 0): dy = (psi[x, y-1]-psi[x, y+1])/(pas_espace_psi)

    dxx = (psi[x+1, y]+psi[x-1, y]-2*psi[x,y])/(pas_espace_psi**2)
    dyy = (psi[x, y-1]+psi[x, y+1]-2*psi[x,y])/(pas_espace_psi**2)
    dxy = (psi[x+1,y-1]-psi[x-1,y-1]+psi[x-1,y+1]-psi[x+1,y+1])/(4*(pas_espace_psi**2))

    #print(dx, dy, x, y)
    
    divn = (dyy*(dx**2) - 2*dx*dy*dxy + dxx*(dy**2)) / ((dx**2 + dy**2)**(3/2))

    return divn

