import numpy

def normeL2(u, pas_espace, pas_temps, maillage): 
    """
    La fonction qui permet de calculer la norme L2 de u
    
    :param u: La fonction de température
    :param pas_espace: Le pas de discrétisation en espace
    :param pas_temps: Le pas de discrétisation en temps

    :return La norme L2:

    """
    iteration_max = u.shape[0]
    jt = numpy.zeros((1, iteration_max))

    for k in range(iteration_max):
        jt[0, k] = numpy.sum(u**2)*(pas_espace**2)

    norme_L2_integree = numpy.sqrt(numpy.sum(jt)*(pas_temps))

    return norme_L2_integree

def normeL1(u, pas_espace, pas_temps, maillage): 
    """
    La fonction qui permet de calculer la norme L1 de u
    
    :param u: La fonction de température
    :param pas_espace: Le pas de discrétisation en espace
    :param pas_temps: Le pas de discrétisation en temps

    :return La norme L1:

    """
    iteration_max = u.shape[0]
    jt = numpy.zeros((1, iteration_max))

    for k in range(iteration_max):
        jt[0, k] = numpy.sum(abs(u))*(pas_espace**2)

    norme_L1_integree = numpy.sum(jt)*(pas_temps)

    return norme_L1_integree

def normeH10(u, pas_espace, pas_temps, maillage): 
    """
    La fonction qui permet de calculer la norme H10 de u
    
    :param u: La fonction de température
    :param pas_espace: Le pas de discrétisation en espace
    :param pas_temps: Le pas de discrétisation en temps

    :return La norme H10 (norme L2 de grad u):

    """
    iteration_max = u.shape[0]
    jt = numpy.zeros((1, iteration_max))

    dxu = (u[:, 1::, :-1:]-u[:, 0:-1:, :-1:])*pas_espace
    dyu = (u[:, :-1:, 1::]-u[:, :-1:, 0:-1:])*pas_espace

    for k in range(iteration_max):
        jt[0, k] = numpy.sum((dxu)**2 + (dyu)**2)*(pas_espace**2)

    norme_L2_grad_integree = numpy.sum(jt)*(pas_temps)

    return norme_L2_grad_integree

def normeH1(u, pas_espace, pas_temps, maillage): 
    """
    La fonction qui permet de calculer la norme H1 de u
    
    :param u: La fonction de température
    :param pas_espace: Le pas de discrétisation en espace
    :param pas_temps: Le pas de discrétisation en temps

    :return La norme H1:

    """
    return normeL2(u, pas_espace, pas_temps, maillage) + normeH10(u, pas_espace, pas_temps, maillage)