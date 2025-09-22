
def sauvegarde(u, maillage, volume, energie, file_name, iteration_opti):

    lines_to_write = []

    # Sauvegarde numéro itération
    lines_to_write.append("iteration,"+str(iteration_opti) + "\n")

    # Sauvegarde energie
    lines_to_write.append("energie,"+str(iteration_opti)+","+str(energie) + "\n")
    # Sauvegarde volume
    lines_to_write.append("volume,"+str(iteration_opti)+","+str(volume) + "\n")

    # Sauvegarde de la solution
    nombre_iterations, N, M = u.shape
    for k in range(nombre_iterations):
        for i in range(N):
            for j in range(M):
                if u[k,i,j] != 0: lines_to_write.append("u," + str(k) + "," + str(i) + "," + str(j) + "," + str(u[k,i,j]) + "\n")

    # Sauvegarde du maillage
    N, M = maillage.shape
    for i in range(N):
        for j in range(M):
            if maillage[i,j] != 0: lines_to_write.append("maillage," + str(i) + "," + str(j) + "," + str(maillage[i,j]) + "\n")

    
    with open(file_name, "w") as fp:
        fp.writelines(lines_to_write)
        fp.close()

