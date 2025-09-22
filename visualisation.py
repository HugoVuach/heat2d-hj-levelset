import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def charger_sauvegarde(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    u_data = []
    max_k = -1
    shape = None

    for line in lines:
        parts = line.strip().split(',')
        if parts[0] == 'u':
            k, i, j, val = map(float, parts[1:])
            k, i, j = int(k), int(i), int(j)
            while len(u_data) <= k:
                u_data.append({})
            u_data[k][(i, j)] = val
            if shape is None or i+1 > shape[0] or j+1 > shape[1]:
                shape = (i+1, j+1)

    u_array = np.zeros((len(u_data), *shape))
    for k, data in enumerate(u_data):
        for (i, j), val in data.items():
            u_array[k, i, j] = val

    return u_array

def animer_diffusion(fichiers):
    all_u = [charger_sauvegarde(f) for f in fichiers]
    fig, axes = plt.subplots(1, len(all_u), figsize=(6 * len(all_u), 5))

    if len(all_u) == 1:
        axes = [axes]

    ims = []
    for t in range(all_u[0].shape[0]):
        frames = []
        for ax, u in zip(axes, all_u):
            im = ax.imshow(u[t], cmap='hot', vmin=0, vmax=1, animated=True)
            frames.append(im)
        ims.append(frames)

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def lire_sauvegarde(file_name, N):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    u_data = []
    maillage = np.zeros((N,N))
    volume = 0
    for line in lines:
        parts = line.strip().split(',')
        if parts[0] == 'u':
            k, i, j, val = map(float, parts[1:])
            k, i, j = int(k), int(i), int(j)
            while len(u_data) <= k:
                u_data.append({})
            u_data[k][(i, j)] = val
        elif parts[0] == 'maillage':
            i, j, val = int(parts[1]), int(parts[2]), float(parts[3])
            maillage[i, j] = val
        elif parts[0] == 'volume':
            volume = float(parts[2])

    N_t = []
    for k, data in enumerate(u_data):
        u_mat = np.zeros(maillage.shape)
        for (i, j), val in data.items():
            u_mat[i, j] = val
        u_plus = np.where(maillage == 0, u_mat, 0)
        integral = np.sum(u_plus)*(1/N**2)
        N_t.append(integral)

    N_t = np.array(N_t)
    if (N_t[0] != 0):
        N_t -= N_t[0]
    return N_t

def tracer_N_t(fichiers, N, coeff_diffusion_chaud, pas_temps):
    plt.figure(figsize=(10, 6))
    max_t = 0
    for file in fichiers:
        N_t = lire_sauvegarde(file, N)
        max_t = max(max_t, len(N_t))
        plt.plot(N_t, label=file)

    """
    # Courbes théoriques en pointillés
    T = np.arange(max_t)
    t = T*pas_temps
    D = coeff_diffusion_chaud  # Constante arbitraire (ajustable selon besoin)
    n = 3
    plt.plot(T, (D * t)**((n - 2)/2), '--', color='black', label=r'$(D_+t)^{n-d/2}$ (d=2)')
    plt.plot(T, (D * t)**((n - 2.9)/2), '--', color='black', label=r'$(D_+t)^{n-d/2}$ (d=2.9)')
    """

    plt.xlabel("Temps (itérations de la chaleur)")
    plt.ylabel("$N(t)$")
    plt.title("Comparaison de la vitesse de diffusion")
    plt.legend()
    plt.grid(True)
    plt.show()

animer_diffusion(["250_cool/save_iteration_1.txt", "250_cool/save_iteration_20.txt"])
#animer_diffusion(["save_iteration_1.txt", "250_cool/save_iteration_20.txt"])
#tracer_N_t(["250_cool/save_iteration_1.txt", "250_cool/save_iteration_20.txt"], N=250, coeff_diffusion_chaud=1, pas_temps=1e-4)
