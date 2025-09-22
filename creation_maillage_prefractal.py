import _env
import numpy as np
import math
import setup_resolution_equation

def bresenham(x0, y0, x1, y1):
    """Algorithme de Bresenham pour tracer une ligne entre deux points."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def koch_segment(x0, y0, x1, y1, depth):
    """Retourne les points du segment de Koch récursif."""
    if depth == 0:
        return [(x0, y0), (x1, y1)]
    else:
        dx = x1 - x0
        dy = y1 - y0
        x2 = x0 + dx / 3
        y2 = y0 + dy / 3
        x3 = x0 + dx * 0.5 - math.sqrt(3) * dy / 6
        y3 = y0 + dy * 0.5 + math.sqrt(3) * dx / 6
        x4 = x0 + 2 * dx / 3
        y4 = y0 + 2 * dy / 3

        return (
            koch_segment(x0, y0, x2, y2, depth - 1)[:-1] +
            koch_segment(x2, y2, x3, y3, depth - 1)[:-1] +
            koch_segment(x3, y3, x4, y4, depth - 1)[:-1] +
            koch_segment(x4, y4, x1, y1, depth - 1)
        )

def draw_koch_snowflake(matrix_size=500, depth=3):
    """Crée une matrice avec un flocon de Koch."""
    mat = np.zeros((matrix_size, matrix_size), dtype=int)

    # Triangle initial
    size = matrix_size * 0.6
    cx, cy = matrix_size // 2, matrix_size // 2
    angles = [0, 120, 240]
    points = []
    for a in angles:
        rad = math.radians(a)
        x = cx + size * math.cos(rad) / 2
        y = cy + size * math.sin(rad) / 2
        points.append((x, y))

    for i in range(3):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % 3]
        koch_points = koch_segment(x0, y0, x1, y1, depth)
        for (x0, y0), (x1, y1) in zip(koch_points, koch_points[1:]):
            for x, y in bresenham(int(x0), int(y0), int(x1), int(y1)):
                if 0 <= x < matrix_size and 0 <= y < matrix_size:
                    mat[y, x] = 1
    return mat


import numpy as np
from collections import deque

def fill_inside(mat):
    h, w = mat.shape
    visited = np.zeros_like(mat, dtype=bool)
    
    # Commencer le flood fill à partir de tous les bords
    queue = deque()

    # Ajouter les pixels de bord de la matrice
    for x in range(w):
        if mat[0, x] == 0:
            queue.append((0, x))
        if mat[h - 1, x] == 0:
            queue.append((h - 1, x))
    for y in range(h):
        if mat[y, 0] == 0:
            queue.append((y, 0))
        if mat[y, w - 1] == 0:
            queue.append((y, w - 1))

    # Flood fill pour marquer l'extérieur
    while queue:
        y, x = queue.popleft()
        if visited[y, x] or mat[y, x] == 1:
            continue
        visited[y, x] = True
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mat[ny, nx] == 0:
                queue.append((ny, nx))

    # Tous les 0 non visités sont à l'intérieur du flocon
    filled_mat = mat.copy()
    filled_mat[(~visited) & (mat == 0)] = 1
    return filled_mat

def scale_maillage(maillage, factor=5):
    M = factor * maillage.shape[0]
    maillage_scaled = np.zeros((M,M))
    for x in range(M):
        for y in range(M):
            maillage_scaled[x,y] = maillage[x//5, y//5]

    return maillage_scaled

def creation_maillage_koch(N):
    """
    La fonction qui permet de créer un maillage avec un domaine en flocon

    :param  N: Permet de créer un maillage NxN

    :return:

    """
    mat = draw_koch_snowflake(N//5, depth=2)  ##creuse
    mat_maillage = scale_maillage(fill_inside(mat), 5)
    
    noeuds_lambda_condition, tableau_des_normales = setup_resolution_equation.trouver_normales(mat_maillage)

    return mat_maillage, noeuds_lambda_condition, tableau_des_normales