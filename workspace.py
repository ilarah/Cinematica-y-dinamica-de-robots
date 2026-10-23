"""
workspace.py
Herramientas para analizar el espacio de trabajo de un robot (rtb.DHRobot),
validar su cinemática inversa y comparar trayectorias ctraj + IK contra jtraj.
"""

import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3


# =====================================================================
# 1. ESPACIO DE TRABAJO
# =====================================================================

def _validar_limites(robot):
    lo, hi = robot.qlim
    if np.any(hi <= lo):
        raise ValueError("Define qlim en todas las juntas "
                         "(rad en revolutas, unidades de longitud en prismáticas).")


def espacio_trabajo(robot, N=20000, devolver_q=False):
    """Muestrea N configuraciones dentro de qlim y devuelve las posiciones del efector."""
    _validar_limites(robot)
    Q = np.random.uniform(robot.qlim[0], robot.qlim[1], size=(N, robot.n))
    P = robot.fkine(Q).t
    return (Q, P) if devolver_q else P


def manipulabilidad(robot, Q, P=None):
    """Índice de Yoshikawa sobre la parte lineal del Jacobiano (sirve en 3D y planares)."""
    if P is None:
        P = robot.fkine(Q).t
    filas = [k for k in range(3) if np.ptp(P[:, k]) > 1e-9]
    w = np.empty(len(Q))
    for i, q in enumerate(Q):
        J = robot.jacob0(q)[filas, :]
        w[i] = np.sqrt(max(np.linalg.det(J @ J.T), 0.0))
    return w


# =====================================================================
# 2. TRAYECTORIAS Y COMPARATIVA ctraj vs jtraj
# =====================================================================

def poligono(vertices, pasos=15, cerrar=True):
    """Une un ctraj por lado. Devuelve los puntos (M,3) y los índices de los vértices."""
    v = [np.asarray(p, float) for p in vertices]
    if cerrar:
        v.append(v[0])
    pts = [rtb.ctraj(SE3(*A), SE3(*B), pasos).t for A, B in zip(v[:-1], v[1:])]
    pts = np.vstack([pts[0]] + [t[1:] for t in pts[1:]])
    idx = np.arange(len(v)) * (pasos - 1)       # posición de cada vértice en pts
    return pts, idx


def jtraj_vertices(qv, pasos=15):
    """jtraj entre q de vértices consecutivos. Misma longitud que la salida de poligono()."""
    return np.vstack([rtb.jtraj(qv[i], qv[i+1], pasos).q[(0 if i == 0 else 1):]
                      for i in range(len(qv) - 1)])


def comparar(puntos, Pj, idx, P=None, qs=None, qj=None,
             titulo='ctraj + IK  vs  jtraj entre vértices'):
    """Panel 2x2: 3D (sobre la nube si se pasa P), planta, alzado y articulaciones.
    Solo para robots 3D (en planares usar mostrar)."""
    fig = plt.figure('Espacio de trabajo')
    fig.clf()                                   # misma ventana de siempre
    fig.set_size_inches(12, 9)

    # 3D, sobre la nube del espacio de trabajo si se la pasas
    ax3 = fig.add_subplot(2, 2, 1, projection='3d')
    if P is not None:
        ax3.scatter(*P[::5].T, s=1, alpha=0.05, color='gray')
    ax3.plot(*puntos.T, 'b', label='ctraj + IK')
    ax3.plot(*Pj.T, 'r--', label='jtraj')
    ax3.plot(*puntos[idx].T, 'ko', label='vértices')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    base = P if P is not None else puntos
    ax3.set_box_aspect(np.maximum(np.ptp(base, axis=0), 1e-3))   # evita proporción 0
    ax3.set_title('Vista 3D'); ax3.legend(fontsize=8)

    # Planta y alzado
    for pos, (i, j, tit) in zip((2, 3), [(0, 1, 'Planta (XY)'), (0, 2, 'Alzado (XZ)')]):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(puntos[:, i], puntos[:, j], 'b')
        ax.plot(Pj[:, i], Pj[:, j], 'r--')
        ax.plot(puntos[idx, i], puntos[idx, j], 'ko')
        ax.set_title(tit); ax.set_aspect('equal'); ax.grid(True)

    # Articulaciones contra el paso (solo si se pasan qs y qj)
    if qs is not None and qj is not None:
        ax = fig.add_subplot(2, 2, 4)
        for j in range(qs.shape[1]):
            ax.plot(np.degrees(qs[:, j]), '-',  color=f'C{j}', label=f'q{j+1} ctraj')
            ax.plot(np.degrees(qj[:, j]), '--', color=f'C{j}', label=f'q{j+1} jtraj')
        ax.set_xlabel('paso'); ax.set_ylabel('grados'); ax.grid(True)
        ax.set_title('Articulaciones'); ax.legend(fontsize=6, ncol=2)

    fig.suptitle(titulo)
    plt.show()
    print("Desviación máxima de jtraj:", np.linalg.norm(Pj - puntos, axis=1).max().round(3))


def comparar_jtraj(robot, puntos, qs, valido, idx, P=None):
    """Valida los vértices, arma jtraj entre ellos y grafica la comparativa.
    Devuelve (qj, Pj), o (None, None) si algún vértice no es válido."""
    if not valido[idx].all():
        malos = np.where(~valido[idx])[0] + 1
        print(f"Vértices no válidos: {list(malos)}. Ajusta la figura.")
        return None, None

    pasos = idx[1] - idx[0] + 1                 # el mismo que usó poligono()
    qj = jtraj_vertices(qs[idx], pasos)
    Pj = robot.fkine(qj).t
    comparar(puntos, Pj, idx, P=P, qs=qs, qj=qj)
    return qj, Pj


# =====================================================================
# 3. VALIDACIÓN DE LA CINEMÁTICA INVERSA
# =====================================================================

def validar_ik(robot, qs, objetivo, tol=1e-4):
    """Revisa cada q de la IK: (a) llega al punto pedido, (b) respeta qlim.
    Devuelve (qs normalizados, valido, colores)  verde = correcta, rojo = no."""
    qs = np.atleast_2d(qs).copy()
    objetivo = np.atleast_2d(objetivo)
    rev = [l.isrevolute for l in robot.links]
    qs[:, rev] = np.angle(np.exp(1j * qs[:, rev]))      # (-π, π] solo en revolutas

    con_q = np.isfinite(qs).all(axis=1)                  # filas donde la IK dio algo
    llega = np.zeros(len(qs), dtype=bool)
    if con_q.any():                                      # evita fkine sobre arreglo vacío
        llega[con_q] = np.linalg.norm(
            robot.fkine(qs[con_q]).t - objetivo[con_q], axis=1) < tol

    lo, hi = robot.qlim
    dentro = np.all((qs >= lo - 1e-9) & (qs <= hi + 1e-9), axis=1)   # NaN da False

    valido = llega & dentro
    colores = np.where(valido, 'g', 'r')

    print(f"{valido.sum()} de {len(qs)} puntos correctos")
    for k in np.where(~valido)[0][:1]:
        motivo = ("sin solución de IK" if not con_q[k]
                  else "no llega al punto" if not llega[k] else "viola qlim")
        print(f"Primer problema: punto {k+1} ({motivo})")
    return qs, valido, colores


# ---- Opcionales: validación con tres colores y margen a los límites ----

def clasificar(robot, qs, ok):
    """Normaliza ángulos y clasifica cada punto de la trayectoria.
    Devuelve (qs normalizados, valido, colores):
      verde = IK convergió y respeta qlim
      naranja = IK convergió pero viola qlim
      rojo = sin solución (fuera del espacio de trabajo)"""
    qs = qs.copy()
    rev = np.array([l.isrevolute for l in robot.links])
    qs[:, rev] = np.angle(np.exp(1j * qs[:, rev]))     # (-π, π], solo revolutas

    tol = 1e-9
    dentro = np.all((qs >= robot.qlim[0] - tol) & (qs <= robot.qlim[1] + tol), axis=1)
    valido = ok & dentro
    colores = np.where(valido, 'g', np.where(ok, 'orange', 'r'))

    print(f"{valido.sum()} de {len(qs)} puntos realizables")
    for k in np.where(~valido)[0][:1]:                 # primer punto problemático
        motivo = "sin solución de IK" if not ok[k] else "fuera de qlim"
        print(f"Primer punto con problema: {k+1} ({motivo})")
    return qs, valido, colores


def margen_limites(robot, Q):
    """Margen normalizado a los límites, por punto (la peor junta manda).
       1 = centro del rango | 0 = justo en el límite | <0 = fuera de qlim"""
    Q = np.atleast_2d(Q)
    lo, hi = robot.qlim
    m = np.minimum(Q - lo, hi - Q) / ((hi - lo) / 2)
    return m.min(axis=1)          # NaN se propaga: los puntos sin IK quedan en NaN


def colores_limites(m, umbral=0.1):
    """rojo = sin solución (NaN) | naranja = viola qlim
       amarillo = a menos del 10% del límite | verde = holgado"""
    return np.where(np.isnan(m), 'r',
           np.where(m < 0, 'orange',
           np.where(m < umbral, 'gold', 'g')))


# =====================================================================
# 4. GRAFICADO
# =====================================================================

# (eje horizontal, eje vertical, título con el sentido de observación)
_VISTAS = {
    'xy': (0, 1, 'Planta (XY) - vista desde +Z'),
    'xz': (0, 2, 'Alzado (XZ) - vista desde -Y'),
    'yz': (1, 2, 'Perfil (YZ) - vista desde +X'),
}
_PLANO = {(0, 1): 'xy', (0, 2): 'xz', (1, 2): 'yz'}


def _pts_robot(robot, q):
    return np.array([robot.base.t] + [T.t for T in robot.fkine_all(q)]
                    + [robot.fkine(q).t])


def _dibujar(ax, P, c, vista, robot, q, puntos, col_pts, s, alpha, elev, azim):
    if vista == '3d':
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=c, cmap='viridis', s=s, alpha=alpha)
        if robot is not None and q is not None:
            ax.plot(*_pts_robot(robot, q).T, '-o', color='k', lw=3, zorder=10)
        if puntos is not None:
            ax.scatter(*puntos.T, color=col_pts, s=40, zorder=11)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect(np.maximum(np.ptp(P, axis=0), 1e-3))   # evita proporción 0
        ax.view_init(elev=elev, azim=azim)
        ax.set_title('Vista 3D')
    else:
        i, j, tit = _VISTAS[vista]
        sc = ax.scatter(P[:, i], P[:, j], c=c, cmap='viridis', s=s, alpha=alpha)
        if robot is not None and q is not None:
            pts = _pts_robot(robot, q)
            ax.plot(pts[:, i], pts[:, j], '-o', color='k', lw=3, zorder=10)
        if puntos is not None:
            ax.scatter(puntos[:, i], puntos[:, j], color=col_pts, s=40, zorder=11)
        ax.set_xlabel('XYZ'[i]); ax.set_ylabel('XYZ'[j])
        ax.set_aspect('equal'); ax.grid(True)
        ax.set_title(tit)
    return sc


def mostrar(P, vista='3d', c=None, etiqueta=None, robot=None, q=None,
            puntos=None, colores_puntos='r', s=1, alpha=0.3,
            titulo='Espacio de trabajo', elev=25, azim=-60):
    """
    vista: '3d' | 'xy' (planta) | 'xz' (alzado) | 'yz' (perfil) | 'todas'
    robot, q: si se dan, se dibuja el brazo en esa configuración (proyectado en cada vista)
    puntos: arreglo (M,3) con puntos objetivo / trayectoria
    colores_puntos: un color o una lista de M colores
    Reutiliza una única figura: cada llamada reemplaza la anterior.
    """
    P = np.asarray(P)
    puntos = None if puntos is None else np.atleast_2d(puntos)
    var = tuple(k for k in range(3) if np.ptp(P[:, k]) > 1e-9)
    if c is None:
        c = P[:, var[-1]]

    fig = plt.figure('Espacio de trabajo')
    fig.clf()

    if len(var) == 2:                       # robot planar: solo existe un plano
        vista = _PLANO[var]
    vista = vista.lower()
    args = (P, c, None, robot, q, puntos, colores_puntos, s, alpha, elev, azim)

    if vista == 'todas':
        fig.set_size_inches(12, 10)
        axs = [fig.add_subplot(2, 2, 1, projection='3d')] + \
              [fig.add_subplot(2, 2, k) for k in (2, 3, 4)]
        for ax, v in zip(axs, ['3d', 'xy', 'xz', 'yz']):
            sc = _dibujar(ax, *args[:2], v, *args[3:])
    else:
        fig.set_size_inches(8, 7)
        ax = fig.add_subplot(111, projection='3d' if vista == '3d' else None)
        sc = _dibujar(ax, *args[:2], vista, *args[3:])
        axs = [ax]

    if etiqueta:
        fig.colorbar(sc, ax=axs, label=etiqueta, shrink=0.7)
    fig.suptitle(titulo)
    plt.show()
    return fig