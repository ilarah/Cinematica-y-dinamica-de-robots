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
    pasos = int(pasos)
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

    pasos = int(idx[1] - idx[0] + 1)                 # el mismo que usó poligono()
    qj = jtraj_vertices(qs[idx], pasos)
    Pj = robot.fkine(qj).t
    comparar(puntos, Pj, idx, P=P, qs=qs, qj=qj)
    return qj, Pj


# =====================================================================
# 3. VALIDACIÓN DE LA CINEMÁTICA INVERSA
# =====================================================================

def _ajustar_a_qlim(robot, qs):
    """Lleva cada revoluta al equivalente q+2πk dentro de [lo, hi];
    si no existe ninguno, la deja en (-π, π] (y será marcada como inválida)."""
    qs = qs.copy()
    lo, hi = robot.qlim
    ks = 2 * np.pi * np.array([-1, 0, 1])
    for j, l in enumerate(robot.links):
        if not l.isrevolute:
            continue
        cand = qs[:, j][:, None] + ks                       # (n, 3)
        ok = (cand >= lo[j] - 1e-9) & (cand <= hi[j] + 1e-9)
        elegido = cand[np.arange(len(qs)), np.where(ok, np.abs(cand), np.inf).argmin(axis=1)]
        envuelto = np.angle(np.exp(1j * qs[:, j]))
        qs[:, j] = np.where(ok.any(axis=1), elegido, envuelto)
    return qs


def _evaluar(robot, qs, objetivo, tol):
    """Devuelve (con_q, llega, dentro) para cada fila de qs."""
    con_q = np.isfinite(qs).all(axis=1)
    llega = np.zeros(len(qs), dtype=bool)
    if con_q.any():
        llega[con_q] = np.linalg.norm(
            robot.fkine(qs[con_q]).t - objetivo[con_q], axis=1) < tol
    lo, hi = robot.qlim
    dentro = np.all((qs >= lo - 1e-9) & (qs <= hi + 1e-9), axis=1)   # NaN da False
    return con_q, llega, dentro


def _ik_analitica(robot, p, atras, s):
    """IK analítica del antropomórfico 3 GDL (MDH) para una rama dada.
    atras: q1 + 180°.  s = +1 / -1: signo de q3 (codo)."""
    L1, L2, L3 = robot.links[0].d, robot.links[2].a, robot.tool.t[0]
    x, y, z = p
    q1 = np.arctan2(y, x) + (np.pi if atras else 0.0)
    r = x*np.cos(q1) + y*np.sin(q1)
    zz = z - L1
    c3 = (r**2 + zz**2 - L2**2 - L3**2) / (2*L2*L3)
    if abs(c3) > 1:
        return np.full(3, np.nan)
    q3 = s*np.arccos(c3)
    q2 = np.arctan2(zz, r) - np.arctan2(L3*np.sin(q3), L2 + L3*np.cos(q3))
    return _ajustar_a_qlim(robot, np.array([[q1, q2, q3]]))[0]


def semilla(robot, puntos):
    """q0 analítica para el primer punto, eligiendo la rama (adelante/atrás x codo)
    que mantiene más puntos dentro de qlim. None si ninguna rama sirve."""
    lo, hi = robot.qlim
    mejor, mejor_n = None, 0
    for atras in (False, True):
        for s in (1, -1):
            qs = np.array([_ik_analitica(robot, p, atras, s) for p in puntos])
            ok = np.all((qs >= lo - 1e-9) & (qs <= hi + 1e-9), axis=1)   # NaN da False
            if ok[0] and ok.sum() > mejor_n:
                mejor, mejor_n = qs[0], ok.sum()
    return mejor


def validar_ik(robot, qs, objetivo, tol=1e-4, reparar=True, intentos=30,
               seed=0, cerrar=True):
    """Revisa cada q de la IK: (a) llega al punto pedido, (b) respeta qlim.
    Si reparar=True y la trayectoria no salió completa:
      1) la rehace entera desde una semilla analítica (misma rama en todo el recorrido),
      2) reintenta los puntos que sigan fallando con semillas vecinas y aleatorias (seed).
    cerrar=True: si el último punto coincide con el primero, fuerza qs[-1] = qs[0].
    Devuelve (qs, valido, colores)."""
    qs = _ajustar_a_qlim(robot, np.atleast_2d(qs))
    objetivo = np.atleast_2d(objetivo)

    con_q, llega, dentro = _evaluar(robot, qs, objetivo, tol)
    valido = llega & dentro
    n_inicial = valido.sum()

    if reparar and not valido.all():
        # --- 1) rehacer la trayectoria completa desde la semilla analítica ---
        q0 = semilla(robot, objetivo)
        if q0 is not None:
            qn, q_prev = np.full_like(qs, np.nan), q0
            for k, p in enumerate(objetivo):
                sol = robot.ikine_LM(SE3(*p), mask=[1, 1, 1, 0, 0, 0],
                                     q0=q_prev, joint_limits=True)
                if sol.success:
                    qn[k] = q_prev = sol.q
            qn = _ajustar_a_qlim(robot, qn)
            _, ll, de = _evaluar(robot, qn, objetivo, tol)
            if (ll & de).sum() > valido.sum():          # solo se acepta si mejora
                qs, valido = qn, ll & de

        # --- 2) reparar punto por punto lo que siga fallando ---
        if not valido.all():
            rng = np.random.default_rng(seed)
            lo, hi = robot.qlim
            mask = [1, 1, 1, 0, 0, 0]
            for k in np.where(~valido)[0]:
                vecinos = [qs[j] for j in (k - 1, k + 1)
                           if 0 <= j < len(qs) and valido[j]]
                semillas = vecinos + [rng.uniform(lo, hi) for _ in range(intentos)]
                ref = vecinos[0] if vecinos else None
                mejor, mejor_d = None, np.inf
                for q0 in semillas:
                    sol = robot.ikine_LM(SE3(*objetivo[k]), mask=mask,
                                         q0=q0, joint_limits=True)
                    if not sol.success:
                        continue
                    q = _ajustar_a_qlim(robot, sol.q[None, :])
                    _, ll, de = _evaluar(robot, q, objetivo[k:k+1], tol)
                    if ll[0] and de[0]:
                        d = np.linalg.norm(q[0] - ref) if ref is not None else 0.0
                        if d < mejor_d:             # la más cercana al vecino = continuidad
                            mejor, mejor_d = q[0], d
                        if ref is None:
                            break
                if mejor is not None:
                    qs[k] = mejor
                    valido[k] = True

    con_q, llega, dentro = _evaluar(robot, qs, objetivo, tol)
    valido = llega & dentro
    reparados = valido.sum() - n_inicial

    # figura cerrada en el espacio -> que cierre también en articulares
    if cerrar and len(qs) > 1 and valido[0] and valido[-1] \
            and np.allclose(objetivo[0], objetivo[-1]):
        qs[-1] = qs[0]

    colores = np.where(valido, 'g', 'r')
    print(f"{valido.sum()} de {len(qs)} puntos correctos"
          + (f" ({reparados} reparados)" if reparados > 0 else ""))
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