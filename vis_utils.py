"""
viz_utils.py  -  utilidades de visualización para transformaciones homogéneas.
 
Contenido:
    Punto, Vector, Vect, dropline      -> dibujo estático
    dibujar_frame                      -> dibuja un sistema de referencia
    animar_transformaciones            -> anima una secuencia de pasos elementales
                                          mostrando los frames intermedios
"""
import numpy as np
from matplotlib.animation import FuncAnimation
from spatialmath import SE3
from spatialmath.base import h2e, plotvol3, trinterp
 
COLORES_EJES = ('red', 'green', 'blue')
 
 
# ----------------------------------------------------------------------------
# Dibujo estático
# ----------------------------------------------------------------------------
def Punto(origen, color, ax):
    """Punto en coordenadas homogéneas [x, y, z, 1]."""
    origen = np.asarray(h2e(origen), dtype=float).flatten()
    return ax.scatter(origen[0], origen[1], origen[2], color=color, s=60)
 
 
def Vector(ax, origen1, origen2, color, label=None):
    """Flecha desde origen1 hasta origen2 (puntos planos o SE3)."""
    p1 = origen1.t if hasattr(origen1, 't') else origen1
    p2 = origen2.t if hasattr(origen2, 't') else origen2
    p1 = np.asarray(p1, dtype=float).flatten()[:3]
    p2 = np.asarray(p2, dtype=float).flatten()[:3]
    direccion = p2 - p1
    ax.quiver(p1[0], p1[1], p1[2], *direccion, color=color,
              arrow_length_ratio=0.15, label=label)
 
 
def Vect(ax, origen1, origen2, estilo='--'):
    """Línea simple entre dos puntos."""
    p1 = np.asarray(origen1, dtype=float).flatten()[:3]
    p2 = np.asarray(origen2, dtype=float).flatten()[:3]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], estilo)
 
 
def dropline(ax, punto, estilo=':', color='gray', etiquetas=True):
    """Líneas guía punteadas del punto hacia los planos coordenados."""
    punto = np.asarray(h2e(punto), dtype=float).flatten()
    x, y, z = punto
 
    ax.plot([x, x], [y, y], [0, z], estilo, color=color)
    ax.plot([x, x], [0, y], [0, 0], estilo, color=color)
    ax.plot([0, x], [y, y], [0, 0], estilo, color=color)
 
    if etiquetas:
        ax.text(x, y, z / 2, f'{z:g}', color=color)
        ax.text(x / 2, y, 0, f'{x:g}', color=color)
        ax.text(x, y / 2, 0, f'{y:g}', color=color)
 
 
def dibujar_frame(ax, T, largo=1.5, estilo='-', alpha=1.0, etiqueta=None, lw=2,
                  desfase=0.25):
    """
    Dibuja los tres ejes de un SE3 como líneas (permite estilo punteado).
    La etiqueta se coloca detrás del origen (en sentido opuesto a los tres
    ejes, así no se encima con ninguno) y lleva un fondo blanco para leerse bien.
    `desfase` controla qué tan separada queda.
    """
    o = T.t
    artistas = []
    for i, c in enumerate(COLORES_EJES):
        p = o + largo * T.R[:, i]
        (linea,) = ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]],
                           estilo, color=c, alpha=alpha, lw=lw)
        artistas.append(linea)
    if etiqueta:
        pos = o - desfase * (T.R @ np.ones(3))
        artistas.append(ax.text(pos[0], pos[1], pos[2], etiqueta, fontsize=9,
                                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                          ec='none', alpha=0.7)))
    return artistas
 
 
# ----------------------------------------------------------------------------
# Animación de una secuencia de transformaciones
# ----------------------------------------------------------------------------
def animar_transformaciones(pasos, punto_b=None, modo='movil', dim=(-1, 11),
                            cuadros=30, largo_ejes=1.5, intervalo=40,
                            imprimir=True, mostrar_coords=False):
    """
    Anima una secuencia de transformaciones ELEMENTALES (una rotación pura o
    una traslación pura por paso), dejando un frame punteado al terminar cada
    paso y la estela punteada del origen de B y del punto.
 
    Parámetros
    ----------
    pasos    : lista de SE3 elementales, en el orden en que se aplican.
               Ej.: [SE3.Trans(6, -3, 8), SE3.Rz(90, 'deg')]
    punto_b  : (opcional) coordenadas [x, y, z] de un punto fijo en el sistema B.
    modo     : 'movil' -> cada paso se expresa respecto al sistema móvil
                          (post-multiplicación, T = T * paso). Es el que usa DH.
               'fijo'  -> cada paso se expresa respecto al sistema fijo A
                          (pre-multiplicación, T = paso * T).
    cuadros  : cuadros de animación por paso.
 
    Devuelve (anim, T_total). Guarda `anim` en una variable para que la
    animación no se destruya.
    """
    if modo not in ('movil', 'fijo'):
        raise ValueError("modo debe ser 'movil' o 'fijo'")
 
    pasos = list(pasos)
    n = len(pasos)
 
    def compone(T, P):
        return T * P if modo == 'movil' else P * T
 
    def fraccion(paso, s):
        # fracción s (0..1) de un paso elemental, interpolada desde la identidad
        return SE3(trinterp(np.eye(4), paso.A, s))
 
    # Pose acumulada al inicio de cada paso (T_ini[i]) y al final de todos
    T_ini = [SE3()]
    for p in pasos:
        T_ini.append(compone(T_ini[-1], p))
    T_total = T_ini[-1]
 
    tiene_punto = punto_b is not None
    if tiene_punto:
        Pb_h = np.r_[np.asarray(punto_b, dtype=float).flatten()[:3], 1.0]
 
    if imprimir:
        for i, Ti in enumerate(T_ini[1:], start=1):
            print(f'--- Pose acumulada después del paso {i} ({modo}) ---')
            print(Ti)
            if tiene_punto:
                print('Punto en A (homogéneo):', np.round(Ti.A @ Pb_h, 3))
 
    # Escena
    ax = plotvol3(dim=list(dim), equal=True)
    SE3().plot(ax=ax, frame='A', color='black')
 
    ejes = [ax.plot([], [], [], color=c, lw=2)[0] for c in COLORES_EJES]
    estela_o, = ax.plot([], [], [], ':', color='gray', lw=1.5)
    estela_p, = ax.plot([], [], [], ':', color='magenta', lw=1.5)
    marcador, = ax.plot([], [], [], 'o', color='magenta', markersize=8)
 
    texto_coords = None
    if tiene_punto and mostrar_coords:
        texto_coords = ax.text2D(0.02, 0.02, '', transform=ax.transAxes,
                                 fontsize=9)
 
    tray_o = [T_ini[0].t]
    tray_p = [(T_ini[0].A @ Pb_h)[:3]] if tiene_punto else []
 
    def update(k):
        i, j = divmod(k, cuadros)
        s = (j + 1) / cuadros
        T = compone(T_ini[i], fraccion(pasos[i], s))
        o = T.t
 
        # ejes móviles de B
        for e, eje in enumerate(ejes):
            p = o + largo_ejes * T.R[:, e]
            eje.set_data_3d([o[0], p[0]], [o[1], p[1]], [o[2], p[2]])
 
        # estela del origen de B
        tray_o.append(o)
        xs, ys, zs = np.array(tray_o).T
        estela_o.set_data_3d(xs, ys, zs)
 
        # punto fijo en B, visto desde A
        if tiene_punto:
            aP = (T.A @ Pb_h)[:3]
            marcador.set_data_3d([aP[0]], [aP[1]], [aP[2]])
            tray_p.append(aP)
            xp, yp, zp = np.array(tray_p).T
            estela_p.set_data_3d(xp, yp, zp)
            if texto_coords is not None:
                texto_coords.set_text(
                    f'Punto en A = [{aP[0]:.1f}, {aP[1]:.1f}, {aP[2]:.1f}]')
 
        # al terminar un paso, deja el frame intermedio punteado
        if j == cuadros - 1:
            dibujar_frame(ax, T, largo_ejes, estilo=':', alpha=0.6,
                          etiqueta=f'{{{i + 1}}}')
 
        ax.set_title(f'Paso {i + 1}/{n}  (modo {modo})')
        return ejes + [estela_o, estela_p, marcador]
 
    anim = FuncAnimation(ax.figure, update, frames=n * cuadros,
                         interval=intervalo, repeat=False)
    return anim, T_total