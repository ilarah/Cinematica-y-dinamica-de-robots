Para tener estos archivos en tu computadora, sigue estos pasos:

1. Haz clic en el botón verde **"Code"** ubicado en la esquina superior derecha de esta página.
2. Selecciona la opción **"Download ZIP"**.
3. Descomprime el archivo `.zip` en tu computadora y ¡listo! Ya puedes abrir los códigos en tu entorno de desarrollo.

*Nota para avanzados:* Si sabes usar Git, puedes clonar el repositorio usando:
```bash
git clone https://github.com/ilarah/Cinematica-y-dinamica-de-robots
```

## Requisitos de Software y Configuración

Para poder ejecutar estos códigos sin errores, necesitas configurar tu entorno de desarrollo con **Python 3.11** y las librerías científicas y de robótica. Sigue estos pasos en tu terminal o Anaconda Prompt:

### 1️. Entorno Base y Jupyter
Instala el núcleo de Python, Jupyter y las herramientas de análisis de datos ejecutando:
```bash
conda install -y numpy scipy matplotlib pandas sympy jupyter ipykernel notebook jupyterlab
```

### 2️. Paquetes de Robótica y Visualización
Instala las librerías especializadas en robótica cinemática, simulación y gráficos avanzados ejecutando:
```bash
python -m pip install roboticstoolbox-python swift-sim spatialmath-python spatialgeometry pillow imageio tqdm seaborn plotly
```

### 3️. Extensiones Obligatorias para VS Code
Si trabajas desde **Visual Studio Code**, es indispensable que instales las siguientes extensiones desde el Marketplace (`Ctrl + Shift + X`):
* **Python** (de Microsoft)
* **Jupyter** (de Microsoft)
