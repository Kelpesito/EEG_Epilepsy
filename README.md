# Pipeline: Procesado de EEG para Epilepsia

Este proyecto proporciona herramientas para visualizar, procesar y extraer características de un EEG, además de etiquetar ventanas de tiempo mediante agrupación (clustering).

## Índice
- [Instalación](#instalacion)
- [Uso](#uso)
    - [Opciones disponibles](#opciones-disponibles)
    - [Funciones disponibles](#funciones-disponibles)
    - [Plots disponibles para visualizar](#plots-disponibles-para-visualizar)
    - [Archivos disponibles para guardar](#archivos-disponibles-para-guardar)
- [Ejemplos de uso](#ejemplos-de-uso)

## Instalación

1. Crear un entorno virtual con conda en la versión 3.13.5.
```bash
conda create -n EEGpip python=3.13.5
```
2. Activar el entorno virtual.
```bash
conda activate EEGpip
```
3. Desplazarse hasta la carpeta del proyecto.
```bash
cd 'carpeta/del/proyecto'
```
4. Instalar las dependencias:
```bash
conda install -c conda-forge hdbscan
pip install -r requirements.txt
```
5. Instalar setup
```bash
pip install --editable .
```

### Requisitos

- Python 3.13
- antropy (para el cálculo de entropías (Features))
- autoreject (Eliminación de artefactos)
- colorama (cambios de color en consola)
- h5py (manejar archivos .h5)
- hdbscan (HDBSCAN (clustering))
- matplotlib
- mne (procesado de EEG)
    - mne_connectivity (Conectividad funcional)
    - mne_features (Extracción de características)
- numpy <= 2.2
- pandas
- PyWavelets (Cálculo de Wavelets)
- scikit_learn (Estandarización de datos y clustering)
- scipy (Calcular estadísticos)
- seaborn
- tqdm (Barras de progreso)
- umap-learn (Reducción UMAP (clustering))

## Uso
```bash
EEGpip [options]
```

### Opciones disponibles
| Opción                   | Descripción                                                                                                                                 | Valores / Uso                                         | Valor por defecto |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-------------------|
| `--function`, `-f`       | Función para ejecutar                                                                                                                       | `run`, `inspect`, `correct`, `features`, `clustering`, `versus`, `labels` | `run`             |
| `--sel_ecg`              | Seleccionar ECG para corrección de artefactos                                                                                               | -                                                     | False             |
| `--epoch_duration`, `-d` | Duración de las ventanas de tiempo (en segundos)                                                                                            | Número                                                | 10                |
| `--montage`, `-m`        | Seleccionar montaje para re-referenciar                                                                                                     | `average`, `bipolar`, `laplacian`                     | `average`         |
| `--dim_reduction`, `-r`  | Modo de reducción de canales                                                                                                                | `multichannel`, `average`                             | `multichannel`    |
| `--save`, `-s`           | Ubicación de los archivos de salida                                                                                                         | Ruta o nombre                                         | None              |
| `--visualize`, `-v`      | Mostrar todos los plots durante la ejecución                                                                                                | -                                                     | False             |
| `--plot_f`               | Mostrar únicamente plots finales (después de re-referenciar, visualización clusters y plot del EEG etiquetado) | -                                                     | False             |
| `--var_selection` | Seleccionar tipos de características para calcular (lista) | `statistical`, `autocorrelation`, `band_power`, `entropy`, `hjorth`, `fractal`, `bispectrum`, `wavelets`, `functional_connectivity` | None (todas las características)

### Funciones disponibles
#### Función inspeccionar EEG
(útil para saber si seleccionar ECG en corrección de artefactos)
```bash
EEGpip -f inspect
```
#### Función para el preprocesado de EEG:
Filtrado, corrección de artefactos, re-referencia y estandarización. 
```bash
EEGpip -f correct --sel_ecg -d -m -s --plot_f -v
```
#### Función extracción de características
```bash
EEGpip -f features --var_selection -r -s (-m)
```
#### Función algoritmo de clustering
(intra-EEG)
```bash
EEGpip -f clustering -s --plot_f -v (-m)
```
#### Función análisis comparativo
```bash
EEGpip -f versus -s
```
#### Función etiquetado de épocas manual
```bash
EEGpip -f labels -s
```
#### Función principal
Preprocesado, extracción de características y clustering
```bash
EEGpip (-f run) --sel_ecg -d -m --var_selection -r -s --plot_f -v
```

### Plots disponibles para visualizar
- EEG crudo
- EEG filtrado
- Épocas rechazadas en Autoreject 1
- Topograma de los componentes ICA
- Gráfico de cada componente ICA
- Puntuación EMG de cada componente ICA
- Puntuación ECG de cada componente ICA
- Épocas rechazadas en Autoreject 2
- EEG después de eliminar artefactos
- EEG después de re-referenciar
- Dendograma de HDBSCAN
- Visualización de los clusters
- EEG etiquetado
- Figuras del análisis comparativo
- EEG en épocas para el etiquetado manual

### Archivos disponibles para guardar
- EEG corregido de artefactos (.fif): `nombre`-epo.fif
- Vector de características (.csv): `nombre`_features.csv
- Etiquetas (.csv): `nombre`_cluster_labels.csv
- Array (n_epochs, n_channels, n_times) (.h5): `nombre`.h5
- Metadatos Array (.json): `nombre`_metadata.json
- Etiquetas ground truth (.npy): `nombre`_labels.npy
- Importancias de las variables en el análisis comparativo pre vs post (.csv): `nombre`_importances.csv
- Figuras del análisis comparativo (.png):
    - Comparación EEG pre vs post: `nombre`_EEG1_vs_EEG2.png
    - Top n características: `nombre`_top_features.png
    - Distribución de la característica `i` por canal: `nombre`_top_features_`i`.png
    - Densidad Espectral de Potencia (PSD) de EEG pre (`n = 1`) y post (`n = 2`): `nombre`_psd_`n`.png
    - Topomapa de EEG pre (`n = 1`) y post (`n = 2`): `nombre`_topomap_`n`.png
    - Matriz de conectividad funcional de EEG pre (`n = 1`), post (`n = 2`) y diferencia (`n = delta`): `nombre`_FC_`n`.png

## Ejemplos de uso
#### Ejecutar pipeline completa: Montaje average, 10 segundos de epoch, mostrar todos los plots, reducción de canales por media (no guardar)
```bash
EEGpip -v -r average
```

#### Inspeccionar EEG (mostrar pre y post filtrado)
```bash
EEGpip -f inspect
```

#### Corregir artefactos: Seleccionar ECG, montaje laplaciano, 2 segundos de epoch, ver solo plots finales y guardar
```bash
EEGpip -f correct --sel_ecg -m laplacian -d 2 --plot_f -s ruta/nombre
```

#### Extraer características: Sin reducción de canales, guardar
```bash
EEGpip -f features -s nombre
```
Es posible importar los datos desde .fif o .h5. Si se importa desde .fif, es necesario añadir el argumento `-m` o `--montage`, si no, por defecto se aplica el montaje average.

#### Extraer características: Únicamente estadísticas, banda de frecuencia y conectividad funcional
```bash
EEGpip -f features --var_selection statistical band_power functional_connectivity
```

#### Realizar el clustering: Visualizar todos los plots y no guardar
```bash
EEGpip -f clustering -v
```
Si se desea visualizar el EEG final (`-v`, `--visualize` o `--plot_f`), y se importa el archivo desde .fif, es necesario añadir el argumento `-m` o `--montage`, si no, por defecto se aplica el montaje average.

#### Análisis comparativo entre dos EEG
```bash
EEGpip -f versus
```

#### Etiquetado manual de EEG, guardando
```bash
EEGpip -f labels -s ruta/nombre
