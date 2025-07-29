# Pipeline: Procesado de EEG para Epilepsia

Este proyecto proporciona herramientas para visualizar, procesar y extraer características de un EEG, además de etiquetar ventanas de tiempo mediante agrupación (clustering).

## Índice
- [Instalación](#instalacion)
- [Uso](#uso)
    - [Opciones disponibles](#opciones-disponibles)
    - [Plots disponibles para visualizar](#plots-disponibles-para-visualizar)
    - [Archivos disponibles para guardar](#archivos-disponibles-para-guardar)
- [Ejemplos de uso](#ejemplos-de-uso)

## Instalación

Primero, instalar las dependencias con (es altamente recomendado hacerlo con conda):
```bash
pip install -r requirements.txt
```
Hay una dependencia que no se puede instalar con pip. Para instalarla:
```bash
conda install -c conda-forge hdbscan
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
- numpy < 2.3
- pandas
- PyWavelets (Cálculo de Wavelets)
- scikit_learn (Estandarización de datos y clustering)
- scipy (Calcular estadísticos)
- seaborn
- tqdm (Barras de progreso)
- umap-learn (Reducción UMAP (clustering))

## Uso
```bash
python main.py -f [options]
```

### Opciones disponibles
| Opción                   | Descripción                                                                                                                                 | Valores / Uso                                         | Valor por defecto |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-------------------|
| `--function`, `-f`       | Función para ejecutar                                                                                                                       | `run`, `inspect`, `correct`, `features`, `clustering` | `run`             |
| `--sel_ecg`              | Seleccionar ECG para corrección de artefactos                                                                                               | -                                                     | False             |
| `--epoch_duration`, `-d` | Duración de las ventanas de tiempo (en segundos)                                                                                            | Número                                                | 10                |
| `--montage`, `-m`        | Seleccionar montaje para re-referenciar                                                                                                     | `average`, `bipolar`, `laplacian`                     | `average`         |
| `--dim_reduction`, `-r`  | Modo de reducción de canales                                                                                                                | `multichannel`, `average`                             | `multichannel`    |
| `--save`, `-s`           | Ubicación de los archivos de salida                                                                                                         | Ruta o nombre                                         | None              |
| `--visualize`, `-v`      | Mostrar todos los plots durante la ejecución                                                                                                | -                                                     | False             |
| `--plot_f`               | Mostrar únicamente plots finales (después de re-referenciar, visualización clusters y plot del EEG etiquetado) | -                                                     | False             |

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

### Archivos disponibles para guardar
- EEG corregido de artefactos (.fif): `nombre`-epo.fif
- Vector de características (.csv): `nombre`_features.csv
- Etiquetas (.csv): `nombre`_cluster_labels.csv
- Array (n_epochs, n_channels, n_times) (.h5): `nombre`.h5
- Metadatos Array (.json): `nombre`_metadata.json

## Ejemplos de uso
#### Ejecutar pipeline completa: Montaje average, 10 segundos de epoch, mostrar todos los plots, reducción de canales por media (no guardar)
```bash
python main.py -v -r average
```

#### Inspeccionar EEG (mostrar pre y post filtrado)
```bash
python main.py -f inspect
```

#### Corregir artefactos: Seleccionar ECG, montaje laplaciano, 2 segundos de epoch, ver solo plots finales y guardar
```bash
python main.py -f correct --sel_ecg -m laplacian -d 2 --plot_f -s ruta/nombre
```

#### Extraer características: Sin reducción de canales, guardar
```bash
python main.py -f features -s nombre
```
Es posible importar los datos desde .fif o .h5. Si se importa desde .fif, es necesario añadir el argumento `-m` o `--montage`, si no, por defecto se aplica el montaje average. 

#### Realizar el clustering: Visualizar todos los plots y no guardar
```bash
python main.py -f clustering -v
```
Si se desea visualizar el EEG final (`-v`, `--visualize` o `--plot_f`), es necesario añadir el argumento `-m` o `--montage`, si no, por defecto se aplica el montaje average.
