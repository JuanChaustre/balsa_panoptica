# Segmentación panóptica de madera balsa

Proyecto para pre-etiquetar localmente las fotos, corregir las anotaciones,
entrenar un modelo panóptico y compararlo con el Mask2Former del Taller 3.
Las imágenes originales y los datos de anotación se mantienen locales; el
checkpoint entrenado para la aplicación en vivo se distribuye por Git LFS.

## Dataset y clases

`tools/preparar_dataset.py` verifica los JPG y crea `data/manifest.csv` y
`data/splits.json`. Las imágenes se ordenan temporalmente y se separan en
train/validation/test, excluyendo ocho fotogramas a cada lado de cada frontera
para reducir la fuga de fotogramas casi idénticos entre particiones.

Las clases iniciales están en `annotation_config.json`:

| Clase | Tipo | Regla |
|---|---|---|
| Pieza de balsa (`balsa_piece`) | Thing | Una instancia por pieza visible |
| Mano/brazo (`hand_arm`) | Thing | Instancia independiente |
| Superficie de trabajo (`work_surface`) | Stuff | Mesa/superficie donde se coloca la balsa |
| Fondo/entorno (`background`) | Stuff | Resto de la escena |

## Descargar el proyecto y sus pesos

Instala Git LFS antes de clonar para descargar también el checkpoint de unos
182 MB. En Ubuntu/Debian:

```bash
sudo apt update
sudo apt install git-lfs
git lfs install
```

En Windows, instala Git LFS y ejecuta `git lfs install` desde Git Bash o
PowerShell. Luego clona el repositorio y descarga los objetos LFS:

```bash
git clone https://github.com/JuanChaustre/balsa_panoptica.git
cd balsa_panoptica
git lfs pull
cd ..
```

Si ya clonaste el repositorio sin Git LFS, desde su carpeta ejecuta:

```bash
git lfs install
git lfs pull
```

El checkpoint y los archivos de configuración requeridos están en
`training/runs/mask2former_balsa_20261117_145746/best_model/`. Comprueba que
`model.safetensors` mida aproximadamente 182 MB; si tiene solo unos bytes,
ejecuta `git lfs pull` después de instalar Git LFS.

## Preparar el entorno de Python

En Ubuntu/Debian instala el soporte `venv` de la misma versión que Python. En
este equipo, que usa Python 3.10:

```bash
sudo apt update
sudo apt install python3.10-venv
```

Si usas otra versión, instala su paquete correspondiente (por ejemplo,
`python3.12-venv`). Esto evita el error `ensurepip is not available` al crear
el entorno.

Desde el directorio que contiene la carpeta clonada `balsa_panoptica`, crea y
activa el entorno e instala las dependencias:

```bash
python3 -m venv --clear .venv-balsa
source .venv-balsa/bin/activate
python -m pip install --upgrade pip
python -m pip install -r balsa_panoptica/requirements.txt
```

`--clear` permite rehacer la carpeta parcial que puede dejar un intento fallido
de creación. Para salir del entorno usa `deactivate`. Si ya creaste el entorno,
basta con activarlo con `source .venv-balsa/bin/activate`.

En Windows PowerShell, desde la carpeta que contiene `balsa_panoptica`:

```powershell
py -m venv .venv-balsa
.\.venv-balsa\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r balsa_panoptica/requirements.txt
```

Para aprovechar CUDA, instala una versión de PyTorch compatible con la GPU y
CUDA del equipo.

## Aplicación para revisar y corregir

Desde el directorio que contiene `balsa_panoptica`, inicia la aplicación local:

```bash
python -m streamlit run balsa_panoptica/app.py
```

La app permite navegar y filtrar imágenes, ver las máscaras superpuestas,
cambiar la clase o eliminar una máscara y agregar/borrar regiones con
polígonos. También permite marcar imágenes como revisadas y descargar las
anotaciones revisadas como COCO JSON, incluidas las categorías thing y stuff.
Los cambios se guardan en `balsa_panoptica/data/corrections/`, sin modificar
`proposals.json` ni las máscaras originales. La exportación incluye solo las
imágenes marcadas como revisadas.

## Detección en tiempo real

La aplicación de cámara usa el mejor checkpoint entrenado y procesa los frames
localmente con Mask2Former:

```bash
python -m streamlit run balsa_panoptica/live_app.py
```

Abre la dirección local que muestra Streamlit, inicia la cámara y concede
permiso al navegador. Se selecciona por defecto el checkpoint más reciente en
`training/runs/mask2former_balsa_*/best_model/`. La aplicación muestra las
instancias thing y las regiones stuff con color, clase y confianza. Para cada
pieza de balsa estima largo × ancho en píxeles desde los ejes principales de su
máscara. Es una medida aproximada de la silueta, no una dimensión física ni una
medición calibrada. No persiste ni envía los fotogramas. Usa entrada de
256×256 como en el entrenamiento. La velocidad dependerá del equipo y la
cámara; no se ha optimizado ni certificado un requisito específico de FPS.

### Usar el modelo ya entrenado

El checkpoint elegido para el proyecto está en
`balsa_panoptica/training/runs/mask2former_balsa_20261117_145746/best_model/`.
La aplicación detecta automáticamente el run más reciente que tenga
`best_model/model.safetensors`; no hay que volver a entrenar para ejecutar la
cámara si ese checkpoint está disponible. Debe contener, como mínimo,
`model.safetensors`, `config.json` y `preprocessor_config.json`.

El checkpoint está incluido en el repositorio mediante Git LFS. No hace falta
volver a entrenar ni copiarlo por separado cuando `git lfs pull` termina
correctamente. Si el checkpoint falta, verifica la instalación de Git LFS y
ejecuta ese comando desde la carpeta del repositorio.

Resultados del checkpoint actual en el test de 17 imágenes: IoU macro **71.56%**,
mAP **53.30%**, F1 macro **82.13%** y precisión macro **82.20%**. Cumple los
umbrales de IoU y F1 de la guía; no alcanza el umbral mAP de 60%. El informe
completo está en `entregables/paquete_informe_balsa_20261117/` en el checkout
actual; los paquetes de entrega se excluyen del repositorio fuente.

Para repetir la suite de pruebas, desde el directorio que contiene
`balsa_panoptica` ejecuta:

```bash
python -m unittest discover -s balsa_panoptica/tests -v
```

## Cargador del dataset COCO

`training/coco_dataset.py` carga el JSON COCO revisado y sus imágenes, decodifica
las máscaras RLE y las convierte en targets panópticos para Mask2Former.
Mantiene cada instancia thing por separado, agrupa cada clase stuff en una
máscara por imagen y marca los píxeles sin etiqueta como `IGNORE_INDEX` en el
mapa panóptico. Importante: las máscaras binarias que recibe Mask2Former no
incluyen una máscara de validez por píxel, por lo que las zonas sin etiqueta
quedan fuera de todos los segmentos objetivo y pueden penalizar predicciones
allí. Completa las regiones de fondo/entorno antes de entrenar. Si hay máscaras
superpuestas, asigna los píxeles en el orden mano/brazo, balsa, superficie y
fondo; las correcciones manuales deben seguir evitando esos solapamientos.

Para validar todas las imágenes revisadas:

```bash
python -m balsa_panoptica.training.coco_dataset \
  --split all --batch-size 1 --image-size 256
```

El cargador también acepta `--split train`, `--split validation` o
`--split test`. La partición original contiene 181 imágenes de entrenamiento,
4 de validación y 13 de prueba; después de fusionar el lote nuevo, las
particiones quedan en 195 train, 9 validation y 17 test. Conviene revisar más
imágenes de validation antes de interpretar métricas. Este comando valida y
prepara los datos, pero no inicia por sí solo el entrenamiento del modelo.

## Entrenamiento de Mask2Former

Entrena un Mask2Former Swin-Tiny inicializado desde
`facebook/mask2former-swin-tiny-coco-panoptic`. La primera ejecución necesita
conexión a Hugging Face para descargar el checkpoint (a menos que ya esté en
la caché). Desde la raíz del workspace:

```bash
python -m balsa_panoptica.training.train_mask2former \
  --epochs 20 --image-size 256 --batch-size 1
```

El entrenamiento usa aumentos aleatorios de espejo horizontal/vertical, AMP en
CUDA y parada temprana según la pérdida de validación. Solo utiliza las
particiones train y validation; no evalúa ni usa test para elegir el modelo.
La salida se crea en una carpeta nueva bajo
`balsa_panoptica/training/runs/`: `best_model/` contiene el checkpoint con menor
pérdida de validación, `last_training.pt` el último estado, y
`loss_history.csv`/`loss_curve.png` el historial. La partición actual de
validación tiene solo 9 imágenes, así que el checkpoint elegido y su pérdida
son muy inciertos; no interpretes una curva favorable como evidencia de buena
generalización. Los solapamientos se resuelven con la prioridad documentada
arriba y se reportan durante el entrenamiento.

Se pueden ajustar `--learning-rate`, `--patience`, `--device` (`auto`, `cuda`,
`cpu`) y usar `--no-amp`. Para comprobar el ciclo con una época corta se puede
añadir `--epochs 1 --max-train-batches 1 --max-validation-batches 1`; incluso
esa prueba carga el checkpoint y realiza una actualización real, no es un
entrenamiento completo.

Para calcular las métricas que pide la guía sobre el conjunto test (sin usarlo
para elegir el checkpoint):

```bash
python -m balsa_panoptica.training.evaluate_mask2former \
  --run-dir balsa_panoptica/training/runs/mask2former_balsa_YYYYMMDD_HHMMSS
```

El evaluador guarda `test_metrics.json` junto al checkpoint. IoU, precisión y
F1 son promedios macro por píxel sobre las cuatro clases, excluyendo píxeles
sin etiqueta. mAP es AP de máscaras de instancia para las clases thing
(`balsa_piece` y `hand_arm`), promediado en umbrales de IoU 0.50 a 0.95. Para
que las etiquetas incompletas no cuenten como errores, las máscaras de instancia
predichas se recortan a los píxeles anotados. Los resultados deben interpretarse
con cautela: el split test tiene 17 imágenes. No son comparables directamente
con métricas del Taller 2 si aquel modelo usa clases o imágenes distintas.

## Ampliar el dataset y preparar el entrenamiento

Para fusionar `balsa_coco.json` con `balsa_coco2.json`, regenerar IDs COCO únicos
y extender las particiones sin mover imágenes antiguas, ejecuta:

```bash
python -m balsa_panoptica.training.merge_coco_datasets \
  --delete-unreviewed-new-images
```

La fusión escribe `data/annotations/balsa_coco_merged.json` y
`data/splits_merged.json`; conserva intactos los dos COCO fuente y
`data/splits.json`. Las fotos WhatsApp se ordenan por su nombre de captura y
las que comparten el mismo nombre base (con sufijos `(1)`, `(2)`, etc.) se
mantienen juntas. Los grupos se asignan cronológicamente a train, validation y
test, aproximadamente en proporción 60/20/20; en este lote son 14, 5 y 4
fotos, respectivamente. Así se evita repartir fotos con la misma marca de
captura entre particiones. La opción de borrado elimina únicamente fotos del
lote que no aparecen en `balsa_coco2.json`. Los demás JPG no se eliminan. La
fusión actual tiene 224 imágenes anotadas: 195 train, 9 validation y 17 test.

El entrenador usa por defecto el COCO y las particiones fusionados:

```bash
python -m balsa_panoptica.training.train_mask2former \
  --epochs 20 --image-size 256 --batch-size 1
```

## Informe de métricas y comparación

Después de entrenar, genera un informe comparable en estructura al de
Cartografía con:

```bash
python -m balsa_panoptica.training.generate_comparison_report \
  --run-dir balsa_panoptica/training/runs/mask2former_balsa_YYYYMMDD_HHMMSS
```

El generador evalúa el checkpoint en test y crea `comparison/comparison.md`,
`comparison.csv`, gráficas de métricas y curvas de pérdidas. También produce
ejemplos con fotos nuevas: imagen, anotación revisada y predicción. Por defecto
selecciona hasta seis fotos del lote, independientemente de su partición, y
muestra junto a cada nombre si es train, validation o test. Para incluir las
23 fotos usa `--max-examples 23`. Las imágenes de train son ejemplos in-sample;
validation y test son ejemplos de particiones retenidas. El informe las presenta
como visualización cualitativa; las métricas se calculan aparte sobre test. El
Mask2Former cartográfico conserva sus métricas históricas en su test original;
sus clases e imágenes son diferentes, por lo que el informe explica que la
tabla no representa una comparación controlada.

## Pre-etiquetado local

El pre-etiquetado usa Grounding DINO Tiny para proponer bloques de madera y
manos/brazos a partir de texto, y SAM ViT-B para generar sus máscaras. No hay
un modelo preentrenado con la clase específica “madera balsa” en el proyecto:
estas propuestas pueden fallar y deben corregirse antes del entrenamiento.
Las regiones stuff (superficie de trabajo y fondo) se completan durante la
anotación; no se asignan automáticamente para evitar pseudo-etiquetas de
entorno engañosas.

Con el entorno virtual activado y desde el directorio que contiene
`balsa_panoptica`:

```bash
python balsa_panoptica/tools/preparar_dataset.py

python balsa_panoptica/tools/prelabel.py --limit 3
```

La primera ejecución descarga los pesos públicos de Hugging Face al caché local
(aprox. 1 GB entre ambos modelos); las fotos no se envían. Si las propuestas de
prueba son útiles, procesa las 322 imágenes actuales con:

```bash
python balsa_panoptica/tools/prelabel.py
```

Cada ejecución crea un directorio nuevo bajo `data/prelabels/` con
`proposals.json` y máscaras PNG binarias en `masks/`. Los resultados son
propuestas, no anotaciones de referencia ni un formato listo para entrenar.
Corregir clases, límites, instancias y regiones stuff antes de convertirlos al
formato panóptico de entrenamiento. `detection_score` y `sam_iou_estimate` son
salidas de los modelos, no métricas de calidad de las anotaciones.

Para exportar las máscaras thing a COCO Instance Segmentation (RLE), ejecuta:

```bash
python balsa_panoptica/tools/exportar_coco.py \
  balsa_panoptica/data/prelabels/run_YYYYMMDD_HHMMSS
```

El export no incluye máscaras stuff ni los JPG, y no reemplaza la revisión
humana. Carga las mismas imágenes desde `../images` junto con
`coco_instances.json` en una herramienta de anotación compatible con COCO;
corrige las instancias y completa allí las regiones de superficie y fondo.

Para proponer etiquetas solo a un grupo de fotos, usa `--filename-pattern`:

```bash
python balsa_panoptica/tools/prelabel.py \
  --filename-pattern 'WhatsApp Image 2026-11-17*'
```

La app de corrección encontrará esa ejecución como una ejecución independiente;
al exportar, COCO contendrá únicamente las imágenes de esa ejecución que se
marquen como revisadas.

## Comparación con el Taller 3

El Mask2Former anterior fue entrenado para clases cartográficas (acera,
manzana construida, zona verde y carretera), por lo que sus predicciones no
son etiquetas válidas para balsa. La comparación del nuevo modelo debe hacerse
con ambos modelos sobre el mismo conjunto de prueba de balsa y con las mismas
definiciones de clase; los resultados cartográficos originales se reportan
aparte.
