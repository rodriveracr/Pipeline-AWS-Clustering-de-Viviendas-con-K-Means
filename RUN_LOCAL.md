Instrucciones para ejecutar el pipeline localmente

Requisitos:
- Python 3.8+
- Tener el archivo `Viviendas.xlsx` o `Viviendas_limpio.csv` en el directorio del proyecto

Pasos rápidos (Windows PowerShell):

1. Crear y activar entorno virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias

```powershell
pip install -r requirements.txt
```

3. Convertir Excel a CSV (si subiste `Viviendas.xlsx`)

```powershell
python convert_xlsx_to_csv.py
```

4. Ejecutar el runner local (genera `viviendas_local_with_clusters.csv`)

```powershell
python local_run.py
```

Salida esperada:
- `Viviendas_limpio.csv` (dataset limpio)
- `viviendas_local_with_clusters.csv` (dataset con columna `cluster`)

Notas:
- Para ejecutar el pipeline real en AWS, revisa `pipeline.py`, crea el rol IAM según `iam_policy.json` y configura `aws configure`.
- `pipeline.py` usa SageMaker y `sagemaker.get_execution_role()` — si ejecutas desde tu máquina local debes pasar un `ROLE_ARN` válido o ejecutar desde una instancia/Notebook con el rol adjunto.
