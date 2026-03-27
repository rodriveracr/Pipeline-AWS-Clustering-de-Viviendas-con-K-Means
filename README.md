# Pipeline AWS — Clustering de Viviendas con K-Means
**Autor:** Rodolfo Villarreal Rivera (`rodriveracr`)

---

## Arquitectura

```
Viviendas_limpio.csv
        │
        ▼
┌─────────────┐     upload raw CSV      ┌────────────────────────┐
│   Local /   │ ─────────────────────►  │  S3 (rodriveracr-      │
│   EC2       │                         │  viviendas-pipeline)   │
└─────────────┘                         └────────────┬───────────┘
        │                                            │
        │  descarga + preprocesa                     │ train.protobuf
        ▼                                            ▼
┌─────────────────┐                    ┌─────────────────────────┐
│  EC2 t3.medium  │ ─── sube datos ──► │  SageMaker K-Means      │
│  (preprocessing)│                    │  (ml.c5.xlarge, k=4)    │
└─────────────────┘                    └────────────┬────────────┘
                                                    │ etiquetas cluster
                                                    ▼
                                       ┌─────────────────────────┐
                                       │  DynamoDB               │
                                       │  rodriveracr_viviendas  │
                                       │  _clusters              │
                                       └─────────────────────────┘
```

---

## Dataset — `Viviendas_limpio.csv`
| Columna       | Tipo     | Descripción                    |
|---------------|----------|--------------------------------|
| Precio        | float    | Precio de la vivienda (€)      |
| Superficie    | int      | Superficie en m²               |
| Antiguedad    | float    | Años de antigüedad             |
| VPO           | int      | Vivienda de protección oficial |
| Calefaccion   | string   | Eléctrica / Central / Gas / No |
| Habitaciones  | int      | Número de habitaciones         |
| Ascensor      | string   | Si / No                        |
| Garaje        | string   | Si / No                        |
| Precio_m2     | float    | Precio por m²                  |

**748 registros** — los clusters K-Means agrupan viviendas similares.

---

## Archivos del proyecto

| Archivo           | Propósito                                        |
|-------------------|--------------------------------------------------|
| `pipeline.py`     | Pipeline principal (S3 → EC2 → SageMaker → DynamoDB) |
| `launch_ec2.py`   | Lanza instancia EC2 con el pipeline             |
| `ec2_userdata.sh` | Script de bootstrap para la instancia EC2       |
| `iam_policy.json` | Políticas IAM necesarias                        |

---

## Ejecución rápida

### Opción A — Correr localmente (con credenciales AWS configuradas)
```bash
pip install boto3 pandas numpy scikit-learn sagemaker
python pipeline.py
```

### Opción B — Correr en EC2 (recomendado para producción)
```bash
# 1. Configurar credenciales AWS
aws configure

# 2. Subir código y lanzar EC2
python launch_ec2.py

# 3. Ver logs en tiempo real
aws s3 cp s3://rodriveracr-viviendas-pipeline/logs/ . --recursive
```

---

## Requisitos previos

1. **Cuenta AWS** con los servicios habilitados: S3, EC2, SageMaker, DynamoDB
2. **Rol IAM** `EC2SageMakerPipelineRole` creado con la política de `iam_policy.json`
3. **Key pair EC2** llamado `rodriveracr-key` (o editar `launch_ec2.py`)
4. **Python 3.8+** con `boto3`, `sagemaker`, `pandas`, `scikit-learn`

---

## Estructura en S3

```
s3://rodriveracr-viviendas-pipeline/
├── viviendas/
│   ├── raw/
│   │   └── Viviendas_limpio.csv
│   ├── processed/
│   │   ├── train.protobuf
│   │   └── viviendas_procesado.csv
│   ├── model/
│   │   └── <job-name>/output/model.tar.gz
│   └── inference/
│       ├── input.csv
│       └── output/input.csv.out
├── code/
│   └── pipeline.py
└── logs/
    └── pipeline_output_<timestamp>.log
```

---

## Tabla DynamoDB — `rodriveracr_viviendas_clusters`

Cada ítem tiene:
- `vivienda_id` (PK) — índice del registro
- `cluster` — grupo K-Means asignado (0–3)
- `owner` — `rodriveracr`
- Todas las features numéricas del dataset

---

*Pipeline generado para rodriveracr — Rodolfo Villarreal Rivera*
