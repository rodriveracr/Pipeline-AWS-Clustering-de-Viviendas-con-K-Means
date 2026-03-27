"""
Pipeline AWS - Clustering de Viviendas con K-Means
Autor: Rodolfo Villarreal Rivera (rodriveracr)
Servicios: S3 → EC2 (preprocesamiento) → SageMaker K-Means → DynamoDB
"""

import boto3
import pandas as pd
import numpy as np
import json
import io
import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
REGION          = "us-east-1"
BUCKET_NAME     = "rodriveracr-viviendas-pipeline"
PREFIX          = "viviendas"
DYNAMODB_TABLE  = "rodriveracr_viviendas_clusters"
N_CLUSTERS      = 4          # número de clusters K-Means
LOCAL_CSV       = "Viviendas_limpio.csv"

session    = boto3.Session(region_name=REGION)
s3         = session.client("s3")
dynamodb   = session.resource("dynamodb", region_name=REGION)
sm_session = sagemaker.Session(boto_session=session)


# ═══════════════════════════════════════════════════════════════
# PASO 1 — Crear bucket S3 y subir el dataset original
# ═══════════════════════════════════════════════════════════════
def paso1_subir_a_s3():
    print("\n[PASO 1] Subiendo dataset a S3...")

    # Crear bucket si no existe
    existing = [b["Name"] for b in s3.list_buckets()["Buckets"]]
    if BUCKET_NAME not in existing:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        print(f"  ✅ Bucket creado: {BUCKET_NAME}")
    else:
        print(f"  ℹ️  Bucket ya existe: {BUCKET_NAME}")

    # Subir CSV original
    s3_key = f"{PREFIX}/raw/{LOCAL_CSV}"
    s3.upload_file(LOCAL_CSV, BUCKET_NAME, s3_key)
    print(f"  ✅ Archivo subido → s3://{BUCKET_NAME}/{s3_key}")
    return s3_key


# ═══════════════════════════════════════════════════════════════
# PASO 2 — Preprocesamiento (lógica que correría en EC2)
# ═══════════════════════════════════════════════════════════════
def paso2_preprocesar_en_ec2(raw_s3_key: str):
    """
    Este bloque representa la lógica que se ejecutaría en una
    instancia EC2 (p.ej. t3.medium).  Aquí se corre localmente
    para ilustrar el pipeline end-to-end, pero el script
    `ec2_userdata.sh` adjunto muestra cómo lanzarlo en EC2 real.
    """
    print("\n[PASO 2] Preprocesamiento (lógica EC2)...")

    # Descargar desde S3
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=raw_s3_key)
    df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
    print(f"  Dataset cargado: {df.shape}")

    # Eliminar columnas innecesarias
    df = df.drop(columns=["Unnamed: 0", "Transaccion"], errors="ignore")

    # Encodear variables categóricas
    le = LabelEncoder()
    for col in ["Calefaccion", "Ascensor", "Garaje", "VPO"]:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Imputar nulos en Antiguedad con la mediana
    df["Antiguedad"] = df["Antiguedad"].fillna(df["Antiguedad"].median())

    # Escalar features
    feature_cols = ["Precio", "Superficie", "Antiguedad", "VPO",
                    "Calefaccion", "Habitaciones", "Ascensor",
                    "Garaje", "Precio_m2"]
    scaler       = StandardScaler()
    X_scaled     = scaler.fit_transform(df[feature_cols])

    # Guardar en formato RecordIO-protobuf que SageMaker KMeans espera
    from sagemaker.amazon.common import write_numpy_to_dense_tensor

    buf = io.BytesIO()
    write_numpy_to_dense_tensor(buf, X_scaled.astype("float32"))
    buf.seek(0)

    train_key = f"{PREFIX}/processed/train.protobuf"
    s3.put_object(Bucket=BUCKET_NAME, Key=train_key, Body=buf.getvalue())
    print(f"  ✅ Datos procesados subidos → s3://{BUCKET_NAME}/{train_key}")

    # También guardar CSV procesado (para referencia / DynamoDB)
    csv_buf  = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_key  = f"{PREFIX}/processed/viviendas_procesado.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=csv_key,
                  Body=csv_buf.getvalue().encode())
    print(f"  ✅ CSV procesado subido   → s3://{BUCKET_NAME}/{csv_key}")

    return train_key, df, feature_cols


# ═══════════════════════════════════════════════════════════════
# PASO 3 — Entrenar K-Means con SageMaker
# ═══════════════════════════════════════════════════════════════
def paso3_sagemaker_kmeans(train_key: str):
    print(f"\n[PASO 3] Entrenando K-Means en SageMaker (k={N_CLUSTERS})...")

    role        = get_execution_role()          # rol IAM asociado a SageMaker
    output_path = f"s3://{BUCKET_NAME}/{PREFIX}/model/"

    # Imagen oficial de KMeans de SageMaker
    image_uri = sagemaker.image_uris.retrieve(
        framework="kmeans",
        region=REGION,
    )

    estimator = sagemaker.estimator.Estimator(
        image_uri=image_uri,
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        output_path=output_path,
        sagemaker_session=sm_session,
    )

    estimator.set_hyperparameters(
        k=N_CLUSTERS,
        feature_dim=9,       # número de features (len(feature_cols))
        mini_batch_size=100,
        epochs=5,
    )

    train_input = sagemaker.inputs.TrainingInput(
        s3_data=f"s3://{BUCKET_NAME}/{train_key}",
        content_type="application/x-recordio-protobuf",
    )

    estimator.fit({"train": train_input})
    print(f"  ✅ Entrenamiento completado. Modelo en: {output_path}")
    return estimator


# ═══════════════════════════════════════════════════════════════
# PASO 4 — Inferencia: obtener etiquetas de cluster por registro
# ═══════════════════════════════════════════════════════════════
def paso4_inferencia(estimator, df: pd.DataFrame, feature_cols: list):
    print("\n[PASO 4] Ejecutando predicciones (batch transform)...")

    # Subir datos de inferencia en CSV (SageMaker KMeans acepta text/csv)
    X  = df[feature_cols].values.astype("float32")
    infer_buf = io.BytesIO()
    np.savetxt(infer_buf, X, delimiter=",")
    infer_buf.seek(0)

    infer_key = f"{PREFIX}/inference/input.csv"
    s3.put_object(Bucket=BUCKET_NAME, Key=infer_key,
                  Body=infer_buf.getvalue())

    transformer = estimator.transformer(
        instance_count=1,
        instance_type="ml.c5.xlarge",
        output_path=f"s3://{BUCKET_NAME}/{PREFIX}/inference/output/",
        accept="application/json",
        assemble_with="Line",
    )

    transformer.transform(
        data=f"s3://{BUCKET_NAME}/{infer_key}",
        content_type="text/csv",
        split_type="Line",
        wait=True,
    )

    # Descargar resultados
    result_key = f"{PREFIX}/inference/output/input.csv.out"
    obj    = s3.get_object(Bucket=BUCKET_NAME, Key=result_key)
    lines  = obj["Body"].read().decode().strip().split("\n")
    labels = [int(json.loads(l)["closest_cluster"]) for l in lines]
    df["cluster"] = labels

    print(f"  ✅ Distribución de clusters:\n{pd.Series(labels).value_counts().sort_index()}")
    return df


# ═══════════════════════════════════════════════════════════════
# PASO 5 — Guardar resultados en DynamoDB
# ═══════════════════════════════════════════════════════════════
def paso5_guardar_en_dynamodb(df: pd.DataFrame):
    print("\n[PASO 5] Cargando resultados en DynamoDB...")

    # Crear tabla si no existe
    existing_tables = [t.name for t in dynamodb.tables.all()]
    if DYNAMODB_TABLE not in existing_tables:
        table = dynamodb.create_table(
            TableName=DYNAMODB_TABLE,
            KeySchema=[
                {"AttributeName": "vivienda_id", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "vivienda_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        print(f"  ✅ Tabla DynamoDB creada: {DYNAMODB_TABLE}")
    else:
        table = dynamodb.Table(DYNAMODB_TABLE)
        print(f"  ℹ️  Tabla ya existe: {DYNAMODB_TABLE}")

    # Insertar registros en lotes
    with table.batch_writer() as batch:
        for idx, row in df.iterrows():
            item = {
                "vivienda_id": str(idx),
                "owner":       "rodriveracr",
                "cluster":     int(row["cluster"]),
            }
            # Agregar resto de columnas convirtiendo tipos numpy
            for col in df.columns:
                if col not in item:
                    val = row[col]
                    if isinstance(val, (np.integer,)):
                        val = int(val)
                    elif isinstance(val, (np.floating,)):
                        val = float(val)
                    item[col] = val
            batch.put_item(Item=item)

    print(f"  ✅ {len(df)} registros insertados en DynamoDB")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  PIPELINE VIVIENDAS — rodriveracr (Rodolfo Villarreal Rivera)")
    print("=" * 60)

    raw_key              = paso1_subir_a_s3()
    train_key, df, fcols = paso2_preprocesar_en_ec2(raw_key)
    estimator            = paso3_sagemaker_kmeans(train_key)
    df_with_clusters     = paso4_inferencia(estimator, df, fcols)
    paso5_guardar_en_dynamodb(df_with_clusters)

    print("\n✅ Pipeline completado exitosamente.")
    print(f"   Datos en S3        : s3://{BUCKET_NAME}/{PREFIX}/")
    print(f"   Tabla DynamoDB     : {DYNAMODB_TABLE}")
    print(f"   Propietario        : rodriveracr")
