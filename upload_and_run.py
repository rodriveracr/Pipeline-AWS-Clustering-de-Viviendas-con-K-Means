"""
upload_and_run.py

Sube el CSV local a S3 y ejecuta `pipeline.py` con los parámetros proporcionados.
Requiere que `aws configure` esté hecho o que existan variables de entorno AWS.
"""
import argparse
import boto3
import os
import subprocess
import sys

parser = argparse.ArgumentParser(description="Subir CSV a S3 y ejecutar pipeline.py")
parser.add_argument("--local-csv", default="Viviendas_limpio.csv")
parser.add_argument("--bucket", default=None, help="Nombre del bucket S3 (si omite, usa el del pipeline.py)")
parser.add_argument("--prefix", default=None)
parser.add_argument("--role-arn", default=None, help="ARN del rol para SageMaker (opcional)")
parser.add_argument("--skip-sagemaker", action="store_true", help="Subir y preprocesar, pero no ejecutar SageMaker")
args = parser.parse_args()

if not os.path.exists(args.local_csv):
    print(f"ERROR: archivo no encontrado: {args.local_csv}")
    sys.exit(2)

# Determinar bucket/prefix: si no se pasan, usar valores por defecto en pipeline.py
bucket = args.bucket
prefix = args.prefix

# Cargar boto3 y subir
session = boto3.Session()
s3 = session.client('s3')

if bucket is None:
    print("Nota: no se proporcionó --bucket; usando el bucket por defecto en pipeline.py. Ejecutando pipeline con --upload-only y dejando que pipeline cree el bucket si es necesario.")
    # Ejecutar pipeline.py con --local-csv y --upload-only
    cmd = [sys.executable, "pipeline.py", "--local-csv", args.local_csv, "--upload-only"]
    if args.role_arn:
        cmd += ["--role-arn", args.role_arn]
    print("Ejecutando:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Subida realizada por pipeline.py (modo --upload-only).")
else:
    s3_key = f"{prefix or 'viviendas'}/raw/{os.path.basename(args.local_csv)}"
    print(f"Subiendo {args.local_csv} → s3://{bucket}/{s3_key} ...")
    s3.upload_file(args.local_csv, bucket, s3_key)
    print("Subida completada.")
    # Si no piden ejecución posterior, terminar
    if args.skip_sagemaker:
        print("Fin (modo --skip-sagemaker).")
        sys.exit(0)
    # Ejecutar pipeline.py apuntando al bucket/prefix
    cmd = [sys.executable, "pipeline.py", "--local-csv", args.local_csv, "--bucket", bucket, "--prefix", prefix or 'viviendas']
    if args.role_arn:
        cmd += ["--role-arn", args.role_arn]
    print("Ejecutando pipeline:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Ejecución de pipeline.py finalizada.")
