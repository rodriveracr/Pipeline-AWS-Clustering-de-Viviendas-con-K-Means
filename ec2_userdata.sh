#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# EC2 User Data Script — rodriveracr (Rodolfo Villarreal Rivera)
# Ejecutar en: Amazon Linux 2 / Ubuntu 22.04
# Instance type sugerida: t3.medium
# ─────────────────────────────────────────────────────────────────

set -e

echo "=== [1/5] Actualizando paquetes ==="
sudo apt-get update -y || sudo yum update -y

echo "=== [2/5] Instalando Python y pip ==="
sudo apt-get install -y python3 python3-pip || \
    sudo yum install -y python3 python3-pip

echo "=== [3/5] Instalando dependencias del pipeline ==="
pip3 install --upgrade \
    boto3 \
    pandas \
    numpy \
    scikit-learn \
    sagemaker \
    pyarrow

echo "=== [4/5] Descargando pipeline desde S3 ==="
BUCKET="rodriveracr-viviendas-pipeline"
aws s3 cp s3://${BUCKET}/code/pipeline.py      /home/ec2-user/pipeline.py
aws s3 cp s3://${BUCKET}/viviendas/raw/Viviendas_limpio.csv \
             /home/ec2-user/Viviendas_limpio.csv

echo "=== [5/5] Ejecutando pipeline ==="
cd /home/ec2-user
python3 pipeline.py 2>&1 | tee pipeline_output.log

# Subir log de ejecución a S3
aws s3 cp pipeline_output.log \
    s3://${BUCKET}/logs/pipeline_output_$(date +%Y%m%d_%H%M%S).log

echo "=== Pipeline completado. Log subido a S3 ==="
