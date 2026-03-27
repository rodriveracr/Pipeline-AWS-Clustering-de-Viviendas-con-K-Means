"""
launch_ec2.py — Lanza la instancia EC2 que ejecuta el pipeline
Autor: rodriveracr (Rodolfo Villarreal Rivera)
"""

import boto3, base64, textwrap

REGION        = "us-east-1"
BUCKET        = "rodriveracr-viviendas-pipeline"
INSTANCE_TYPE = "t3.medium"
# Amazon Linux 2023 en us-east-1 (actualizar según región)
AMI_ID        = "ami-0c02fb55956c7d316"
KEY_PAIR      = "rodriveracr-key"        # ← cambia por tu key pair
IAM_ROLE      = "EC2SageMakerPipelineRole"  # rol con permisos S3 + SageMaker

ec2 = boto3.client("ec2", region_name=REGION)
s3  = boto3.client("s3",  region_name=REGION)

# Subir pipeline.py al bucket antes de lanzar EC2
print("Subiendo código al bucket...")
s3.upload_file("pipeline.py",       BUCKET, "code/pipeline.py")
s3.upload_file("Viviendas_limpio.csv", BUCKET,
               "viviendas/raw/Viviendas_limpio.csv")

# Leer user-data script
with open("ec2_userdata.sh") as f:
    user_data = f.read()

user_data_b64 = base64.b64encode(user_data.encode()).decode()

print(f"Lanzando instancia EC2 ({INSTANCE_TYPE})...")
response = ec2.run_instances(
    ImageId=AMI_ID,
    InstanceType=INSTANCE_TYPE,
    MinCount=1, MaxCount=1,
    KeyName=KEY_PAIR,
    UserData=user_data,                # boto3 acepta texto plano también
    IamInstanceProfile={"Name": IAM_ROLE},
    TagSpecifications=[{
        "ResourceType": "instance",
        "Tags": [
            {"Key": "Name",  "Value": "rodriveracr-pipeline"},
            {"Key": "Owner", "Value": "rodriveracr"},
            {"Key": "Project", "Value": "viviendas-kmeans"},
        ],
    }],
    # Terminar automáticamente cuando el user-data finalice (opcional)
    InstanceInitiatedShutdownBehavior="terminate",
)

instance_id = response["Instances"][0]["InstanceId"]
print(f"✅ Instancia lanzada: {instance_id}")
print(f"   Monitorear: aws ec2 describe-instances --instance-ids {instance_id}")
print(f"   Logs en: s3://{BUCKET}/logs/")
