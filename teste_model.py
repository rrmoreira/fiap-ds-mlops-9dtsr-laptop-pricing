import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import json
import os
import shutil

# Configura o URI do rastreamento do MLflow
mlflow.set_tracking_uri("https://dagshub.com/rrmoreira/fiap-ds-mlops-9dtsr-laptop-pricing.mlflow")

# Configurações
model_name = "laptop-pricing-model-brl"
artifact_relative_path = "model/model.pkl"
client = MlflowClient()

# 1. Buscar todas as versões do modelo
versions = client.search_model_versions(f"name='{model_name}'")

# 2. Obter a versão mais recente
latest = max(versions, key=lambda v: int(v.version))

# 3. Baixar o artefato
download_path = client.download_artifacts(
    run_id=latest.run_id,
    path=artifact_relative_path,
    dst_path="models"
)
# Move model.pkl to the root of 'models' if it's inside a subfolder
src_path = os.path.join(download_path)
dst_path = os.path.join("models", "model.pkl")
# If downloaded path is not already 'models/model.pkl', move it
if src_path != dst_path:
    shutil.move(src_path, dst_path)
    # Remove the now-empty 'models/model' folder if it exists
    model_folder = os.path.join("models", "model")
    if os.path.isdir(model_folder):
        shutil.rmtree(model_folder)

    
    
model_metadata = {
    "model_name": model_name,
    "version": latest.version,
    "run_id": latest.run_id,
    "source": latest.source,
    "downloaded_at": datetime.now().isoformat()
}

with open("models/model_metadata.json", "w") as f:
    json.dump(model_metadata, f, indent=2)