import os
import sys
import s3fs
import joblib

# Ajouter le chemin vers le module `src`
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Vérification si le module `src` est accessible
try:
    import src
except ImportError as e:
    print(f"Failed to import src module: {e}")

# Étape 5: Chargement du pipeline de prétraitement sauvegardé
try:
    complete_pipeline = joblib.load('complete_preprocessor_pipeline.pkl')
except Exception as e:
    print(f"Failed to load the pipeline: {e}")
    sys.exit(1)

# Étape 6: Exportation du pipeline sauvegardé vers S3 en utilisant s3fs avec endpoint spécifique
BUCKET_OUT = "mthomassin/preprocessor"
FILE_KEY_OUT_S3 = "preprocessor.pkl"
FILE_PATH_OUT_S3 = os.path.join(BUCKET_OUT, FILE_KEY_OUT_S3)

fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"},
    anon=True
)
#  mc cp complete_preprocessor_pipeline.pkl s3/mthomassin/preprocessor/complete_preprocessor_pipeline.pkl
try:
    with fs.open(f"s3://{FILE_PATH_OUT_S3}", 'wb') as file_out:
        joblib.dump(complete_pipeline, file_out)
    print(f"Pipeline successfully uploaded to s3://{FILE_PATH_OUT_S3}")
except Exception as e:
    print(f"Failed to upload pipeline to S3: {e}")
