import mlflow
from src.config.load_config import load_config
from src.data.load_data import load_csv_data
from src.utils.print_utils import print_with_padding


model_name = "random_forest_detection"
version = 1

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{model_name}/{version}"
)

print_with_padding("READ config.yaml")
config = load_config()
print(config)
# Access model hyperparameters from the configuration
model_params = {model: config["models"][model] for model in config["models"]}
CSIC_FILEPATH = config["data_path"]
print(model_params["RandomForestClassifier"])
print_with_padding("CHARGEMENT DES DONNEES")
print_with_padding("READ CSV")

csic_data = load_csv_data(CSIC_FILEPATH)
print(csic_data.head())
