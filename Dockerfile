FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /ML-SOC

# Mettre à jour apt-get et installer les dépendances du système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt en premier pour tirer parti du cache Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le reste du projet
COPY . .

# Définir la commande par défaut pour exécuter le script principal
CMD ["python3", "main.py"]

# Vous pouvez décommenter cette ligne si vous avez besoin d'exécuter train.py pendant la construction
# RUN python3 train.py
