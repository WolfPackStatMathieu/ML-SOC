# Utiliser l'image de base avec Python 3.10.9
FROM inseefrlab/onyxia-python-minimal:py3.10.9

# Définir le répertoire de travail
WORKDIR /ML-SOC

# Préparer l'environnement et installer les dépendances
RUN apt-get clean && apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . .

# Exposer le port que l'application utilise
EXPOSE 8000

# Définir la commande par défaut pour exécuter l'application
CMD ["python3", "main.py"]
