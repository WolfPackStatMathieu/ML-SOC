# Utiliser l'image de base Ubuntu
FROM inseefrlab/onyxia-python-minimal:py3.10.9


# Définir le répertoire de travail
WORKDIR /ML-SOC

# Préparer l'environnement et installer les dépendances
RUN apt-get clean && apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3.10 \
    python3.10-distutils \
    python3-pip \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Installer pip pour la version spécifique de Python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . .

# Exposer le port que l'application utilise
EXPOSE 8000

# Définir la commande par défaut pour exécuter l'application
CMD ["python3.10", "main.py"]
