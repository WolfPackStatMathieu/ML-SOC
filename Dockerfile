# Utiliser l'image de base Ubuntu
FROM ubuntu:22.04


# Définir le répertoire de travail
WORKDIR ${HOME}/ML-SOC

# Mettre à jour apt-get et installer les dépendances du système
RUN apt-get -y update && \
    apt-get install -y python3-pip curl

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . .

# Exposer le port que l'application utilise
EXPOSE 8000

# Définir la commande par défaut pour exécuter l'application
CMD ["python3", "main.py"]
