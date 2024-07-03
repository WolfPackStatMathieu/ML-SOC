FROM ubuntu:22.04
WORKDIR ${HOME}/ML-SOC
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python3", "main.py"]

# Copier les scripts et le code source
COPY . .

# Exécuter train.py pendant la construction de l'image
# RUN python3 train.py



# CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]