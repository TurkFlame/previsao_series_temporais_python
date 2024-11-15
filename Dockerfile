# Usa a imagem base do Python 3.10.0
FROM python:3.10.0

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia todos os arquivos do diretório atual para o diretório de trabalho do container
COPY . .

RUN pip install numpy scipy scikit-learn statsmodels pmdarima
RUN pip install --upgrade numpy
# Define o comando padrão para executar o programa
CMD ["python", "./main.py"]
