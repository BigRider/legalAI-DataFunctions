from pyspark.sql import DataFrame
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import pandas as pd


def carregar_palavras_baixo_calao(caminho_csv: str, coluna_palavra: str) -> list:
    """
    Carrega uma lista de palavras de baixo calão a partir de um arquivo CSV.

    :param caminho_csv: Caminho para o arquivo CSV contendo as palavras de baixo calão.
    :param coluna_palavra: Nome da coluna no CSV que contém as palavras ofensivas.
    :return: Lista de palavras ofensivas.
    """
    # Carrega o arquivo CSV em um DataFrame Pandas
    palavras_df = pd.read_csv(caminho_csv)
    return palavras_df[coluna_palavra].dropna().str.strip().tolist()


def limpar_texto(texto: str, palavras_proibidas: set) -> str:
    """
    Remove palavras de baixo calão de um texto.

    :param texto: Texto a ser processado.
    :param palavras_proibidas: Conjunto de palavras proibidas.
    :return: Texto com as palavras proibidas removidas.
    """
    if not texto:
        return texto
    palavras = texto.split()
    palavras_limpas = [
        palavra if palavra.lower() not in palavras_proibidas else '' 
        for palavra in palavras
    ]
    return ' '.join(filter(None, palavras_limpas))


def remover_palavras_baixo_calao_csv(df: DataFrame, campo: str, caminho_csv: str, coluna_palavra: str) -> DataFrame:
    """
    Remove palavras de baixo calão de um campo específico de um DataFrame PySpark usando palavras carregadas de um CSV.

    :param df: DataFrame PySpark contendo os dados.
    :param campo: Nome do campo onde as palavras serão removidas.
    :param caminho_csv: Caminho para o arquivo CSV contendo palavras de baixo calão.
    :param coluna_palavra: Nome da coluna no CSV que contém as palavras ofensivas.
    :return: DataFrame atualizado com as palavras de baixo calão removidas.
    """
    # Carrega as palavras de baixo calão do CSV
    palavras_proibidas = set(carregar_palavras_baixo_calao(caminho_csv, coluna_palavra))
    
    # Define a UDF para limpar o texto
    limpar_udf = udf(lambda texto: limpar_texto(texto, palavras_proibidas), StringType())
    
    # Aplica a UDF ao campo especificado
    return df.withColumn(campo, limpar_udf(col(campo)))


