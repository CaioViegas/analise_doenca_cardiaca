import pandas as pd
import sys
from pathlib import Path
from carregar_dataset import carregamento
from typing import Optional, Dict
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import REPORTS_DIR, RAW_DIR, PROCESSED_DIR


def tradutor(df: pd.DataFrame, colunas_traducao: Dict[str, str],
             valores_traducao: Dict[str, Dict[str, str]], salvar: bool = True,
             caminho_salvamento: Optional[Path] = None, nome_arquivo: Optional[str] = None) -> pd.DataFrame:
    """
    Traduz as colunas e valores de um DataFrame com base nos dicionários fornecidos.

    Parâmetros:
    df (pd.DataFrame): O DataFrame a ser traduzido.
    colunas_traducao (Dict[str, str]): Dicionário para traduzir os nomes das colunas.
    valores_traducao (Dict[str, Dict[str, str]]): Dicionário para traduzir os valores das colunas.
    salvar (bool, opcional): Indica se o DataFrame traduzido deve ser salvo como um arquivo CSV. O padrão é True.
    caminho_salvamento (Path, opcional): O caminho onde o arquivo CSV será salvo. O padrão é o diretório atual.
    nome_arquivo (str, opcional): O nome do arquivo CSV a ser salvo. O padrão é "dataset_traduzido.csv".

    Retorna:
    pd.DataFrame: O DataFrame traduzido.
    """
    df = df.rename(columns=colunas_traducao)

    for coluna, traducao in valores_traducao.items():
        if coluna in df.columns:
            df[coluna] = df[coluna].replace(traducao)

    if salvar:
        if caminho_salvamento is None:
            caminho_salvamento = Path.cwd()  
        if nome_arquivo is None:
            nome_arquivo = "dataset_traduzido.csv"

        caminho_salvamento.mkdir(parents=True, exist_ok=True)  
        caminho_completo = caminho_salvamento / nome_arquivo
        df.to_csv(caminho_completo, index=False)

    return df

def carregar_dicionario_colunas(arquivo: Path) -> Dict[str, str]:
    """
    Carrega um dicionário de tradução de colunas a partir de um arquivo de texto.

    Parâmetros:
    arquivo (Path): O caminho para o arquivo de texto contendo as traduções das colunas.

    Retorna:
    Dict[str, str]: Dicionário com as traduções dos nomes das colunas.
    """
    colunas_traducao = {}
    with open(arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            chave, valor = linha.strip().split(':')
            colunas_traducao[chave.strip()] = valor.strip()
    return colunas_traducao

def carregar_dicionario_valores(arquivo: Path) -> Dict[str, Dict[str, str]]:
    """
    Carrega um dicionário de tradução de valores a partir de um arquivo de texto.

    Parâmetros:
    arquivo (Path): O caminho para o arquivo de texto contendo as traduções dos valores.

    Retorna:
    Dict[str, Dict[str, str]]: Dicionário com as traduções dos valores das colunas.
    """
    valores_traducao = {}
    with open(arquivo, 'r', encoding='utf-8') as f:
        for linha in f:
            coluna, chave, valor = linha.strip().split(':')
            coluna = coluna.strip()
            chave = chave.strip()
            valor = valor.strip()
            if coluna not in valores_traducao:
                valores_traducao[coluna] = {}
            valores_traducao[coluna][chave] = valor
    return valores_traducao

if __name__ == "__main__":
    caminho_colunas = REPORTS_DIR / "traducao_colunas.txt"
    caminho_valores = REPORTS_DIR / "valores_traducao.txt"

    df = carregamento(caminho_kaggle="fedesoriano/heart-failure-prediction", arquivo="heart.csv", salvar=False, caminho_salvamento=RAW_DIR)

    colunas_traducao = carregar_dicionario_colunas(caminho_colunas)
    valores_traducao = carregar_dicionario_valores(caminho_valores)

    df_traduzido = tradutor(df, colunas_traducao, valores_traducao, salvar=True, caminho_salvamento=PROCESSED_DIR)

    for coluna in df_traduzido.columns:
        print(f"{coluna}: {df_traduzido[coluna].unique()}")