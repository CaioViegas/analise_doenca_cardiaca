import pandas as pd
import sys
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from category_encoders import BinaryEncoder, TargetEncoder, HashingEncoder
from typing import List, Dict, Optional
from carregar_dataset import carregamento
from traducao_dataset import carregar_dicionario_colunas, carregar_dicionario_valores, tradutor
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, REPORTS_DIR, RAW_DIR


def mapear_binarios(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Mapeia valores binários ('Sim' e 'Não') para valores numéricos (1 e 0).

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas a serem mapeadas.
        colunas (List[str]): Lista de colunas com valores binários a serem transformadas.

    Retorna:
        pd.DataFrame: DataFrame com as colunas mapeadas.
    """
    for coluna in colunas:
        df[coluna] = df[coluna].map({'Sim': 1, 'Nao': 0})
    return df

def aplicar_label_encoding(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Aplica Label Encoding para transformar valores categóricos em valores numéricos inteiros.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas categóricas a serem transformadas.
        colunas (List[str]): Lista de colunas categóricas a serem transformadas.

    Retorna:
        pd.DataFrame: DataFrame com as colunas transformadas.
    """
    label_encoder = LabelEncoder()
    for coluna in colunas:
        df[coluna] = label_encoder.fit_transform(df[coluna])
    return df

def aplicar_ordinal_encoding(df: pd.DataFrame, colunas_categorias: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Aplica Ordinal Encoding para transformar valores categóricos em valores numéricos, respeitando uma ordem definida.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas categóricas a serem transformadas.
        colunas_categorias (Dict[str, List[str]]): Dicionário onde as chaves são os nomes das colunas e os valores são listas com as categorias em ordem desejada.

    Retorna:
        pd.DataFrame: DataFrame com as colunas transformadas.
    """
    for coluna, categorias in colunas_categorias.items():
        ordinal_encoder = OrdinalEncoder(categories=[categorias])
        df[coluna] = ordinal_encoder.fit_transform(df[[coluna]])
    return df

def aplicar_onehot_encoding(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Aplica One-Hot Encoding para criar colunas dummies a partir de valores categóricos.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas categóricas a serem transformadas.
        colunas (List[str]): Lista de colunas categóricas a serem transformadas.

    Retorna:
        pd.DataFrame: DataFrame com as colunas codificadas.
    """
    for coluna in colunas:
        encoder = OneHotEncoder(drop='first')
        valores_codificados = encoder.fit_transform(df[[coluna]]).toarray()
        colunas_codificadas = encoder.get_feature_names_out([coluna])
        df_codificado = pd.DataFrame(valores_codificados, columns=colunas_codificadas, index=df.index)
        df = pd.concat([df, df_codificado], axis=1).drop(columns=[coluna])
    return df

def aplicar_binary_encoding(df: pd.DataFrame, colunas: List[str]) -> pd.DataFrame:
    """
    Aplica Binary Encoding para transformar valores categóricos em representações binárias compactas.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas categóricas a serem transformadas.
        colunas (List[str]): Lista de colunas categóricas a serem transformadas.

    Retorna:
        pd.DataFrame: DataFrame com as colunas transformadas.
    """
    encoder = BinaryEncoder(cols=colunas)
    df = encoder.fit_transform(df)
    return df

def aplicar_target_encoding(df: pd.DataFrame, colunas: List[str], alvo: str) -> pd.DataFrame:
    """
    Aplica Target Encoding para transformar valores categóricos com base na média do valor da variável alvo.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas categóricas a serem transformadas.
        colunas (List[str]): Lista de colunas categóricas a serem transformadas.
        alvo (str): Nome da coluna que representa a variável alvo.

    Retorna:
        pd.DataFrame: DataFrame com as colunas transformadas.
    """
    encoder = TargetEncoder(cols=colunas)
    df = encoder.fit_transform(df, df[alvo])
    return df

def aplicar_hashing_encoding(df: pd.DataFrame, colunas: List[str], n_componentes: int = 8) -> pd.DataFrame:
    """
    Aplica Hashing Encoding para transformar valores categóricos em representações numéricas fixas usando hashing.

    Parâmetros:
        df (pd.DataFrame): DataFrame contendo as colunas categóricas a serem transformadas.
        colunas (List[str]): Lista de colunas categóricas a serem transformadas.
        n_componentes (int): Número de componentes (features) geradas para cada coluna (default: 8).

    Retorna:
        pd.DataFrame: DataFrame com as colunas transformadas.
    """
    encoder = HashingEncoder(cols=colunas, n_components=n_componentes)
    df = encoder.fit_transform(df)
    return df

def executar_transformacoes(
    df: pd.DataFrame, 
    colunas_binarias: Optional[List[str]] = None, colunas_label: Optional[List[str]] = None, colunas_ordinal: Optional[Dict[str, List[str]]] = None, colunas_onehot: Optional[List[str]] = None,
    colunas_binary: Optional[List[str]] = None, colunas_target: Optional[List[str]] = None, colunas_hashing: Optional[List[str]] = None, alvo: Optional[str] = None,
    n_componentes_hashing: Optional[int] = 8, salvar: bool = True, caminho_salvamento: Optional[str] = None) -> pd.DataFrame:
    """
    Executa uma série de transformações de codificação em colunas categóricas de um DataFrame.

    Parâmetros:
        ...
        colunas_hashing (Optional[List[str]]): Lista de colunas para aplicação de Hashing Encoding.
        n_componentes_hashing (Optional[int]): Número de componentes para Hashing Encoding (default: 8).
        ...
    """
    if colunas_binarias:
        df = mapear_binarios(df, colunas_binarias)
    if colunas_label:
        df = aplicar_label_encoding(df, colunas_label)
    if colunas_ordinal:
        df = aplicar_ordinal_encoding(df, colunas_ordinal)
    if colunas_onehot:
        df = aplicar_onehot_encoding(df, colunas_onehot)
    if colunas_binary:
        df = aplicar_binary_encoding(df, colunas_binary)
    if colunas_target and alvo:
        df = aplicar_target_encoding(df, colunas_target, alvo)
    if colunas_hashing:
        df = aplicar_hashing_encoding(df, colunas_hashing, n_componentes_hashing)
    if salvar:
        if caminho_salvamento is None:
            caminho_salvamento = PROCESSED_DIR  
        caminho_salvamento.mkdir(parents=True, exist_ok=True)  
        df.to_csv(caminho_salvamento / "dataset_transformado.csv", index=False)
    return df

if __name__ == '__main__':
    caminho_traducao_colunas = REPORTS_DIR / "traducao_colunas.txt"
    caminho_traducao_valores = REPORTS_DIR / "valores_traducao.txt"

    df = carregamento(caminho_kaggle="fedesoriano/heart-failure-prediction", arquivo="heart.csv", salvar=False, caminho_salvamento=RAW_DIR)

    colunas_traducao = carregar_dicionario_colunas(caminho_traducao_colunas)
    valores_traducao = carregar_dicionario_valores(caminho_traducao_valores)

    df = tradutor(df, colunas_traducao, valores_traducao, salvar=False)

    df = aplicar_label_encoding(df, ['Sexo'])

    df = mapear_binarios(df, ['AnginaExercicio'])

    df = aplicar_onehot_encoding(df, ['TipoDorTorax', 'EletrocardiogramaRepouso', 'InclinacaoST'])

    print(df.info())

    for coluna in df.columns:
        print(f"{coluna}: {df[coluna].unique()}")
