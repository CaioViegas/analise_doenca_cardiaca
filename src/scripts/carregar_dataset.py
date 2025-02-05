import pandas as pd
import kagglehub
import sys
from typing import Optional
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import RAW_DIR  

def carregamento(caminho_kaggle: Optional[str] = None, caminho_salvamento: Optional[Path] = None, pasta: Optional[str] = None, arquivo: str = None,  salvar: bool = True) -> pd.DataFrame:
    """
    Esta função carrega um conjunto de dados do Kaggle e, opcionalmente, salva-o como um arquivo CSV.

    Parâmetros:
    caminho_kaggle (str, opcional): O caminho para o conjunto de dados do Kaggle. Se não for fornecido, a função irá baixar a versão mais recente.
    caminho_salvamento (Path, opcional): O caminho onde o arquivo CSV será salvo. Se não for fornecido, o arquivo será salvo no diretório de processamento.
    pasta (str, opcional): O nome da pasta dentro do diretório do Kaggle onde o arquivo está localizado. O padrão é None.
    arquivo (str): O nome do arquivo que deve ser carregado.
    salvar (bool, opcional): Indica se o dataset deve ser salvo como um arquivo CSV. O padrão é True.

    Retorna:
    pd.DataFrame: O conjunto de dados carregado.
    """
    path = kagglehub.dataset_download(caminho_kaggle)

    if pasta is not None:
        path = Path(path) / pasta

    arquivo_csv = Path(path) / arquivo
    df = pd.read_csv(arquivo_csv)
    
    if salvar:
        if caminho_salvamento is None:
            caminho_salvamento = RAW_DIR  

        caminho_salvamento.mkdir(parents=True, exist_ok=True)  

        caminho_completo_salvamento = caminho_salvamento / arquivo
        df.to_csv(caminho_completo_salvamento, index=False)

    return df

if __name__ == "__main__":
    df = carregamento(caminho_kaggle="fedesoriano/heart-failure-prediction", arquivo="heart.csv", salvar=True, caminho_salvamento=RAW_DIR)

    print(df.head())
    print(f"{df.columns}")
    print(f"{df.info()}")