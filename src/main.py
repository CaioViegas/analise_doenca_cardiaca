import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "scripts"))
from carregar_dataset import carregamento
from traducao_dataset import carregar_dicionario_colunas, carregar_dicionario_valores, tradutor
from transformacao_dataset import executar_transformacoes
from otimizador_modelo import executar_pipeline
from typing import Optional, Dict, Any
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR, REPORTS_DIR, RAW_DIR, MODELS_DIR

def main(caminho_kaggle: Optional[str], caminho_salvamento_original: Optional[str], caminho_salvamento_traducao: Optional[str], caminho_salvamento_transformado: Optional[str], caminho_salvamento_modelo: Optional[str], pasta: Optional[str] = None, arquivo: str = None, transformacoes_config: dict = None, dicionario_colunas: Optional[str] = None, dicionario_valores: Optional[str] = None, coluna_alvo: str = None, parametros: Dict[str, Any] = None):
    
    df = carregamento(caminho_kaggle=caminho_kaggle, arquivo=arquivo, salvar=True, caminho_salvamento=caminho_salvamento_original)

    colunas_traducao = carregar_dicionario_colunas(dicionario_colunas)
    valores_traducao = carregar_dicionario_valores(dicionario_valores)
    
    df = tradutor(df, colunas_traducao, valores_traducao, caminho_salvamento=caminho_salvamento_traducao, salvar=True)

    df = executar_transformacoes(df, colunas_binarias=transformacoes_config.get("colunas_binarias"),
        colunas_label=transformacoes_config.get("colunas_label"), colunas_onehot=transformacoes_config.get("colunas_onehot"), salvar=True, caminho_salvamento=caminho_salvamento_transformado)

    df = executar_pipeline(df, coluna_alvo, parametros, caminho_salvar=caminho_salvamento_modelo)

if __name__ == "__main__":
    transformacoes_config = {
        "colunas_label": ['Sexo'], "colunas_binarias": ['AnginaExercicio'], "colunas_onehot": ['TipoDorTorax', 'EletrocardiogramaRepouso', 'InclinacaoST']
    }

    param_grid = {
        'lr__penalty': ['l1', 'l2', 'elasticnet', None],
        'lr__C': [0.01, 0.1, 1, 10, 100],  
        'lr__solver': ['liblinear', 'lbfgs', 'saga'],
        'lr__max_iter': [100, 200, 500]
    }

    caminho_do_kaggle = "fedesoriano/heart-failure-prediction"
    arquivo_kaggle = "heart.csv"
    caminho_traducao_colunas = REPORTS_DIR / "traducao_colunas.txt"
    caminho_traducao_valores = REPORTS_DIR / "valores_traducao.txt"
    caminho_raw = RAW_DIR
    caminho_processed = PROCESSED_DIR
    caminho_modelo = MODELS_DIR / "modelo_otimizado.joblib"

    main(caminho_kaggle=caminho_do_kaggle, arquivo=arquivo_kaggle, transformacoes_config=transformacoes_config, dicionario_colunas=caminho_traducao_colunas, 
         dicionario_valores=caminho_traducao_valores, caminho_salvamento_original=caminho_raw, caminho_salvamento_traducao=PROCESSED_DIR, caminho_salvamento_transformado=PROCESSED_DIR,
         coluna_alvo='DoencaCardiaca', caminho_salvamento_modelo=caminho_modelo, parametros=param_grid)
    