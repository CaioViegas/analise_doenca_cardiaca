import pandas as pd
import numpy as np
import sys
import joblib
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import PROCESSED_DIR

def preparar_dados(df: pd.DataFrame, coluna_alvo: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepara os dados para treinamento.
    
    Args:
        df: DataFrame contendo os dados.
        coluna_alvo: Nome da coluna alvo.
    
    Returns:
        X: DataFrame com as features.
        y: Série com a variável alvo.
    """
    X = df.drop(columns=[coluna_alvo], axis=1)
    y = df[coluna_alvo]
    return X, y

def criar_pipeline() -> Pipeline:
    """
    Cria um pipeline com RobustScaler e LogisticRegression.
    
    Returns:
        Pipeline: Pipeline configurado.
    """
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('lr', LogisticRegression(random_state=101))
    ])
    return pipeline

def calcular_metricas(y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calcula métricas de classificação.
    
    Args:
        y_test: Valores reais da variável alvo.
        y_pred: Valores preditos pelo modelo.
    
    Returns:
        Dict: Dicionário com as métricas calculadas.
    """
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred) if len(np.unique(y_test)) == 2 else roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')
    matriz_confusao = confusion_matrix(y_test, y_pred)
    
    resultados = {
        'Acurácia': acuracia,
        'Precisão': precisao,
        'Recall': recall,
        'F1-score': f1,
        'ROC-AUC': roc_auc,
        'Matriz de Confusão': matriz_confusao
    }
    return resultados

def otimizar_modelo(X_train: pd.DataFrame, y_train: pd.Series, pipeline: Pipeline,
    parametros: Dict[str, Any], metodo: str = 'grid') -> BaseEstimator:
    """
    Otimiza o modelo usando GridSearchCV ou RandomizedSearchCV.
    
    Args:
        X_train: DataFrame com as features de treinamento.
        y_train: Série com a variável alvo de treinamento.
        pipeline: Pipeline do modelo.
        parametros: Dicionário de parâmetros para otimização.
        metodo: Método de otimização ('grid' ou 'random').
    
    Returns:
        BaseEstimator: Melhor modelo encontrado.
    """
    if metodo == 'grid':
        busca = GridSearchCV(pipeline, param_grid=parametros, cv=5, scoring='accuracy', n_jobs=-1)
    elif metodo == 'random':
        busca = RandomizedSearchCV(pipeline, param_distributions=parametros, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10, random_state=101)
    
    busca.fit(X_train, y_train)
    print(f"Melhores parâmetros encontrados ({metodo}): {busca.best_params_}")
    print(f"Melhor score de validação cruzada ({metodo}): {busca.best_score_:.4f}")
    return busca.best_estimator_

def avaliar_modelo_cv(modelo: BaseEstimator, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> None:
    """
    Avalia o modelo usando validação cruzada.
    
    Args:
        modelo: Modelo a ser avaliado.
        X: DataFrame com as features.
        y: Série com a variável alvo.
        cv: Número de folds para validação cruzada.
    """
    scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')
    print(f"Acurácia média com validação cruzada ({cv} folds): {scores.mean():.4f} (± {scores.std():.4f})")

def comparar_modelos(
    modelo_grid: BaseEstimator, modelo_random: BaseEstimator,
    X_test: pd.DataFrame, y_test: pd.Series, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> None:
    """
    Compara os resultados de dois modelos, incluindo métricas de validação cruzada.
    
    Args:
        modelo_grid: Modelo otimizado com GridSearchCV.
        modelo_random: Modelo otimizado com RandomizedSearchCV.
        X_test: DataFrame com as features de teste.
        y_test: Série com a variável alvo de teste.
        X: DataFrame com todas as features (para validação cruzada).
        y: Série com todas as variáveis alvo (para validação cruzada).
        cv: Número de folds para validação cruzada.
    """
    y_pred_grid = modelo_grid.predict(X_test)
    y_pred_random = modelo_random.predict(X_test)
    
    resultados_grid = calcular_metricas(y_test, y_pred_grid)
    resultados_random = calcular_metricas(y_test, y_pred_random)
    
    resultados_grid['Matriz de Confusão'] = str(resultados_grid['Matriz de Confusão'])
    resultados_random['Matriz de Confusão'] = str(resultados_random['Matriz de Confusão'])
    
    scores_grid = cross_val_score(modelo_grid, X, y, cv=cv, scoring='accuracy')
    scores_random = cross_val_score(modelo_random, X, y, cv=cv, scoring='accuracy')
    
    resultados_grid['Validação Cruzada (Acurácia)'] = f"{scores_grid.mean():.4f} (± {scores_grid.std():.4f})"
    resultados_random['Validação Cruzada (Acurácia)'] = f"{scores_random.mean():.4f} (± {scores_random.std():.4f})"
    
    df_comparacao = pd.DataFrame({
        'GridSearchCV': resultados_grid,
        'RandomizedSearchCV': resultados_random
    }).T
    
    print("\nComparação de métricas:")
    print(df_comparacao[['Acurácia', 'Precisão', 'Recall', 'F1-score', 'ROC-AUC', 'Validação Cruzada (Acurácia)', 'Matriz de Confusão']])

def executar_pipeline(df: pd.DataFrame, coluna_alvo: str, parametros: Dict[str, Any], caminho_salvar: Optional[Union[str, None]] = None) -> None:
    """
    Executa o pipeline de treinamento e avaliação do modelo.
    
    Args:
        df: DataFrame contendo os dados.
        coluna_alvo: Nome da coluna alvo.
        parametros: Dicionário de parâmetros para otimização.
        caminho_salvar: Caminho para salvar o modelo (opcional).
    """
    X, y = preparar_dados(df, coluna_alvo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    
    pipeline = criar_pipeline()
    
    print("Otimizando com GridSearchCV...")
    modelo_grid = otimizar_modelo(X_train, y_train, pipeline, parametros, metodo='grid')
    print("\nOtimizando com RandomizedSearchCV...")
    modelo_random = otimizar_modelo(X_train, y_train, pipeline, parametros, metodo='random')
    
    print("\nComparando resultados:")
    comparar_modelos(modelo_grid, modelo_random, X_test, y_test, X, y)
    
    if caminho_salvar:
        melhor_modelo = modelo_grid if accuracy_score(y_test, modelo_grid.predict(X_test)) >= accuracy_score(y_test, modelo_random.predict(X_test)) else modelo_random
        joblib.dump(melhor_modelo, caminho_salvar)
        print(f"\nMelhor modelo salvo em {caminho_salvar}")
        
if __name__ == "__main__":
    param_grid = {
        'lr__penalty': ['l1', 'l2', 'elasticnet', None],
        'lr__C': [0.01, 0.1, 1, 10, 100],  
        'lr__solver': ['liblinear', 'lbfgs', 'saga'],
        'lr__max_iter': [100, 200, 500]
    }

    caminho_dataset = PROCESSED_DIR / 'dataset_transformado.csv'
    df = pd.read_csv(caminho_dataset)

    df = df[df['Colesterol'] != 0]
    
    executar_pipeline(df, 'DoencaCardiaca', param_grid, caminho_salvar=None)