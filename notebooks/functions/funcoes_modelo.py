import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def treinar_modelo_classificacao(modelo, resultados, y_train, y_test, dados_escalados):
    for nome, scaled in dados_escalados.items():
        X_train_scaled = scaled['X_train']
        X_test_scaled = scaled['X_test']
        
        modelo.fit(X_train_scaled, y_train)
        
        y_pred = modelo.predict(X_test_scaled)
        y_pred_proba = modelo.predict_proba(X_test_scaled)  
        
        acuracia = accuracy_score(y_test, y_pred)
        precisao = precision_score(y_test, y_pred, average='weighted')  
        recall = recall_score(y_test, y_pred, average='weighted')      
        f1 = f1_score(y_test, y_pred, average='weighted')              
        
        if len(np.unique(y_test)) == 2:  
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:  
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        
        matriz_confusao = confusion_matrix(y_test, y_pred)
        
        resultados[nome] = {
            'Acurácia': acuracia,
            'Precisão': precisao,
            'Recall': recall,
            'F1-score': f1,
            'ROC-AUC': roc_auc,
            'Matriz de Confusão': matriz_confusao,
        }

def apresentar_resultados(modelo, y_train, y_test, dados_escalados):
    dicionario_resultados = {}
    treinar_modelo_classificacao(modelo, dicionario_resultados, y_train, y_test, dados_escalados)
    df_modelo = pd.DataFrame(dicionario_resultados).T
    return df_modelo