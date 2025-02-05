from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from sklearn.linear_model import LogisticRegression

def aplicar_undersampling(X, y):
    """
    Aplica undersampling para reduzir a classe majoritária.
    Retorna X e y balanceados.
    """
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    return X_resampled, y_resampled

def aplicar_oversampling(X, y):
    """
    Aplica oversampling para aumentar a classe minoritária.
    Retorna X e y balanceados.
    """
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    return X_resampled, y_resampled

def aplicar_smote(X, y):
    """
    Aplica SMOTE para criar instâncias sintéticas da classe minoritária.
    Retorna X e y balanceados.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def aplicar_smoteenn(X, y):
    """
    Combina SMOTE com undersampling para balancear as classes.
    Retorna X e y balanceados.
    """
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    return X_resampled, y_resampled

def aplicar_pesos_classes(X, y):
    """
    Ajusta os pesos das classes durante o treinamento do modelo.
    Retorna o modelo treinado com pesos balanceados.
    """
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)
    return model