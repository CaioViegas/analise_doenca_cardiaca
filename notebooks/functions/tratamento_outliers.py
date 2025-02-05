from scipy.stats.mstats import winsorize
from sklearn.impute import SimpleImputer

def identificar_outliers_iqr(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)]

def aplicar_winsorization(df, coluna, limites=(0.05, 0.05)):
    df[coluna] = winsorize(df[coluna], limits=limites)
    return df

def aplicar_trimming(df, coluna):
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    return df[(df[coluna] >= limite_inferior) & (df[coluna] <= limite_superior)]

def aplicar_imputation(df, coluna):
    imputer = SimpleImputer(strategy='median')
    df[coluna] = imputer.fit_transform(df[[coluna]])
    return df