import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency

def analisar_relacao_com_target(df, coluna, target):
    grupo_0 = df[df[target] == 0][coluna]
    grupo_1 = df[df[target] == 1][coluna]
    
    print(f"\nAnálise descritiva para {coluna}:")
    print(f"Grupo {target} == 0: Média={grupo_0.mean():.2f}, Mediana={grupo_0.median():.2f}")
    print(f"Grupo {target} == 1: Média={grupo_1.mean():.2f}, Mediana={grupo_1.median():.2f}")
    
    stat, p = ttest_ind(grupo_0, grupo_1)
    print(f"Teste t: Estatística={stat:.2f}, p-valor={p:.4f}")
    if p < 0.05:
        print("Há uma diferença significativa entre os grupos.")
    else:
        print("Não há diferença significativa entre os grupos.")
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target, y=coluna, data=df)
    plt.title(f"Distribuição de {coluna} por {target}")
    plt.show()

def analisar_chi_square(df, coluna_categorica, target):
    tabela_contingencia = pd.crosstab(df[coluna_categorica], df[target])
    
    stat, p, dof, expected = chi2_contingency(tabela_contingencia)
    print(f"\nTeste Qui-Quadrado para {coluna_categorica} e {target}:")
    print(f"Estatística={stat:.2f}, p-valor={p:.4f}")
    if p < 0.05:
        print("Há uma associação significativa entre as variáveis.")
    else:
        print("Não há associação significativa entre as variáveis.")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(tabela_contingencia, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Tabela de Contingência: {coluna_categorica} vs {target}")
    plt.show()


def comparacao_valores(df, metrica: str):
    melhor_resultado = df.loc[df[metrica].idxmax()]
    print(f"Melhor resultado para {metrica}:")
    print(melhor_resultado)