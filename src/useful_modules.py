import scipy.stats as stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, normaltest
import statsmodels.api as sm
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analise_automatica(df, target_col='DefaultStatus', significance_level=0.05):

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 10))
    
    print(f"Analisando o atributo: {target_col}")
    print("="*60)
    
    for col in categorical_cols:
        print(f"\nAnálise do atributo categórico: {col}")
        print("-"*60)
        
        cross_tab = pd.crosstab(df[col], df[target_col], normalize='index')
        print(f"Proporcão de {target_col} por {col}:")
        print(cross_tab)
        
        cross_tab.plot(kind='bar', stacked=True)
        plt.title(f'Proporção por {col}')
        plt.ylabel('Proporção')
        plt.show()
        
        contingency_table = pd.crosstab(df[col], df[target_col])
        if contingency_table.size >= 2:
            if np.all(contingency_table >= 5): 
                chi2, p, _, _ = chi2_contingency(contingency_table)
                print(f"p-valor do teste Qui^2: {p}")
                if p < significance_level:
                    print(f"Relação significativa encontrada entre {col} e {target_col} (p < {significance_level}).")
                else:
                    print(f"Sem Relação significativa encontrada entre {col} e {target_col} (p >= {significance_level}).")
            else:
                print("Aviso: Validade do teste Qui^2 questionável pois algumas frequências esperadas são menores que 5")
        else:
            print("Qui^2 não aplicável (categorias insuficientes).")
    
    for col in numerical_cols:
        print(f"\nAnalisando o atributo numérico: {col}")
        print("-"*60)
        
        sns.boxplot(x=target_col, y=col, data=df)
        plt.title(f'Distribuição de {target_col} por {col}')
        plt.show()
        
        group0 = df[df[target_col] == 0][col].dropna()
        group1 = df[df[target_col] == 1][col].dropna()
        
        print(f"Média de {col} Para {target_col}=0: {group0.mean()}")
        print(f"Média de {col} Para {target_col}=1: {group1.mean()}")
        
        _, p_normal0 = normaltest(group0)
        _, p_normal1 = normaltest(group1)
        normal0 = p_normal0 > significance_level
        normal1 = p_normal1 > significance_level
        
        print(f"p-valor para o teste de normalidade quando {target_col} = 0: {p_normal0}")
        print(f"p-valor para o teste de normalidade quando {target_col} = 1: {p_normal1}")
        
        if normal0 and normal1:
            t_stat, t_p_value = ttest_ind(group0, group1, equal_var=False)
            print(f"p-valor pelo teste t: {t_p_value}")
            if t_p_value < significance_level:
                print(f"Diferença entre as médias significativa entre grupos de {target_col} para {col} (p < {significance_level}).")
            else:
                print(f"Diferença não significativa entre as médias entre grupos de {target_col} para {col} (p >= {significance_level}).")
        else:
            print("Teste t não aplicável (normalidade não pode ser assumida)")
        
        mw_stat, mw_p_value = mannwhitneyu(group0, group1)
        print(f"p-valor do teste Mann-Whitney: {mw_p_value}")
        if mw_p_value < significance_level:
            print(f"Diferença entre as distribuições significativa entre grupos de {target_col} para {col} (p < {significance_level}).")
        else:
            print(f"Diferença não significativa entre as distribuições entre grupos de {target_col} para {col} (p >= {significance_level}).")
    
    print("\nMatriz de correlação entre as variáveis numéricas (considerando coeficiente de spearman):")
    print("-"*60)
    corr_matrix = df[numerical_cols].corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.show()