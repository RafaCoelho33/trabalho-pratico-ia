import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, tree
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.utils import shuffle
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from yellowbrick.classifier import ConfusionMatrix
from mlxtend.evaluate import paired_ttest_5x2cv

warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_csv("Tabela_Doencas_-_Training.csv")


label_data = ['skin_rash',	'continuous_sneezing',	'chills',	'vomiting',	'fatigue',	'cough',	'high_fever',	'breathlessness',	'sweating',	'headache',	'nausea',	'loss_of_appetite',	'pain_behind_the_eyes',	'back_pain',	'swelled_lymph_nodes',	'malaise',	'phlegm',	'throat_irritation',	'redness_of_eyes',	'sinus_pressure',	'runny_nose',	'congestion',	'chest_pain',	'fast_heart_rate',	'loss_of_smell',	'muscle_pain',	'red_spots_over_body',	'rusty_sputum',	'prognosis']

one_hot_data = ['joint_pain']

lb = LabelEncoder() 


for col in label_data:
    data[col] = lb.fit_transform(data[col])

#Use pandas get_dummies for one-hot encoding
data = pd.get_dummies(data, columns=one_hot_data)

imputer = KNNImputer(missing_values=-1)
imputer.fit_transform(data)

scaler = MinMaxScaler()
data[label_data] = scaler.fit_transform(data[label_data])

data[label_data].head()


x = data.drop(columns=["prognosis"])
y = data["prognosis"]

y = lb.fit_transform(y)

X_treino, X_teste, y_treino, y_teste = train_test_split(x, y, test_size = 0.2, random_state = 23)

modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)

#Teste do Modelo
prevision = modelo.predict(X_teste)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(modelo, x, y, cv=kfold, scoring='accuracy')

print(f'Acurácia para cada fold: {scores}')
print(f'Acurácia média: {scores.mean()}')



confusion_matrix(y_teste, prevision)
cm = ConfusionMatrix(modelo)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)

decisionTreeAccuracy = cm.score(X_teste, y_teste)

cm

print(classification_report(y_teste, prevision))
print(accuracy_score(y_teste, prevision))

tree.plot_tree(Y)
plt.show()

# Criar um DataFrame com os rótulos verdadeiros, previstos e os dados de teste
df_resultados = pd.DataFrame({'True': y_teste, 'Predicted': prevision})
df_resultados = pd.concat([df_resultados, X_teste], axis=1)  # Adiciona as features de teste ao DataFrame

# Extrair as regras da árvore
def extrair_regras(arvore, features):
    regras = []
    for node_id in range(arvore.node_count):
        if arvore.children_left[node_id] == arvore.children_right[node_id]:
            # Folha da árvore
            continue
        else:
            regra = (features[arvore.feature[node_id]], arvore.threshold[node_id], node_id)
            regras.append(regra)
    return regras

# Calcular a cobertura para cada regra
n_samples = len(X_teste)
coverage = []
regras = extrair_regras(modelo.tree_, X_teste.columns)

for feature, threshold, node_id in regras:
    mask = (X_teste[feature] <= threshold)
    regra_cobertura = mask.sum() / n_samples
    coverage.append(regra_cobertura)

# Inicializar um dicionário para armazenar as contagens de instâncias corretas para cada regra
instancias_corretas_por_regra = {}

# Iterar sobre as regras
for i, (regra, rule_coverage) in enumerate(zip(regras, coverage)):
    feature, threshold, node_id = regra
    
    # Aplicar a máscara da regra aos dados de teste
    mask = (df_resultados[feature] <= threshold)
    
    # Filtrar as instâncias que atendem à regra
    instancias_regra = df_resultados[mask]
    
    # Contar as instâncias corretas (True Positive)
    instancias_corretas = instancias_regra[instancias_regra['True'] == instancias_regra['Predicted']]
    
    # Calcular a porcentagem de instâncias corretas para a regra
    porcentagem_corretas = len(instancias_corretas) / len(instancias_regra) * 100

    # Armazenar a porcentagem no dicionário
    instancias_corretas_por_regra[f'Regra {i + 1}'] = porcentagem_corretas

    # Imprimir as instâncias corretas para a regra
    print(f"Regra {i + 1}: Se {feature} <= {threshold:.2f} no nó {node_id}")
    print(f"Porcentagem de Instâncias Corretas: {porcentagem_corretas:.2f}%\n")


#Definição dos Hiperparâmetros
rf = RandomForestRegressor(random_state=23, n_jobs=-1)

parameters = {
    'n_estimators': [5, 10, 30, 50],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 50],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3],
    'bootstrap': [True, False],
}

grid = GridSearchCV(rf, parameters, verbose=1, scoring='r2')
grid.fit(X_treino, y_treino)
print('Best Model: ' + str(grid.best_estimator_))


model = RandomForestClassifier(n_estimators=2, max_features='sqrt', random_state = 23, bootstrap = False, max_depth = 50, min_samples_leaf = 3, n_jobs = -1)
model.fit(X_treino, y_treino)
prevision2 = model.predict(X_teste)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')

randomForestAccuracy = scores.mean()

print(f'Acurácia para cada fold: {scores}')
print(f'Acurácia média: {scores.mean()}')

confusion_matrix(y_teste, prevision2)
cm2 = ConfusionMatrix(model)
cm2.fit(X_treino, y_treino)
cm2.score(X_teste, y_teste)

cm

print(classification_report(y_teste, prevision2))
print(accuracy_score(y_teste, prevision2))


for decision_tree in model.estimators_:
   tree.plot_tree(decision_tree)
   plt.show() 

   # Criar um DataFrame com os rótulos verdadeiros, previstos e os dados de teste
   df_resultados = pd.DataFrame({'True': y_teste, 'Predicted': prevision})
   df_resultados = pd.concat([df_resultados, X_teste], axis=1)  # Adiciona as features de teste ao DataFrame

   # Extrair as regras da árvore
   def extrair_regras(arvore, features):
      regras = []
      for node_id in range(arvore.node_count):
         if arvore.children_left[node_id] == arvore.children_right[node_id]:
               # Folha da árvore
               continue
         else:
               regra = (features[arvore.feature[node_id]], arvore.threshold[node_id], node_id)
               regras.append(regra)
      return regras

   # Calcular a cobertura para cada regra
   n_samples = len(X_teste)
   coverage = []
   regras = extrair_regras(modelo.tree_, X_teste.columns)

   for feature, threshold, node_id in regras:
      mask = (X_teste[feature] <= threshold)
      regra_cobertura = mask.sum() / n_samples
      coverage.append(regra_cobertura)

   # Inicializar um dicionário para armazenar as contagens de instâncias corretas para cada regra
   instancias_corretas_por_regra = {}

   # Iterar sobre as regras
   for i, (regra, rule_coverage) in enumerate(zip(regras, coverage)):
      feature, threshold, node_id = regra
      
      # Aplicar a máscara da regra aos dados de teste
      mask = (df_resultados[feature] <= threshold)
      
      # Filtrar as instâncias que atendem à regra
      instancias_regra = df_resultados[mask]
      
      # Contar as instâncias corretas (True Positive)
      instancias_corretas = instancias_regra[instancias_regra['True'] == instancias_regra['Predicted']]
      
      # Calcular a porcentagem de instâncias corretas para a regra
      porcentagem_corretas = len(instancias_corretas) / len(instancias_regra) * 100

      # Armazenar a porcentagem no dicionário
      instancias_corretas_por_regra[f'Regra {i + 1}'] = porcentagem_corretas

      # Imprimir as instâncias corretas para a regra
      print(f"Regra {i + 1}: Se {feature} <= {threshold:.2f} no nó {node_id}")
      print(f"Porcentagem de Instâncias Corretas: {porcentagem_corretas:.2f}%\n")


model = GaussianNB()
naive = model.fit(X_treino, y_treino)

y_pred = model.predict(X_teste)
accuracy = accuracy_score(y_pred, y_teste)

print(classification_report(y_teste, y_pred))
print(accuracy_score(y_teste, y_pred))

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')

naiveBayesAccuracy = scores.mean()

print(f'Acurácia para cada fold: {scores}')
print(f'Acurácia média: {scores.mean()}')

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_teste)

# Obtém a variância explicada pelos componentes principais
explained_variance = pca.explained_variance_ratio_
print("Variância explicada por cada CP:", explained_variance)
cumulative_explained_variance = explained_variance.cumsum()
print("Variância acumulativa explicada:", cumulative_explained_variance)

# Plotagem das previsões
plt.figure(figsize=(8, 6))
sc = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_teste, cmap='viridis', edgecolor='k')
plt.title('Naive Bayes')

# Legenda
classes = sorted(set(y_teste))
legend_labels = ['Menor Chance (0)', 'Maior Chance (1)']
sc = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_teste, cmap='viridis', edgecolor='k')
plt.legend(handles=sc.legend_elements()[0], labels=legend_labels, title='Classes')

# Eixos
plt.xlabel('Primeiro Componente Principal')
plt.ylabel('Segundo Componente Principal')

plt.show()

# Matriz de Confusão
cm = confusion_matrix(y_teste, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5, annot_kws={"size": 16})
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()


modelo1 = DecisionTreeClassifier()
modelo2 = RandomForestClassifier()
modelo3 = GaussianNB()

# Acurácias dos modelos
acuracias_modelos = [decisionTreeAccuracy, randomForestAccuracy, naiveBayesAccuracy]

acuracias_array = np.array(acuracias_modelos)

# Realize o teste t pareado para todas as combinações possíveis de pares de modelos
for i in range(len(acuracias_modelos)):
    for j in range(i + 1, len(acuracias_modelos)):
        t, p = paired_ttest_5x2cv(estimator1=[modelo1, modelo2, modelo3][i], estimator2=[modelo1, modelo2, modelo3][j], X=x, y=y)
        alpha = 0.05

        print(f'Testando Modelo{i + 1} vs Modelo{j + 1}:')
        print('t statistic: %.3f' % t)
        print('alpha ', alpha)
        print('p value: %.3f' % p)

        if p > alpha:
            print("Fail to reject null hypothesis\n")
        else:
            print("Reject null hypothesis\n")
