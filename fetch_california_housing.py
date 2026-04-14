from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Загружаем датасет 
california = fetch_california_housing(as_frame=True)
data = california.frame

#смотрим первые 5 строк
print(data.head())

#смотрим сколько и где пропущены значения
print(data.isna().sum()) 

y = data['MedHouseVal']
X = data.drop('MedHouseVal', axis=1)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler # для масштабирования данных

# масштабируем данные
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучаем модель
model_lin = LinearRegression()
model_lin.fit(X_train_scaled, y_train)

from sklearn.metrics import mean_squared_error, r2_score

test = model_lin.predict(X_test_scaled)
train = model_lin.predict(X_train_scaled)

#считаем параметры rmse и r2, рисуем гистограмму
rmse_train = np.sqrt(mean_squared_error(y_train, train))
rmse_test = np.sqrt(mean_squared_error(y_test, test))

r2_train = r2_score(y_train, train)
r2_test = r2_score(y_test, test)

print(f'RMSE_train:{rmse_train} | RMSE_test:{rmse_test} | R2_train:{r2_train} | R2_test:{r2_test}')

plt.figure(figsize=(8,6))
plt.hist(y, bins=40, color='seagreen', edgecolor='w')
plt.title('Стоимость домов')
plt.xlabel('Стоимость')
plt.ylabel('Количество')
plt.show()

data = data[data.MedHouseVal < 5] # убираем значения больше 5

#считаем корреляционную матрицу

corr = data.corr(method='spearman')

plt.figure(figsize=(8,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()

data = data.drop(['AveBedrms'], axis=1) # удаляем только AveBedrms т.к он мультиколлинеарный признак

# Повторяю обучение и рассчет исходя из новых данных

y_2 = data['MedHouseVal']
X_2 = data.drop('MedHouseVal', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучаем модель
model_lin = LinearRegression()
model_lin.fit(X_train_scaled, y_train)

test = model_lin.predict(X_test_scaled)
train = model_lin.predict(X_train_scaled)

#считаем параметры rmse и r2
rmse_train = np.sqrt(mean_squared_error(y_train, train))
rmse_test = np.sqrt(mean_squared_error(y_test, test))

r2_train = r2_score(y_train, train)
r2_test = r2_score(y_test, test)

print('Показатели после удаления AveBedrms ')
print(f'RMSE_train:{rmse_train} | RMSE_test:{rmse_test} | R2_train:{r2_train} | R2_test:{r2_test}')

#считаем и удаляем выбросы
Q1 = X_2.quantile(0.25)
Q3 = X_2.quantile(0.75)
IQR = Q3 - Q1

lower_b = Q1 - 1.5 * IQR
upper_b = Q3 + 1.5 * IQR

outliers = ~((X_2 < lower_b) | (X_2 > upper_b)).any(axis=1)

X_clean = X_2[outliers]
y_clean = y_2[outliers]

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучаем модель
model_lin = LinearRegression()
model_lin.fit(X_train_scaled, y_train)


test = model_lin.predict(X_test_scaled)
train = model_lin.predict(X_train_scaled)

#считаем параметры rmse и r2
rmse_train = np.sqrt(mean_squared_error(y_train, train))
rmse_test = np.sqrt(mean_squared_error(y_test, test))

r2_train = r2_score(y_train, train)
r2_test = r2_score(y_test, test)

print('Показатели после удаления выбросов ')
print(f'RMSE_train:{rmse_train} | RMSE_test:{rmse_test} | R2_train:{r2_train} | R2_test:{r2_test}')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.hist(y, bins=40, color='seagreen', edgecolor='w')
ax1.set_title('Исходная стоимость домов')
ax1.set_xlabel('Стоимость')
ax1.set_ylabel('Количество')

ax2.hist(y_clean, bins=40, color='firebrick', edgecolor='w')
ax2.set_title('Стоимость домов после удаления выбросов')
ax2.set_xlabel('Стоимость')
ax2.set_ylabel('Количество')

plt.tight_layout() 
plt.show()

#математически изменяеи признаки
math_transform = X_clean.copy()

math_transform['MedInc'] = np.log1p(math_transform['MedInc']) # логарифмируем MedInc
math_transform['HouseAge'] = np.square(math_transform['HouseAge']) # возводим возраст дома в квадрат
math_transform['Population'] = np.sqrt(math_transform['Population']) # извлекаем корень из Population

X_train, X_test, y_train, y_test = train_test_split(math_transform, y_clean, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# обучаем модель
model_lin = LinearRegression()
model_lin.fit(X_train_scaled, y_train)

test = model_lin.predict(X_test_scaled)
train = model_lin.predict(X_train_scaled)

#считаем параметры rmse и r2
rmse_train = np.sqrt(mean_squared_error(y_train, train))
rmse_test = np.sqrt(mean_squared_error(y_test, test))

r2_train = r2_score(y_train, train)
r2_test = r2_score(y_test, test)

print('Результаты после математической трансформации')
print(f'RMSE_train:{rmse_train} | RMSE_test:{rmse_test} | R2_train:{r2_train} | R2_test:{r2_test}')

#таблица с результатами 

end = [
{
    'Модель': 'Без изменений данных',
    'RMSE': '0.74',
    'R2': '0.57',
    'Преобразования': 'Без изменений'
},
{   'Модель': 'Без мультиколлинеарности',
    'RMSE': '0.65',
    'R2': '0.55',
    'Преобразования': 'Удаление AveBedrms и всех значений больше 5 '

},
{   'Модель': 'Без выбросов',
    'RMSE': '0.58',
    'R2': '0.60',
    'Преобразования': 'Удаление выбросов IQR ' 

},
{   'Модель': 'Мат. преобразования',
    'RMSE': '0.60',
    'R2': '0.57',
    'Преобразования': 'MedInc(log), HouseAge(^2), Population(sqrt) '

}]

data_end = pd.DataFrame(end)

print(data_end.to_string(index=False))
