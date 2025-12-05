import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import json

# Определяем пути
current_dir = Path(__file__).parent  # ml/src/
ml_dir = current_dir.parent

input_path = ml_dir / 'data' / 'raw' / 'train.csv'
train_output_path = ml_dir / 'data' / 'processed' / 'train_processed.csv'
test_output_path = ml_dir / 'data' / 'processed' / 'test_processed.csv'
preprocessor_path = ml_dir / 'data' / 'processed' / 'preprocessor_params.pkl'

#print("Загрузка данных(первые 5 млн для обучения)")
data = pd.read_csv(input_path, nrows=5000000)

#print("Разделяем на обучающую и тестовую выборки (5 млн на train, остальное на test)")
train_data = data.iloc[:5000000]  # Первые 5 млн строк

# Для теста загружаем следующие строки (если файл больше 5 млн)
try:
    test_data = pd.read_csv(input_path, skiprows=range(1, 5000001))
    #print(f"Загружено тестовых данных: {len(test_data)} строк")
except Exception as e:
    # Если файл меньше, создаем тестовую выборку из обучающей
    #print(f"Файл меньше 5 млн строк, создаем тестовую выборку из обучающей: {e}")
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=42,
                                             stratify=train_data['Response'])
    #print(f"Создана тестовая выборка: {len(test_data)} строк")


# Функция для обучения препроцессора (рассчитывает параметры только на train)
def fit_preprocessor(train_df):
    """
    Рассчитывает все параметры препроцессинга на обучающих данных
    """
    preprocessor_params = {}

    #print("Обучение препроцессора на train данных...")

    # 1. Медианы для заполнения пропусков в числовых признаках
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    medians = {}
    for col in numeric_cols:
        if col not in ['Response']:  # Исключаем целевую переменную
            medians[col] = train_df[col].median()
    preprocessor_params['medians'] = medians
    #print(f"  Рассчитаны медианы для {len(medians)} признаков")

    # 2. LabelEncoders для категориальных признаков
    label_encoders = {}

    # Для бинарных категориальных признаков
    binary_cols = ['Gender', 'Vehicle_Damage']
    for col in binary_cols:
        if col in train_df.columns:
            le = LabelEncoder()
            # Преобразуем в строку и обучаем
            train_df[col] = train_df[col].astype(str)
            le.fit(train_df[col])
            label_encoders[col] = le
            #print(f"  Обучен LabelEncoder для {col}")

    # Для Vehicle_Age - создаем маппинг
    if 'Vehicle_Age' in train_df.columns:
        vehicle_age_mapping = {
            '< 1 Year': 0,
            '1-2 Year': 1,
            '> 2 Years': 2
        }
        preprocessor_params['vehicle_age_mapping'] = vehicle_age_mapping
        #print("  Создан маппинг для Vehicle_Age")

    preprocessor_params['label_encoders'] = label_encoders

    # 3. Квантили для удаления выбросов
    if 'Annual_Premium' in train_df.columns:
        premium_q99 = train_df['Annual_Premium'].quantile(0.999)
        preprocessor_params['premium_q99'] = premium_q99
        #print(f"  Рассчитан 99.9% квантиль для Annual_Premium: {premium_q99}")

    # 4. Границы для Age групп
    if 'Age' in train_df.columns:
        age_bins = [18, 25, 35, 50, 65, 100]
        preprocessor_params['age_bins'] = age_bins
        #print(f"  Определены границы Age групп: {age_bins}")

    # 5. Квантили для Premium_Group
    if 'Annual_Premium' in train_df.columns:
        try:
            # Определяем квантили на train
            premium_quantiles = np.percentile(train_df['Annual_Premium'], [0, 20, 40, 60, 80, 100])
            preprocessor_params['premium_quantiles'] = premium_quantiles.tolist()
            #print(f"  Рассчитаны квантили для Premium_Group: {premium_quantiles}")
        except Exception as e:
            print(f"  Не удалось рассчитать квантили для Premium_Group: {e}")

    # 6. StandardScaler для числовых признаков
    numeric_features = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel']
    numeric_features = [col for col in numeric_features if col in train_df.columns]

    if numeric_features:
        scaler = StandardScaler()
        # Преобразуем в float перед обучением
        train_numeric = train_df[numeric_features].astype('float64')
        scaler.fit(train_numeric)
        preprocessor_params['scaler'] = scaler
        preprocessor_params['numeric_features'] = numeric_features
        #print(f"  Обучен StandardScaler для признаков: {numeric_features}")

    return preprocessor_params


# Функция для применения препроцессора (к train и test)
def apply_preprocessor(df, preprocessor_params, is_train=False):
    """
    Применяет параметры препроцессора к данным
    is_train: True если это обучающие данные (для них удаляем выбросы)
    """
    df_processed = df.copy()

    # 1. Заполнение пропусков медианами (из train)
    if 'medians' in preprocessor_params:
        for col, median_val in preprocessor_params['medians'].items():
            if col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    df_processed.loc[df_processed[col].isnull(), col] = median_val
                    #if is_train:
                        #print(f"  Заполнено пропусков в {col}: {df_processed[col].isnull().sum()}")

    # 2. Кодирование категориальных признаков
    if 'label_encoders' in preprocessor_params:
        for col, le in preprocessor_params['label_encoders'].items():
            if col in df_processed.columns:
                # Преобразуем в строку и применяем кодировку
                df_processed[col] = df_processed[col].astype(str)
                # Используем transform, а не fit_transform!
                try:
                    df_processed[col] = le.transform(df_processed[col])
                except ValueError as e:
                    # Если встретились новые значения в тесте, кодируем как -1
                    #print(f"  Предупреждение: новые значения в {col}, кодируем как -1")
                    # Создаем маппинг известных значений
                    known_values = set(le.classes_)

                    # Функция для преобразования
                    def encode_value(x):
                        return le.transform([x])[0] if x in known_values else -1

                    df_processed[col] = df_processed[col].apply(encode_value)

    # 3. Кодирование Vehicle_Age
    if 'vehicle_age_mapping' in preprocessor_params and 'Vehicle_Age' in df_processed.columns:
        vehicle_age_mapping = preprocessor_params['vehicle_age_mapping']
        df_processed['Vehicle_Age'] = df_processed['Vehicle_Age'].map(vehicle_age_mapping)
        # Заполняем пропуски значением 1 (самое частое)
        if df_processed['Vehicle_Age'].isnull().sum() > 0:
            df_processed.loc[df_processed['Vehicle_Age'].isnull(), 'Vehicle_Age'] = 1

    # 4. Обработка Region_Code и Policy_Sales_Channel
    for col in ['Region_Code', 'Policy_Sales_Channel']:
        if col in df_processed.columns:
            # Преобразуем в строку и извлекаем числа
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].str.extract(r'(\d+)')
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            # Заполняем пропуски медианой из train
            if col in preprocessor_params.get('medians', {}):
                median_val = preprocessor_params['medians'][col]
                df_processed.loc[df_processed[col].isnull(), col] = median_val

    # 5. Удаление выбросов (ТОЛЬКО train!)
    if is_train and 'premium_q99' in preprocessor_params and 'Annual_Premium' in df_processed.columns:
        original_len = len(df_processed)
        premium_q99 = preprocessor_params['premium_q99']
        mask = df_processed['Annual_Premium'] <= premium_q99
        df_processed = df_processed[mask].copy()
        removed_count = original_len - len(df_processed)
        #if removed_count > 0:
            #print(f"  Удалено выбросов из Annual_Premium: {removed_count} строк")

    # 6. Обработка Age (только на train)
    if is_train and 'Age' in df_processed.columns:
        original_len = len(df_processed)
        age_mask = (df_processed['Age'] >= 18) & (df_processed['Age'] <= 100)
        df_processed = df_processed[age_mask].copy()
        removed_count = original_len - len(df_processed)
        #if removed_count > 0:
            #print(f"  Удалено некорректных значений Age: {removed_count} строк")

    # 7. Создание новых признаков
    # Age_Group
    if 'age_bins' in preprocessor_params and 'Age' in df_processed.columns:
        age_bins = preprocessor_params['age_bins']
        df_processed['Age_Group'] = pd.cut(df_processed['Age'],
                                           bins=age_bins,
                                           labels=[0, 1, 2, 3, 4],
                                           include_lowest=True)
        df_processed['Age_Group'] = df_processed['Age_Group'].astype('float64')

    # Premium_Group (используем квантили из train)
    if 'premium_quantiles' in preprocessor_params and 'Annual_Premium' in df_processed.columns:
        premium_quantiles = preprocessor_params['premium_quantiles']
        try:
            df_processed['Premium_Group'] = pd.cut(df_processed['Annual_Premium'],
                                                   bins=premium_quantiles,
                                                   labels=[0, 1, 2, 3, 4],
                                                   include_lowest=True)
            df_processed['Premium_Group'] = df_processed['Premium_Group'].astype('float64')
        except Exception as e:
            print(f"  Не удалось создать Premium_Group: {e}")

    # 8. Удаление ненужных столбцов, id не нужен для обучения
    columns_to_drop = ['id']
    for col in columns_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    # 9. Нормализация числовых признаков
    if 'scaler' in preprocessor_params and 'numeric_features' in preprocessor_params:
        scaler = preprocessor_params['scaler']
        numeric_features = preprocessor_params['numeric_features']

        # Преобразуем в float64
        for col in numeric_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype('float64')

        # Применяем скаляр (transform, не fit_transform!)
        normalized_array = scaler.transform(df_processed[numeric_features])
        normalized_df = pd.DataFrame(normalized_array,
                                     columns=numeric_features,
                                     index=df_processed.index)

        for col in numeric_features:
            df_processed[col] = normalized_df[col]

    return df_processed


# Основной процесс
#print("\n=== ОБУЧЕНИЕ ПРЕПРОЦЕССОРА НА TRAIN ДАННЫХ ===")
preprocessor_params = fit_preprocessor(train_data)

# Сохраняем параметры препроцессора
with open(preprocessor_path, 'wb') as f:
    pickle.dump(preprocessor_params, f)
#print(f"\nПараметры препроцессора сохранены в: {preprocessor_path}")

#print("\n=== ПРИМЕНЕНИЕ ПРЕПРОЦЕССОРА К TRAIN ДАННЫМ ===")
train_processed = apply_preprocessor(train_data, preprocessor_params, is_train=True)

#print("\n=== ПРИМЕНЕНИЕ ПРЕПРОЦЕССОРА К TEST ДАННЫМ ===")
test_processed = apply_preprocessor(test_data, preprocessor_params, is_train=False)

#print("\nСохранение обработанных данных...")
train_processed.to_csv(train_output_path, index=False)
test_processed.to_csv(test_output_path, index=False)

"""
print(f"\nОбработка завершена!")
print(f"Обучающая выборка: {len(train_processed)} строк, {len(train_processed.columns)} столбцов")
print(f"Тестовая выборка: {len(test_processed)} строк, {len(test_processed.columns)} столбцов")
print(f"Сохранено в: {train_output_path} и {test_output_path}")

# Проверка
print("Типы данных в train:")
print(train_processed.dtypes)
print("\nТипы данных в test:")
print(test_processed.dtypes)

if 'Response' in train_processed.columns:
    print("\nРаспределение Response в train:")
    print(train_processed['Response'].value_counts(normalize=True))

if 'Response' in test_processed.columns:
    print("\nРаспределение Response в test:")
    print(test_processed['Response'].value_counts(normalize=True))"""