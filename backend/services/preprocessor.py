import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def apply_preprocessor(df, preprocessor_params, is_train=False):
    """
    Применяет параметры препроцессора к данным
    is_train: True если это обучающие данные (для них удаляем выбросы)
    """
    df_processed = df.copy()

    # 1. Заполнение пропусков медианами (из train)
    if "medians" in preprocessor_params:
        for col, median_val in preprocessor_params["medians"].items():
            if col in df_processed.columns:
                if df_processed[col].isnull().sum() > 0:
                    df_processed.loc[df_processed[col].isnull(), col] = median_val
                    # if is_train:
                    # print(f"  Заполнено пропусков в {col}: {df_processed[col].isnull().sum()}")

    # 2. Кодирование категориальных признаков
    if "label_encoders" in preprocessor_params:
        for col, le in preprocessor_params["label_encoders"].items():
            if col in df_processed.columns:
                # Преобразуем в строку и применяем кодировку
                df_processed[col] = df_processed[col].astype(str)
                # Используем transform, а не fit_transform!
                try:
                    df_processed[col] = le.transform(df_processed[col])
                except ValueError as e:
                    # Если встретились новые значения в тесте, кодируем как -1
                    # print(f"  Предупреждение: новые значения в {col}, кодируем как -1")
                    # Создаем маппинг известных значений
                    known_values = set(le.classes_)

                    # Функция для преобразования
                    def encode_value(x):
                        return le.transform([x])[0] if x in known_values else -1

                    df_processed[col] = df_processed[col].apply(encode_value)

    # 3. Кодирование Vehicle_Age
    if (
        "vehicle_age_mapping" in preprocessor_params
        and "Vehicle_Age" in df_processed.columns
    ):
        vehicle_age_mapping = preprocessor_params["vehicle_age_mapping"]
        df_processed["Vehicle_Age"] = df_processed["Vehicle_Age"].map(
            vehicle_age_mapping
        )
        # Заполняем пропуски значением 1 (самое частое)
        if df_processed["Vehicle_Age"].isnull().sum() > 0:
            df_processed.loc[df_processed["Vehicle_Age"].isnull(), "Vehicle_Age"] = 1

    # 4. Обработка Region_Code и Policy_Sales_Channel
    for col in ["Region_Code", "Policy_Sales_Channel"]:
        if col in df_processed.columns:
            # Преобразуем в строку и извлекаем числа
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].str.extract(r"(\d+)")
            df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")
            # Заполняем пропуски медианой из train
            if col in preprocessor_params.get("medians", {}):
                median_val = preprocessor_params["medians"][col]
                df_processed.loc[df_processed[col].isnull(), col] = median_val

    # 5. Удаление выбросов (ТОЛЬКО train!)
    if (
        is_train
        and "premium_q99" in preprocessor_params
        and "Annual_Premium" in df_processed.columns
    ):
        original_len = len(df_processed)
        premium_q99 = preprocessor_params["premium_q99"]
        mask = df_processed["Annual_Premium"] <= premium_q99
        df_processed = df_processed[mask].copy()
        removed_count = original_len - len(df_processed)
        # if removed_count > 0:
        # print(f"  Удалено выбросов из Annual_Premium: {removed_count} строк")

    # 6. Обработка Age (только на train)
    if is_train and "Age" in df_processed.columns:
        original_len = len(df_processed)
        age_mask = (df_processed["Age"] >= 18) & (df_processed["Age"] <= 100)
        df_processed = df_processed[age_mask].copy()
        removed_count = original_len - len(df_processed)
        # if removed_count > 0:
        # print(f"  Удалено некорректных значений Age: {removed_count} строк")

    # 7. Создание новых признаков
    # Age_Group
    if "age_bins" in preprocessor_params and "Age" in df_processed.columns:
        age_bins = preprocessor_params["age_bins"]
        df_processed["Age_Group"] = pd.cut(
            df_processed["Age"],
            bins=age_bins,
            labels=[0, 1, 2, 3, 4],
            include_lowest=True,
        )
        df_processed["Age_Group"] = df_processed["Age_Group"].astype("float64")

    # Premium_Group (используем квантили из train)
    if (
        "premium_quantiles" in preprocessor_params
        and "Annual_Premium" in df_processed.columns
    ):
        premium_quantiles = preprocessor_params["premium_quantiles"]
        try:
            df_processed["Premium_Group"] = pd.cut(
                df_processed["Annual_Premium"],
                bins=premium_quantiles,
                labels=[0, 1, 2, 3, 4],
                include_lowest=True,
            )
            df_processed["Premium_Group"] = df_processed["Premium_Group"].astype(
                "float64"
            )
        except Exception as e:
            print(f"  Не удалось создать Premium_Group: {e}")

    # 8. Удаление ненужных столбцов, id не нужен для обучения
    columns_to_drop = ["id"]
    for col in columns_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    # 9. Нормализация числовых признаков
    if "scaler" in preprocessor_params and "numeric_features" in preprocessor_params:
        scaler = preprocessor_params["scaler"]
        numeric_features = preprocessor_params["numeric_features"]

        # Преобразуем в float64
        for col in numeric_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype("float64")

        # Применяем скаляр (transform, не fit_transform!)
        normalized_array = scaler.transform(df_processed[numeric_features])
        normalized_df = pd.DataFrame(
            normalized_array, columns=numeric_features, index=df_processed.index
        )

        for col in numeric_features:
            df_processed[col] = normalized_df[col]

    # 10. Кастим категориальные признаки в int
    df_processed["Age_Group"] = df_processed["Age_Group"].astype(int)
    df_processed["Premium_Group"] = df_processed["Premium_Group"].astype(int)

    return df_processed
