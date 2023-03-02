import pandas as pd
from typing import Dict, List, Tuple


def extract_autoregression_features(train_df: pd.DataFrame, target_col: str, freq: str) -> Dict:
    def code_mean(df: pd.DataFrame, cat_feature: str, real_feature: str) -> Dict:
        """
        Возвращает словарь, где ключами являются уникальные категории признака cat_feature,
        а значениями - средние по real_feature
        """
        return dict(df.groupby(cat_feature)[real_feature].mean())

    result = dict()

    # среднее за час
    if freq == 'H':
        train_df['hour'] = train_df.index.hour
        result['avg_hour'] = code_mean(train_df, 'hour', target_col)
        # TODO: отображение на круг? https://habr.com/ru/company/vk/blog/346942/

    # среднее за день недели
    train_df['weekday'] = train_df.index.day_name()
    result['avg_weekday'] = code_mean(train_df, 'weekday', target_col)

    # среднее за день месяца
    train_df['day_of_month'] = train_df.index.day
    result['avg_day_of_month'] = code_mean(train_df, 'day_of_month', target_col)

    # среднее за месяц
    if len(train_df.resample('D').max().index) >= 365:
        train_df['month'] = train_df.index.month
        result['avg_month'] = code_mean(train_df, 'month', target_col)

    return result


def add_autoregression_features(df: pd.DataFrame, extracted_features: Dict) -> Tuple[pd.DataFrame, List[str]]:
    features = []

    # среднее за час
    # TODO: используем эту фичу, только если:
    # 1) она была создана при обучении (т.е. обучались с дискретизацией 1 час), и
    # 2) дискретизация прогноза БМ в переданном df также 1 час
    if (avg_hour := extracted_features.get('avg_hour')) is not None:
        df['hour'] = df.index.hour
        df['avg_hour'] = list(map(avg_hour.get, df['hour']))
        features.append('avg_hour')

    if (avg_weekday := extracted_features.get('avg_weekday')) is not None:
        df['weekday'] = df.index.day_name()
        df['avg_weekday'] = list(map(avg_weekday.get, df['weekday']))
        features.append('avg_weekday')

    if (avg_day_of_month := extracted_features.get('avg_day_of_month')) is not None:
        df['day_of_month'] = df.index.day
        df['avg_day_of_month'] = list(map(avg_day_of_month.get, df['day_of_month']))
    features.append('avg_day_of_month')

    if (avg_month := extracted_features.get('avg_month')) is not None:
        df['month'] = df.index.month
        df['avg_month'] = list(map(avg_month.get, df['month']))
        features.append('avg_month')

    return df, features
