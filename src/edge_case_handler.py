"""
Модуль для обработки граничных случаев
"""
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class EdgeCaseHandler:
    """Обработчик граничных случаев"""
    
    def __init__(self):
        """Инициализация обработчика"""
        self.validation_errors = []
        self.warnings = []
    
    def validate_and_fix_data(self, df: pd.DataFrame, data_type: str = 'transactions') -> pd.DataFrame:
        """
        Валидация и исправление данных
        
        Args:
            df: DataFrame для проверки
            data_type: Тип данных ('transactions', 'clients', 'features')
            
        Returns:
            Исправленный DataFrame
        """
        logger.info(f"Валидация данных типа: {data_type}")
        
        if data_type == 'clients':
            df = self._validate_clients(df)
        elif data_type == 'transactions':
            df = self._validate_transactions(df)
        elif data_type == 'features':
            df = self._validate_features(df)
        
        return df
    
    def _validate_clients(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация данных клиентов"""
        # Проверка обязательных колонок
        required_cols = ['client_code', 'name', 'status', 'age', 'city', 'avg_monthly_balance_KZT']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"Отсутствуют колонки: {missing_cols}")
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
        
        # Проверка и исправление возраста
        if 'age' in df.columns:
            # Исправляем отрицательный возраст
            invalid_age = df['age'] < 18
            if invalid_age.any():
                self.warnings.append(f"Найдено {invalid_age.sum()} клиентов с возрастом < 18")
                df.loc[invalid_age, 'age'] = 18
            
            # Исправляем слишком большой возраст
            too_old = df['age'] > 100
            if too_old.any():
                self.warnings.append(f"Найдено {too_old.sum()} клиентов с возрастом > 100")
                df.loc[too_old, 'age'] = 100
        
        # Проверка и исправление баланса
        if 'avg_monthly_balance_KZT' in df.columns:
            # Отрицательный баланс заменяем на 0
            negative_balance = df['avg_monthly_balance_KZT'] < 0
            if negative_balance.any():
                self.warnings.append(f"Найдено {negative_balance.sum()} клиентов с отрицательным балансом")
                df.loc[negative_balance, 'avg_monthly_balance_KZT'] = 0
            
            # Заполняем пропуски средним значением
            if df['avg_monthly_balance_KZT'].isna().any():
                mean_balance = df['avg_monthly_balance_KZT'].mean()
                df['avg_monthly_balance_KZT'].fillna(mean_balance, inplace=True)
                self.warnings.append("Заполнены пропуски в балансе средним значением")
        
        # Нормализация статусов
        status_mapping = {
            'студент': 'Студент',
            'зп': 'Зарплатный клиент',
            'обычный': 'Стандартный клиент',
            'вип': 'Премиальный клиент',
            'премиум': 'Премиальный клиент',
            'стандарт': 'Стандартный клиент'
        }
        
        df['status'] = df['status'].apply(lambda x: status_mapping.get(x.lower(), x) if pd.notna(x) else 'Стандартный клиент')
        
        return df
    
    def _validate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация транзакций"""
        # Проверка сумм
        if 'amount' in df.columns:
            # Отрицательные суммы делаем положительными
            negative_amounts = df['amount'] < 0
            if negative_amounts.any():
                self.warnings.append(f"Найдено {negative_amounts.sum()} транзакций с отрицательными суммами")
                df.loc[negative_amounts, 'amount'] = df.loc[negative_amounts, 'amount'].abs()
            
            # Заполняем пропуски нулями
            if df['amount'].isna().any():
                df['amount'].fillna(0, inplace=True)
                self.warnings.append("Заполнены пропуски в суммах нулями")
        
        # Проверка валют
        if 'currency' in df.columns:
            valid_currencies = ['KZT', 'USD', 'EUR', 'RUB']
            invalid_currency = ~df['currency'].isin(valid_currencies)
            if invalid_currency.any():
                self.warnings.append(f"Найдено {invalid_currency.sum()} транзакций с неизвестной валютой")
                df.loc[invalid_currency, 'currency'] = 'KZT'
        
        # Проверка дат
        if 'date' in df.columns:
            # Конвертируем в datetime если еще не сделано
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                try:
                    df['date'] = pd.to_datetime(df['date'])
                except:
                    self.warnings.append("Не удалось конвертировать даты, используем текущую дату")
                    df['date'] = pd.Timestamp.now()
        
        return df
    
    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация признаков"""
        # Заполняем пропуски в числовых колонках
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                # Для трат и сумм - заполняем нулями
                if 'spend' in col or 'amount' in col or 'sum' in col:
                    df[col].fillna(0, inplace=True)
                # Для счетчиков - заполняем нулями
                elif 'count' in col:
                    df[col].fillna(0, inplace=True)
                # Для остальных - средним значением
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
        
        # Проверка свободных средств
        if 'free_funds' in df.columns:
            # Если отрицательные, делаем 0
            df.loc[df['free_funds'] < 0, 'free_funds'] = 0
        
        # Проверка соотношений
        if 'outflow_inflow_ratio' in df.columns:
            # Если бесконечность (деление на 0), ставим 1
            df.loc[np.isinf(df['outflow_inflow_ratio']), 'outflow_inflow_ratio'] = 1
            # Заполняем пропуски единицей
            df['outflow_inflow_ratio'].fillna(1, inplace=True)
        
        return df
    
    def handle_extreme_values(self, df: pd.DataFrame, column: str, 
                            lower_percentile: float = 0.01, 
                            upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Обработка экстремальных значений (выбросов)
        
        Args:
            df: DataFrame
            column: Название колонки
            lower_percentile: Нижний процентиль для обрезки
            upper_percentile: Верхний процентиль для обрезки
            
        Returns:
            DataFrame с обработанными выбросами
        """
        if column in df.columns:
            lower_bound = df[column].quantile(lower_percentile)
            upper_bound = df[column].quantile(upper_percentile)
            
            # Обрезаем выбросы
            outliers_low = df[column] < lower_bound
            outliers_high = df[column] > upper_bound
            
            if outliers_low.any() or outliers_high.any():
                self.warnings.append(
                    f"Обработано {outliers_low.sum() + outliers_high.sum()} выбросов в колонке {column}"
                )
                df.loc[outliers_low, column] = lower_bound
                df.loc[outliers_high, column] = upper_bound
        
        return df
    
    def handle_missing_categories(self, features: pd.Series, required_categories: list) -> pd.Series:
        """
        Обработка отсутствующих категорий в признаках
        
        Args:
            features: Series с признаками клиента
            required_categories: Список обязательных категорий
            
        Returns:
            Series с добавленными отсутствующими категориями
        """
        for category in required_categories:
            spend_col = f'spend_{category}'
            if spend_col not in features.index:
                features[spend_col] = 0
        
        return features
    
    def safe_division(self, numerator: float, denominator: float, 
                     default: float = 0) -> float:
        """
        Безопасное деление с обработкой деления на ноль
        
        Args:
            numerator: Числитель
            denominator: Знаменатель
            default: Значение по умолчанию при делении на ноль
            
        Returns:
            Результат деления или значение по умолчанию
        """
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator
    
    def validate_push_notification(self, push: str, max_length: int = 220) -> str:
        """
        Валидация и исправление push-уведомления
        
        Args:
            push: Текст push-уведомления
            max_length: Максимальная длина
            
        Returns:
            Исправленное push-уведомление
        """
        # Обрезаем если слишком длинное
        if len(push) > max_length:
            # Обрезаем и добавляем многоточие
            push = push[:max_length-3] + '...'
            self.warnings.append(f"Push обрезан до {max_length} символов")
        
        # Убираем множественные пробелы
        import re
        push = re.sub(r'\s+', ' ', push)
        
        # Убираем пробелы в начале и конце
        push = push.strip()
        
        # Проверяем на запрещенные символы
        push = re.sub(r'[<>\"\'\\]', '', push)
        
        return push
    
    def get_report(self) -> Dict[str, Any]:
        """
        Получить отчет о валидации
        
        Returns:
            Словарь с ошибками и предупреждениями
        """
        return {
            'errors': self.validation_errors,
            'warnings': self.warnings,
            'total_issues': len(self.validation_errors) + len(self.warnings)
        }


# Функции-помощники для использования в других модулях
def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Безопасное получение значения из объекта
    
    Args:
        obj: Объект (dict, Series, DataFrame)
        key: Ключ/индекс
        default: Значение по умолчанию
        
    Returns:
        Значение или default
    """
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        elif isinstance(obj, pd.Series):
            return obj.get(key, default) if key in obj.index else default
        elif isinstance(obj, pd.DataFrame):
            return obj.get(key, default) if key in obj.columns else default
        else:
            return getattr(obj, key, default)
    except:
        return default


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Ограничить значение в диапазоне
    
    Args:
        value: Значение
        min_value: Минимум
        max_value: Максимум
        
    Returns:
        Ограниченное значение
    """
    return max(min_value, min(value, max_value))


if __name__ == "__main__":
    # Тестирование
    handler = EdgeCaseHandler()
    
    # Тест с некорректными данными
    test_clients = pd.DataFrame({
        'client_code': [1, 2, 3],
        'name': ['Test1', 'Test2', 'Test3'],
        'status': ['студент', 'вип', 'обычный'],
        'age': [-5, 150, 25],  # Некорректные возраста
        'city': ['Алматы', 'Астана', 'Алматы'],
        'avg_monthly_balance_KZT': [-1000, 100000, None]  # Отрицательный баланс и пропуск
    })
    
    print("До валидации:")
    print(test_clients)
    
    fixed_clients = handler.validate_and_fix_data(test_clients, 'clients')
    
    print("\nПосле валидации:")
    print(fixed_clients)
    
    print("\nОтчет о валидации:")
    report = handler.get_report()
    for warning in report['warnings']:
        print(f"⚠️ {warning}")
