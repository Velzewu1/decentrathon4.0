"""
ETL модуль для загрузки и предобработки данных
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и предобработки данных"""
    
    def __init__(self, config_path: str = "conf/weights.yaml"):
        """
        Инициализация загрузчика данных
        
        Args:
            config_path: путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        self.currency_rates = self.config['currency_rates']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Загрузка данных из CSV файла
        
        Args:
            file_path: путь к CSV файлу
            
        Returns:
            DataFrame с загруженными данными
        """
        logger.info(f"Загрузка данных из {file_path}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='cp1251')
        
        logger.info(f"Загружено {len(df)} записей")
        return df
    
    def convert_to_kzt(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Конвертация всех сумм в тенге
        
        Args:
            df: исходный DataFrame
            
        Returns:
            DataFrame с суммами в тенге
        """
        logger.info("Конвертация валют в KZT")
        
        df = df.copy()
        
        # Создаем новую колонку с суммой в тенге
        df['amount_kzt'] = df.apply(
            lambda row: row['amount'] * self.currency_rates.get(row['currency'], 1),
            axis=1
        )
        
        logger.info("Конвертация валют завершена")
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Парсинг дат и добавление временных признаков
        
        Args:
            df: исходный DataFrame
            
        Returns:
            DataFrame с распарсенными датами
        """
        logger.info("Парсинг дат")
        
        df = df.copy()
        
        # Конвертируем строку в datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Добавляем временные признаки
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df['month_year'] = df['date'].dt.to_period('M')
        
        logger.info("Парсинг дат завершен")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Валидация и очистка данных
        
        Args:
            df: исходный DataFrame
            
        Returns:
            Очищенный DataFrame
        """
        logger.info("Валидация данных")
        
        df = df.copy()
        
        # Удаляем строки с пропущенными критическими значениями
        critical_columns = ['client_code', 'date', 'amount', 'currency']
        initial_rows = len(df)
        df = df.dropna(subset=critical_columns)
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.warning(f"Удалено {dropped_rows} строк с пропущенными значениями")
        
        # Проверка корректности значений
        valid_statuses = ['Студент', 'Зарплатный клиент', 'Премиальный клиент', 'Стандартный клиент']
        df = df[df['status'].isin(valid_statuses)]
        
        valid_currencies = ['KZT', 'USD', 'EUR']
        df = df[df['currency'].isin(valid_currencies)]
        
        valid_directions = ['in', 'out']
        df = df[df['direction'].isin(valid_directions)]
        
        # Убираем отрицательные суммы (если есть)
        df = df[df['amount'] >= 0]
        
        logger.info(f"После валидации осталось {len(df)} записей")
        return df
    
    def filter_last_n_months(self, df: pd.DataFrame, n_months: int = 3) -> pd.DataFrame:
        """
        Фильтрация данных за последние N месяцев
        
        Args:
            df: исходный DataFrame
            n_months: количество месяцев
            
        Returns:
            DataFrame с данными за последние N месяцев
        """
        logger.info(f"Фильтрация данных за последние {n_months} месяцев")
        
        df = df.copy()
        
        # Находим максимальную дату в данных
        max_date = df['date'].max()
        
        # Вычисляем минимальную дату (N месяцев назад)
        min_date = max_date - pd.DateOffset(months=n_months)
        
        # Фильтруем данные
        df = df[df['date'] > min_date]
        
        logger.info(f"После фильтрации осталось {len(df)} записей")
        return df
    
    def preprocess(self, file_path: str, n_months: Optional[int] = None) -> pd.DataFrame:
        """
        Полный пайплайн предобработки данных
        
        Args:
            file_path: путь к CSV файлу
            n_months: количество месяцев для фильтрации (если None, берется из конфига)
            
        Returns:
            Предобработанный DataFrame
        """
        logger.info("Начало предобработки данных")
        
        if n_months is None:
            n_months = self.config['general']['aggregation_months']
        
        # Загрузка данных
        df = self.load_data(file_path)
        
        # Валидация
        df = self.validate_data(df)
        
        # Парсинг дат
        df = self.parse_dates(df)
        
        # Фильтрация по времени
        df = self.filter_last_n_months(df, n_months)
        
        # Конвертация валют
        df = self.convert_to_kzt(df)
        
        logger.info("Предобработка данных завершена")
        logger.info(f"Итоговый размер данных: {df.shape}")
        
        return df


def main():
    """Тестовый запуск ETL"""
    loader = DataLoader()
    
    # Создаем тестовые данные для демонстрации
    test_data = pd.DataFrame({
        'client_code': [1, 1, 2, 2, 3],
        'name': ['Иван', 'Иван', 'Мария', 'Мария', 'Петр'],
        'status': ['Премиальный клиент', 'Премиальный клиент', 'Студент', 'Студент', 'Зарплатный клиент'],
        'age': [35, 35, 22, 22, 45],
        'city': ['Алматы', 'Алматы', 'Астана', 'Астана', 'Шымкент'],
        'avg_monthly_balance_KZT': [500000, 500000, 50000, 50000, 200000],
        'date': ['2024-08-01', '2024-08-15', '2024-09-01', '2024-09-10', '2024-08-20'],
        'category': ['Рестораны', 'Такси', 'Продукты', 'Транспорт', 'Путешествия'],
        'amount': [15000, 5000, 10000, 2000, 100000],
        'currency': ['KZT', 'KZT', 'USD', 'KZT', 'EUR'],
        'type': ['card_out', 'card_out', 'card_out', 'card_out', 'fx_buy'],
        'direction': ['out', 'out', 'out', 'out', 'out']
    })
    
    # Сохраняем тестовые данные
    test_data.to_csv('data/test_data.csv', index=False)
    
    # Загружаем и обрабатываем
    processed_df = loader.preprocess('data/test_data.csv')
    print("\nОбработанные данные:")
    print(processed_df.head())
    print(f"\nФорма данных: {processed_df.shape}")


if __name__ == "__main__":
    main()
