"""
ETL модуль для загрузки и объединения данных клиентов, транзакций и переводов
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataLoaderV2:
    """Класс для загрузки и предобработки данных из трех источников"""
    
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
    
    def load_clients(self, clients_file: str = "data/clients.csv") -> pd.DataFrame:
        """
        Загрузка данных клиентов
        
        Args:
            clients_file: путь к файлу с клиентами
            
        Returns:
            DataFrame с клиентами
        """
        logger.info(f"Загрузка клиентов из {clients_file}")
        
        try:
            clients_df = pd.read_csv(clients_file, encoding='utf-8')
        except UnicodeDecodeError:
            clients_df = pd.read_csv(clients_file, encoding='cp1251')
        
        logger.info(f"Загружено {len(clients_df)} клиентов")
        
        # Проверка обязательных колонок
        required_columns = ['client_code', 'name', 'status', 'age', 'city', 'avg_monthly_balance_KZT']
        missing_columns = set(required_columns) - set(clients_df.columns)
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки в clients.csv: {missing_columns}")
        
        return clients_df
    
    def load_transactions(self, client_code: int, data_dir: str = "data") -> Optional[pd.DataFrame]:
        """
        Загрузка транзакций клиента
        
        Args:
            client_code: код клиента
            data_dir: директория с данными
            
        Returns:
            DataFrame с транзакциями или None если файл не найден
        """
        transactions_file = Path(data_dir) / f"client_{client_code}_transactions_3m.csv"
        
        if not transactions_file.exists():
            logger.debug(f"Файл транзакций не найден: {transactions_file}")
            return None
        
        try:
            transactions_df = pd.read_csv(transactions_file, encoding='utf-8')
        except UnicodeDecodeError:
            transactions_df = pd.read_csv(transactions_file, encoding='cp1251')
        
        # Конвертируем дату в datetime
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        
        # Добавляем direction для транзакций (все расходы)
        transactions_df['direction'] = 'out'
        transactions_df['type'] = 'card_out'  # По умолчанию карточные расходы
        
        logger.debug(f"Загружено {len(transactions_df)} транзакций для клиента {client_code}")
        
        return transactions_df
    
    def load_transfers(self, client_code: int, data_dir: str = "data") -> Optional[pd.DataFrame]:
        """
        Загрузка переводов клиента
        
        Args:
            client_code: код клиента
            data_dir: директория с данными
            
        Returns:
            DataFrame с переводами или None если файл не найден
        """
        transfers_file = Path(data_dir) / f"client_{client_code}_transfers_3m.csv"
        
        if not transfers_file.exists():
            logger.debug(f"Файл переводов не найден: {transfers_file}")
            return None
        
        try:
            transfers_df = pd.read_csv(transfers_file, encoding='utf-8')
        except UnicodeDecodeError:
            transfers_df = pd.read_csv(transfers_file, encoding='cp1251')
        
        # Конвертируем дату в datetime
        transfers_df['date'] = pd.to_datetime(transfers_df['date'])
        
        # Для переводов категория определяется по типу
        transfers_df['category'] = transfers_df['type'].apply(self._map_transfer_to_category)
        
        logger.debug(f"Загружено {len(transfers_df)} переводов для клиента {client_code}")
        
        return transfers_df
    
    def _map_transfer_to_category(self, transfer_type: str) -> str:
        """
        Маппинг типа перевода на категорию
        
        Args:
            transfer_type: тип перевода
            
        Returns:
            Категория
        """
        category_mapping = {
            'salary_in': 'Доход',
            'stipend_in': 'Доход',
            'family_in': 'Доход',
            'cashback_in': 'Кешбэк',
            'refund_in': 'Возврат',
            'card_in': 'Пополнение',
            'p2p_out': 'Переводы',
            'card_out': 'Оплата картой',
            'atm_withdrawal': 'Снятие наличных',
            'utilities_out': 'Коммунальные услуги',
            'loan_payment_out': 'Кредит',
            'cc_repayment_out': 'Кредитная карта',
            'installment_payment_out': 'Рассрочка',
            'fx_buy': 'Обмен валют',
            'fx_sell': 'Обмен валют',
            'invest_out': 'Инвестиции',
            'invest_in': 'Инвестиции',
            'deposit_topup_out': 'Депозит',
            'deposit_fx_topup_out': 'Депозит',
            'deposit_fx_withdraw_in': 'Депозит',
            'gold_buy_out': 'Золото',
            'gold_sell_in': 'Золото'
        }
        
        return category_mapping.get(transfer_type, 'Прочее')
    
    def combine_client_data(self, client_code: int, clients_df: pd.DataFrame, 
                           data_dir: str = "data") -> pd.DataFrame:
        """
        Объединение всех данных для одного клиента
        
        Args:
            client_code: код клиента
            clients_df: DataFrame с информацией о клиентах
            data_dir: директория с данными
            
        Returns:
            DataFrame с объединенными данными
        """
        # Получаем информацию о клиенте
        client_info = clients_df[clients_df['client_code'] == client_code].iloc[0]
        
        # Загружаем транзакции и переводы
        transactions_df = self.load_transactions(client_code, data_dir)
        transfers_df = self.load_transfers(client_code, data_dir)
        
        # Объединяем данные
        all_data = []
        
        if transactions_df is not None:
            # Добавляем информацию о клиенте к транзакциям
            for col in ['name', 'status', 'age', 'city', 'avg_monthly_balance_KZT']:
                if col not in transactions_df.columns:
                    transactions_df[col] = client_info[col]
            all_data.append(transactions_df)
        
        if transfers_df is not None:
            # Добавляем информацию о клиенте к переводам
            for col in ['name', 'status', 'age', 'city', 'avg_monthly_balance_KZT']:
                if col not in transfers_df.columns:
                    transfers_df[col] = client_info[col]
            all_data.append(transfers_df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Сохраняем эталонный продукт если есть
            if 'product' in combined_df.columns:
                # Берем первое непустое значение продукта как эталон
                reference_product = combined_df['product'].dropna().iloc[0] if not combined_df['product'].dropna().empty else None
                combined_df['reference_product'] = reference_product
            
            return combined_df
        else:
            # Если нет данных о транзакциях/переводах, возвращаем только информацию о клиенте
            return pd.DataFrame([client_info.to_dict()])
    
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
        
        if 'currency' in df.columns and 'amount' in df.columns:
            # Создаем новую колонку с суммой в тенге
            df['amount_kzt'] = df.apply(
                lambda row: row['amount'] * self.currency_rates.get(row.get('currency', 'KZT'), 1),
                axis=1
            )
        else:
            # Если нет валюты, считаем что все в тенге
            if 'amount' in df.columns:
                df['amount_kzt'] = df['amount']
        
        logger.info("Конвертация валют завершена")
        return df
    
    def load_all_data(self, clients_file: str = "data/clients.csv", 
                     data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Загрузка всех данных
        
        Args:
            clients_file: путь к файлу с клиентами
            data_dir: директория с данными транзакций и переводов
            
        Returns:
            Кортеж (DataFrame с клиентами, DataFrame со всеми транзакциями)
        """
        logger.info("Начало загрузки всех данных")
        
        # Загружаем клиентов
        clients_df = self.load_clients(clients_file)
        
        # Собираем данные по всем клиентам
        all_transactions = []
        
        for client_code in clients_df['client_code'].unique():
            logger.info(f"Обработка клиента {client_code}")
            
            client_data = self.combine_client_data(client_code, clients_df, data_dir)
            if client_data is not None and not client_data.empty:
                all_transactions.append(client_data)
        
        # Объединяем все транзакции
        if all_transactions:
            transactions_df = pd.concat(all_transactions, ignore_index=True)
            transactions_df = self.convert_to_kzt(transactions_df)
            logger.info(f"Загружено {len(transactions_df)} записей для {len(clients_df)} клиентов")
        else:
            # Если нет транзакций, создаем пустой DataFrame с нужными колонками
            transactions_df = pd.DataFrame()
            logger.warning("Не найдено данных о транзакциях")
        
        return clients_df, transactions_df


def main():
    """Тестовый запуск ETL"""
    loader = DataLoaderV2()
    
    # Загружаем все данные
    clients_df, transactions_df = loader.load_all_data()
    
    print(f"\nЗагружено клиентов: {len(clients_df)}")
    print(f"Всего транзакций/переводов: {len(transactions_df)}")
    
    if not transactions_df.empty:
        print("\nПример данных:")
        print(transactions_df.head())
        
        print("\nУникальные категории:")
        if 'category' in transactions_df.columns:
            print(transactions_df['category'].value_counts().head(10))
        
        print("\nУникальные типы:")
        if 'type' in transactions_df.columns:
            print(transactions_df['type'].value_counts().head(10))


if __name__ == "__main__":
    main()
