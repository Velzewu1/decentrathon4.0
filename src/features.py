"""
Модуль для расчета агрегированных признаков клиентов
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Класс для расчета агрегированных признаков"""
    
    def __init__(self, config_path: str = "conf/weights.yaml"):
        """
        Инициализация
        
        Args:
            config_path: путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        
        # Ключевые категории для анализа
        self.key_categories = [
            'Такси', 'Путешествия', 'Отели', 'Билеты',
            'Рестораны', 'Ювелирные изделия', 'Косметика и парфюмерия',
            'Онлайн-сервисы', 'Едим дома', 'Смотрим дома', 'Играем дома'
        ]
        
        # Категории для онлайн-сервисов
        self.online_categories = ['Онлайн-сервисы', 'Едим дома', 'Смотрим дома', 'Играем дома']
        
        # Типы валютных операций
        self.fx_types = ['fx_buy', 'fx_sell']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def calculate_basic_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет базовых агрегатов по клиентам
        
        Args:
            df: DataFrame с транзакциями
            
        Returns:
            DataFrame с агрегированными признаками
        """
        logger.info("Расчет базовых агрегатов")
        
        # Группировка по клиенту
        client_features = df.groupby('client_code').agg({
            'name': 'first',
            'status': 'first',
            'age': 'first',
            'city': 'first',
            'avg_monthly_balance_KZT': 'mean',
            'amount_kzt': ['sum', 'count', 'mean', 'std'],
            'date': ['min', 'max']
        }).reset_index()
        
        # Переименование колонок
        client_features.columns = [
            'client_code', 'name', 'status', 'age', 'city', 'avg_monthly_balance_KZT',
            'total_amount', 'transaction_count', 'avg_transaction', 'std_transaction',
            'first_transaction', 'last_transaction'
        ]
        
        # Расчет количества дней активности
        client_features['days_active'] = (
            client_features['last_transaction'] - client_features['first_transaction']
        ).dt.days + 1
        
        # Коэффициент вариации (volatility)
        client_features['volatility'] = np.where(
            client_features['avg_transaction'] > 0,
            client_features['std_transaction'] / client_features['avg_transaction'],
            0
        )
        client_features['volatility'] = client_features['volatility'].fillna(0)
        
        return client_features
    
    def calculate_category_spending(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет трат по категориям
        
        Args:
            df: DataFrame с транзакциями
            
        Returns:
            DataFrame с тратами по категориям
        """
        logger.info("Расчет трат по категориям")
        
        # Фильтруем только расходы
        expenses = df[df['direction'] == 'out'].copy()
        
        # Агрегация по клиенту и категории
        category_spending = expenses.groupby(['client_code', 'category'])['amount_kzt'].sum().reset_index()
        category_spending = category_spending.pivot(
            index='client_code',
            columns='category',
            values='amount_kzt'
        ).fillna(0).reset_index()
        
        # Добавляем префикс к названиям колонок
        category_columns = [col for col in category_spending.columns if col != 'client_code']
        for col in category_columns:
            category_spending.rename(columns={col: f'spend_{col}'}, inplace=True)
        
        # Расчет общих трат
        spend_columns = [col for col in category_spending.columns if col.startswith('spend_')]
        category_spending['total_spend'] = category_spending[spend_columns].sum(axis=1)
        
        return category_spending
    
    def calculate_fx_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет признаков по валютным операциям
        
        Args:
            df: DataFrame с транзакциями
            
        Returns:
            DataFrame с валютными признаками
        """
        logger.info("Расчет валютных признаков")
        
        fx_features = pd.DataFrame()
        fx_features['client_code'] = df['client_code'].unique()
        
        # FX операции
        fx_transactions = df[df['type'].isin(self.fx_types)]
        fx_volume = fx_transactions.groupby('client_code')['amount_kzt'].sum().reset_index()
        fx_volume.columns = ['client_code', 'fx_operations_volume']
        
        # Траты в иностранной валюте
        foreign_spending = df[(df['currency'].isin(['USD', 'EUR'])) & (df['direction'] == 'out')]
        foreign_volume = foreign_spending.groupby('client_code')['amount_kzt'].sum().reset_index()
        foreign_volume.columns = ['client_code', 'foreign_spending_volume']
        
        # Объединяем
        fx_features = fx_features.merge(fx_volume, on='client_code', how='left')
        fx_features = fx_features.merge(foreign_volume, on='client_code', how='left')
        fx_features = fx_features.fillna(0)
        
        # Общий валютный объем
        fx_features['fx_volume'] = (
            fx_features['fx_operations_volume'] + 
            fx_features['foreign_spending_volume']
        )
        
        return fx_features
    
    def calculate_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет признаков по входящим/исходящим потокам
        
        Args:
            df: DataFrame с транзакциями
            
        Returns:
            DataFrame с признаками потоков
        """
        logger.info("Расчет признаков потоков")
        
        flow_features = pd.DataFrame()
        flow_features['client_code'] = df['client_code'].unique()
        
        # Входящие потоки
        inflows = df[df['direction'] == 'in'].groupby('client_code')['amount_kzt'].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        inflows.columns = ['client_code', 'inflows', 'inflow_count', 'avg_inflow']
        
        # Исходящие потоки
        outflows = df[df['direction'] == 'out'].groupby('client_code')['amount_kzt'].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        outflows.columns = ['client_code', 'outflows', 'outflow_count', 'avg_outflow']
        
        # Объединяем
        flow_features = flow_features.merge(inflows, on='client_code', how='left')
        flow_features = flow_features.merge(outflows, on='client_code', how='left')
        flow_features = flow_features.fillna(0)
        
        # Соотношение расходов к доходам
        flow_features['outflow_inflow_ratio'] = np.where(
            flow_features['inflows'] > 0,
            flow_features['outflows'] / flow_features['inflows'],
            0
        )
        
        # Чистый поток
        flow_features['net_flow'] = flow_features['inflows'] - flow_features['outflows']
        
        return flow_features
    
    def calculate_atm_p2p_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет признаков по банкоматам и P2P переводам
        
        Args:
            df: DataFrame с транзакциями
            
        Returns:
            DataFrame с признаками ATM и P2P
        """
        logger.info("Расчет признаков ATM и P2P")
        
        atm_p2p_features = pd.DataFrame()
        atm_p2p_features['client_code'] = df['client_code'].unique()
        
        # ATM снятия
        atm_transactions = df[df['type'] == 'atm_withdrawal']
        atm_stats = atm_transactions.groupby('client_code')['amount_kzt'].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        atm_stats.columns = ['client_code', 'atm_total', 'atm_count', 'atm_avg']
        
        # P2P переводы
        p2p_transactions = df[df['type'] == 'p2p_out']
        p2p_stats = p2p_transactions.groupby('client_code')['amount_kzt'].agg([
            'sum', 'count', 'mean'
        ]).reset_index()
        p2p_stats.columns = ['client_code', 'p2p_total', 'p2p_count', 'p2p_avg']
        
        # Объединяем
        atm_p2p_features = atm_p2p_features.merge(atm_stats, on='client_code', how='left')
        atm_p2p_features = atm_p2p_features.merge(p2p_stats, on='client_code', how='left')
        atm_p2p_features = atm_p2p_features.fillna(0)
        
        return atm_p2p_features
    
    def calculate_special_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет специальных признаков для определенных продуктов
        
        Args:
            df: DataFrame с транзакциями
            
        Returns:
            DataFrame со специальными признаками
        """
        logger.info("Расчет специальных признаков")
        
        special_features = pd.DataFrame()
        special_features['client_code'] = df['client_code'].unique()
        
        # Признак наличия кредитных операций
        credit_types = ['installment_payment_out', 'cc_repayment_out']
        has_credit = df[df['type'].isin(credit_types)].groupby('client_code')['amount_kzt'].sum().reset_index()
        has_credit.columns = ['client_code', 'credit_payments']
        has_credit['has_credit_activity'] = has_credit['credit_payments'] > 0
        
        # Топ-3 категории трат для каждого клиента
        # Исключаем технические категории из переводов
        excluded_categories = ['Оплата картой', 'Переводы', 'Снятие наличных', 'Доход', 
                              'Кешбэк', 'Возврат', 'Пополнение', 'Коммунальные услуги',
                              'Кредит', 'Кредитная карта', 'Рассрочка', 'Обмен валют',
                              'Инвестиции', 'Депозит', 'Золото', 'Прочее']
        
        expenses = df[(df['direction'] == 'out') & (~df['category'].isin(excluded_categories))]
        top_categories = expenses.groupby(['client_code', 'category'])['amount_kzt'].sum().reset_index()
        
        def get_top_categories(group):
            top3 = group.nlargest(3, 'amount_kzt')
            result = {
                'top_category_1': top3.iloc[0]['category'] if len(top3) > 0 else '',
                'top_category_1_amount': top3.iloc[0]['amount_kzt'] if len(top3) > 0 else 0,
                'top_category_2': top3.iloc[1]['category'] if len(top3) > 1 else '',
                'top_category_2_amount': top3.iloc[1]['amount_kzt'] if len(top3) > 1 else 0,
                'top_category_3': top3.iloc[2]['category'] if len(top3) > 2 else '',
                'top_category_3_amount': top3.iloc[2]['amount_kzt'] if len(top3) > 2 else 0,
            }
            return pd.Series(result)
        
        top_cats_df = top_categories.groupby('client_code').apply(get_top_categories).reset_index()
        
        # Стабильность баланса (для депозитов)
        balance_stability = df.groupby('client_code')['avg_monthly_balance_KZT'].agg(['mean', 'std']).reset_index()
        balance_stability['stability_score'] = np.where(
            balance_stability['mean'] > 0,
            1 - np.minimum(balance_stability['std'] / balance_stability['mean'], 1),
            0
        )
        balance_stability = balance_stability[['client_code', 'stability_score']]
        
        # Объединяем все специальные признаки
        special_features = special_features.merge(has_credit[['client_code', 'credit_payments', 'has_credit_activity']], 
                                                   on='client_code', how='left')
        special_features = special_features.merge(top_cats_df, on='client_code', how='left')
        special_features = special_features.merge(balance_stability, on='client_code', how='left')
        special_features = special_features.fillna(0)
        
        # Заполняем пустые категории
        for col in ['top_category_1', 'top_category_2', 'top_category_3']:
            special_features[col] = special_features[col].fillna('')
        
        return special_features
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание всех признаков
        
        Args:
            df: DataFrame с предобработанными транзакциями
            
        Returns:
            DataFrame с агрегированными признаками по клиентам
        """
        logger.info("Создание всех признаков")
        
        # Базовые агрегаты
        features = self.calculate_basic_aggregates(df)
        
        # Траты по категориям
        category_features = self.calculate_category_spending(df)
        features = features.merge(category_features, on='client_code', how='left')
        
        # Валютные признаки
        fx_features = self.calculate_fx_features(df)
        features = features.merge(fx_features, on='client_code', how='left')
        
        # Признаки потоков
        flow_features = self.calculate_flow_features(df)
        features = features.merge(flow_features, on='client_code', how='left')
        
        # ATM и P2P признаки
        atm_p2p_features = self.calculate_atm_p2p_features(df)
        features = features.merge(atm_p2p_features, on='client_code', how='left')
        
        # Специальные признаки
        special_features = self.calculate_special_features(df)
        features = features.merge(special_features, on='client_code', how='left')
        
        # Заполняем пропуски нулями
        features = features.fillna(0)
        
        # Расчет дополнительных признаков
        features['avg_check'] = features['avg_outflow']  # средний чек
        
        # Ежемесячные пополнения (для накопительного вклада)
        features['monthly_topups'] = features['inflows'] / 3  # делим на количество месяцев
        
        # Свободные средства (для инвестиций и депозитов)
        # Учитываем средний баланс и чистый поток, но оставляем резерв на расходы
        monthly_expenses = features['total_spend'] / 3  # Средние расходы в месяц
        features['free_funds'] = features['avg_monthly_balance_KZT'] - (monthly_expenses * 0.5)  # Оставляем 50% на текущие расходы
        features['free_funds'] = features['free_funds'].clip(lower=0)  # Не может быть отрицательным
        
        logger.info(f"Создано {len(features.columns)} признаков для {len(features)} клиентов")
        
        return features


def main():
    """Тестовый запуск расчета признаков"""
    from etl import DataLoader
    
    # Загружаем и предобрабатываем данные
    loader = DataLoader()
    
    # Создаем расширенные тестовые данные
    test_data = pd.DataFrame({
        'client_code': [1]*10 + [2]*8 + [3]*5,
        'name': ['Иван']*10 + ['Мария']*8 + ['Петр']*5,
        'status': ['Премиальный клиент']*10 + ['Студент']*8 + ['Зарплатный клиент']*5,
        'age': [35]*10 + [22]*8 + [45]*5,
        'city': ['Алматы']*10 + ['Астана']*8 + ['Шымкент']*5,
        'avg_monthly_balance_KZT': [500000]*10 + [50000]*8 + [200000]*5,
        'date': pd.date_range('2024-07-01', periods=10, freq='W').tolist() + 
                pd.date_range('2024-07-01', periods=8, freq='W').tolist() + 
                pd.date_range('2024-07-01', periods=5, freq='W').tolist(),
        'category': ['Рестораны', 'Такси', 'Отели', 'Продукты', 'Ювелирные изделия',
                    'Косметика и парфюмерия', 'Онлайн-сервисы', 'Путешествия', 'Такси', 'Рестораны'] +
                   ['Продукты', 'Транспорт', 'Образование', 'Продукты', 'Транспорт', 
                    'Онлайн-сервисы', 'Едим дома', 'Продукты'] +
                   ['Путешествия', 'Отели', 'Такси', 'Рестораны', 'Продукты'],
        'amount': [15000, 5000, 80000, 10000, 50000, 25000, 3000, 150000, 4000, 20000] +
                 [5000, 2000, 15000, 4000, 1500, 2000, 3000, 6000] +
                 [100000, 60000, 8000, 12000, 15000],
        'currency': ['KZT']*10 + ['KZT', 'KZT', 'USD', 'KZT', 'KZT', 'KZT', 'EUR', 'KZT'] + ['EUR', 'USD', 'KZT', 'KZT', 'KZT'],
        'type': ['card_out', 'card_out', 'card_out', 'card_out', 'card_out',
                'card_out', 'card_out', 'fx_buy', 'atm_withdrawal', 'card_out'] +
               ['card_out', 'p2p_out', 'stipend_in', 'card_out', 'p2p_out',
                'card_out', 'card_out', 'card_out'] +
               ['fx_buy', 'card_out', 'card_out', 'card_out', 'card_out'],
        'direction': ['out']*7 + ['out', 'out', 'out'] + ['out', 'out', 'in', 'out', 'out', 'out', 'out', 'out'] +
                     ['out']*5
    })
    
    # Сохраняем тестовые данные
    test_data.to_csv('data/test_transactions.csv', index=False)
    
    # Загружаем и обрабатываем
    processed_df = loader.preprocess('data/test_transactions.csv')
    
    # Создаем признаки
    feature_eng = FeatureEngineering()
    features_df = feature_eng.create_features(processed_df)
    
    print("\nАгрегированные признаки клиентов:")
    print(features_df[['client_code', 'name', 'total_spend', 'fx_volume', 'outflow_inflow_ratio']].head())
    print(f"\nВсего создано признаков: {len(features_df.columns)}")
    print(f"Клиентов обработано: {len(features_df)}")


if __name__ == "__main__":
    main()
