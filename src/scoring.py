"""
Модуль для расчета benefit score различных банковских продуктов
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


class BenefitScoring:
    """Класс для расчета benefit score различных продуктов"""
    
    def __init__(self, config_path: str = "conf/weights.yaml"):
        """
        Инициализация
        
        Args:
            config_path: путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def benefit_travel_card(self, features: pd.Series) -> float:
        """
        Расчет benefit для карты для путешествий
        
        Formula: 0.04 * (S(Путешествия) + S(Такси) + S(Отели/Билеты)), cap 100k
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['travel_card']
        
        # Суммируем траты по релевантным категориям
        travel_spend = 0
        for category in config['categories']:
            column_name = f'spend_{category}'
            if column_name in features:
                travel_spend += features[column_name]
        
        # Расчет benefit с учетом кешбэка
        benefit = config['cashback_rate'] * travel_spend
        
        # Применяем cap
        benefit = min(benefit, config['cap'])
        
        return benefit
    
    def benefit_premium_card(self, features: pd.Series) -> float:
        """
        Расчет benefit для премиальной карты
        
        Formula: tier_cashback(2/3/4%) * total + 0.04*(рестораны+ювелирка+косметика) + saved_fees(1%, cap 30k)
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['premium_card']
        
        # Определяем уровень кешбэка на основе общих трат
        total_spend = features.get('total_spend', 0)
        
        # Определяем tier
        if total_spend >= config['tier_thresholds']['tier4']:
            tier_rate = config['tier_cashback_rates']['tier4']
        elif total_spend >= config['tier_thresholds']['tier3']:
            tier_rate = config['tier_cashback_rates']['tier3']
        elif total_spend >= config['tier_thresholds']['tier2']:
            tier_rate = config['tier_cashback_rates']['tier2']
        else:
            tier_rate = config['tier_cashback_rates']['tier1']
        
        # Базовый кешбэк
        base_cashback = tier_rate * total_spend
        
        # Дополнительный кешбэк на специальные категории
        special_spend = 0
        for category in config['special_categories']:
            column_name = f'spend_{category}'
            if column_name in features:
                special_spend += features[column_name]
        
        special_cashback = config['special_cashback_rate'] * special_spend
        
        # Экономия на комиссиях (например, за снятие наличных)
        atm_total = features.get('atm_total', 0)
        saved_fees = min(config['saved_fees_rate'] * atm_total, config['saved_fees_cap'])
        
        # Общий benefit
        benefit = base_cashback + special_cashback + saved_fees
        
        # Применяем cap на кешбэк
        cashback_total = base_cashback + special_cashback
        if cashback_total > config['cashback_cap']:
            benefit = config['cashback_cap'] + saved_fees
        
        return benefit
    
    def benefit_credit_card(self, features: pd.Series) -> float:
        """
        Расчет benefit для кредитной карты
        
        Formula: 0.1*S(top3) + 0.1*S(онлайн), cap 100k
        Активируется при наличии installment/cc_repayment_out
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['credit_card']
        
        # Проверяем активацию (наличие кредитной активности)
        if not features.get('has_credit_activity', False):
            return 0
        
        # Кешбэк на топ-3 категории
        top3_spend = (
            features.get('top_category_1_amount', 0) +
            features.get('top_category_2_amount', 0) +
            features.get('top_category_3_amount', 0)
        )
        top3_cashback = config['top_categories_cashback'] * top3_spend
        
        # Кешбэк на онлайн-сервисы
        online_spend = 0
        for category in config['online_categories']:
            column_name = f'spend_{category}'
            if column_name in features:
                online_spend += features[column_name]
        
        online_cashback = config['online_cashback'] * online_spend
        
        # Общий benefit
        benefit = top3_cashback + online_cashback
        
        # Применяем cap
        benefit = min(benefit, config['cap'])
        
        return benefit
    
    def benefit_fx(self, features: pd.Series) -> float:
        """
        Расчет benefit для валютного счета
        
        Formula: 0.01 * fx_volume
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['fx']
        
        fx_volume = features.get('fx_volume', 0)
        benefit = config['cashback_rate'] * fx_volume
        
        return benefit
    
    def benefit_cash_loan(self, features: pd.Series) -> float:
        """
        Расчет benefit для кредита наличными
        
        Formula: если outflows >= 1.5*inflows and avg_balance < 50k, тогда 0.1*(outflows-inflows), иначе -inf
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['cash_loan']
        
        outflows = features.get('outflows', 0)
        inflows = features.get('inflows', 1)  # избегаем деления на 0
        avg_balance = features.get('avg_monthly_balance', 0)
        
        # Проверяем условия активации
        if (outflows >= config['outflow_multiplier'] * inflows and 
            avg_balance < config['balance_threshold']):
            benefit = config['benefit_rate'] * (outflows - inflows)
        else:
            benefit = float('-inf')  # Продукт не подходит
        
        return benefit
    
    def benefit_deposit_multicurrency(self, features: pd.Series) -> float:
        """
        Расчет benefit для мультивалютного депозита
        
        Formula: 0.145*(avg_balance+fx_volume)
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['deposits']['multicurrency']
        
        avg_balance = features.get('avg_monthly_balance', 0)
        fx_volume = features.get('fx_volume', 0)
        
        benefit = config['rate'] * (avg_balance + fx_volume)
        
        return benefit
    
    def benefit_deposit_savings(self, features: pd.Series) -> float:
        """
        Расчет benefit для сберегательного депозита
        
        Formula: 0.165*avg_balance*stability_score
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['deposits']['savings']
        
        avg_balance = features.get('avg_monthly_balance', 0)
        stability_score = features.get('stability_score', 0)
        
        benefit = config['rate'] * avg_balance * stability_score
        
        return benefit
    
    def benefit_deposit_accumulative(self, features: pd.Series) -> float:
        """
        Расчет benefit для накопительного депозита
        
        Formula: 0.155*(avg_balance+monthly_topups)
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['deposits']['accumulative']
        
        avg_balance = features.get('avg_monthly_balance', 0)
        monthly_topups = features.get('monthly_topups', 0)
        
        benefit = config['rate'] * (avg_balance + monthly_topups)
        
        return benefit
    
    def benefit_investments(self, features: pd.Series) -> float:
        """
        Расчет benefit для инвестиций
        
        Formula: если свободные средства >100k, тогда 0.01*avg_balance
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['investments']
        
        free_funds = features.get('free_funds', 0)
        avg_balance = features.get('avg_monthly_balance', 0)
        
        if free_funds > config['free_funds_threshold']:
            benefit = config['benefit_rate'] * avg_balance
        else:
            benefit = 0
        
        return benefit
    
    def benefit_gold(self, features: pd.Series) -> float:
        """
        Расчет benefit для золота
        
        Formula: если avg_balance высокий и есть траты на ювелирку, тогда 0.005*avg_balance
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['gold']
        
        avg_balance = features.get('avg_monthly_balance', 0)
        jewelry_spend = features.get('spend_Ювелирные изделия', 0)
        
        if avg_balance > config['balance_threshold'] and jewelry_spend > 0:
            benefit = config['benefit_rate'] * avg_balance
        else:
            benefit = 0
        
        return benefit
    
    def calculate_all_benefits(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет benefit для всех продуктов для всех клиентов
        
        Args:
            features: DataFrame с признаками клиентов
            
        Returns:
            DataFrame с benefit scores для всех продуктов
        """
        logger.info("Расчет benefit scores для всех продуктов")
        
        benefits = pd.DataFrame()
        benefits['client_code'] = features['client_code']
        
        # Список всех продуктов и соответствующих функций
        product_functions = {
            'Карта для путешествий': self.benefit_travel_card,
            'Премиальная карта': self.benefit_premium_card,
            'Кредитная карта': self.benefit_credit_card,
            'Валютный счет': self.benefit_fx,
            'Кредит наличными': self.benefit_cash_loan,
            'Мультивалютный депозит': self.benefit_deposit_multicurrency,
            'Сберегательный депозит': self.benefit_deposit_savings,
            'Накопительный депозит': self.benefit_deposit_accumulative,
            'Инвестиции': self.benefit_investments,
            'Золото': self.benefit_gold
        }
        
        # Рассчитываем benefit для каждого продукта
        for product_name, benefit_func in product_functions.items():
            logger.info(f"Расчет benefit для продукта: {product_name}")
            benefits[f'benefit_{product_name}'] = features.apply(benefit_func, axis=1)
        
        # Выбираем лучший депозит (так как клиенту нужен только один)
        deposit_columns = [
            'benefit_Мультивалютный депозит',
            'benefit_Сберегательный депозит',
            'benefit_Накопительный депозит'
        ]
        
        # Находим максимальный benefit среди депозитов
        benefits['benefit_Депозит'] = benefits[deposit_columns].max(axis=1)
        
        # Определяем, какой именно депозит лучший
        benefits['best_deposit_type'] = benefits[deposit_columns].idxmax(axis=1)
        benefits['best_deposit_type'] = benefits['best_deposit_type'].str.replace('benefit_', '')
        
        # Удаляем отдельные депозиты из итогового DataFrame
        benefits = benefits.drop(columns=deposit_columns)
        
        logger.info(f"Benefit scores рассчитаны для {len(benefits)} клиентов")
        
        return benefits
    
    def get_product_details(self, features: pd.DataFrame, benefits: pd.DataFrame) -> pd.DataFrame:
        """
        Получение дополнительных деталей для каждого продукта
        
        Args:
            features: DataFrame с признаками клиентов
            benefits: DataFrame с benefit scores
            
        Returns:
            DataFrame с дополнительными деталями
        """
        details = pd.DataFrame()
        details['client_code'] = features['client_code']
        
        # Детали для Travel Card
        details['travel_n_taxi'] = features['atm_count'].astype(int)  # количество поездок на такси
        details['travel_sum_taxi'] = features.get('spend_Такси', 0)
        details['travel_total_spend'] = (
            features.get('spend_Путешествия', 0) +
            features.get('spend_Такси', 0) +
            features.get('spend_Отели', 0) +
            features.get('spend_Билеты', 0)
        )
        
        # Детали для Premium Card
        total_spend = features.get('total_spend', 0)
        config = self.config['premium_card']
        
        # Определяем процент кешбэка
        details['premium_percent'] = features.apply(
            lambda row: 4 if row.get('total_spend', 0) >= config['tier_thresholds']['tier4']
            else 3 if row.get('total_spend', 0) >= config['tier_thresholds']['tier3']
            else 2.5 if row.get('total_spend', 0) >= config['tier_thresholds']['tier2']
            else 2,
            axis=1
        )
        details['premium_restaurant_spend'] = features.get('spend_Рестораны', 0)
        
        # Детали для Credit Card
        details['credit_cat1'] = features['top_category_1']
        details['credit_cat2'] = features['top_category_2']
        details['credit_cat3'] = features['top_category_3']
        
        # Детали для депозитов
        details['deposit_rate'] = benefits.apply(
            lambda row: 14.5 if row.get('best_deposit_type') == 'Мультивалютный депозит'
            else 16.5 if row.get('best_deposit_type') == 'Сберегательный депозит'
            else 15.5,
            axis=1
        )
        
        # Детали для других продуктов
        details['fx_volume'] = features['fx_volume']
        details['loan_amount'] = (features['outflows'] - features['inflows']).clip(lower=0)
        details['investment_min_amount'] = 100000
        details['investment_rate'] = 15  # примерная доходность
        details['gold_min_amount'] = 50000
        
        return details


def main():
    """Тестовый запуск scoring"""
    from etl import DataLoader
    from features import FeatureEngineering
    
    # Создаем тестовые данные
    test_features = pd.DataFrame({
        'client_code': [1, 2, 3],
        'name': ['Иван', 'Мария', 'Петр'],
        'status': ['Премиальный клиент', 'Студент', 'Зарплатный клиент'],
        'avg_monthly_balance': [500000, 50000, 200000],
        'total_spend': [362000, 38500, 195000],
        'spend_Рестораны': [35000, 0, 12000],
        'spend_Такси': [9000, 3500, 8000],
        'spend_Отели': [80000, 0, 60000],
        'spend_Путешествия': [150000, 0, 100000],
        'spend_Ювелирные изделия': [50000, 0, 0],
        'spend_Косметика и парфюмерия': [25000, 0, 0],
        'spend_Онлайн-сервисы': [3000, 5000, 0],
        'spend_Едим дома': [0, 3000, 0],
        'fx_volume': [150000, 31200, 232000],
        'inflows': [0, 15000, 0],
        'outflows': [362000, 38500, 195000],
        'outflow_inflow_ratio': [float('inf'), 2.57, float('inf')],
        'atm_total': [4000, 0, 0],
        'atm_count': [1, 0, 0],
        'has_credit_activity': [False, False, False],
        'top_category_1': ['Путешествия', 'Продукты', 'Путешествия'],
        'top_category_1_amount': [150000, 15000, 100000],
        'top_category_2': ['Отели', 'Образование', 'Отели'],
        'top_category_2_amount': [80000, 15000, 60000],
        'top_category_3': ['Ювелирные изделия', 'Онлайн-сервисы', 'Продукты'],
        'top_category_3_amount': [50000, 5000, 15000],
        'stability_score': [0.9, 0.7, 0.8],
        'monthly_topups': [0, 5000, 0],
        'free_funds': [138000, 26500, 5000]
    })
    
    # Рассчитываем benefit scores
    scoring = BenefitScoring()
    benefits_df = scoring.calculate_all_benefits(test_features)
    
    print("\nBenefit scores для продуктов:")
    print(benefits_df.head())
    
    # Получаем детали
    details_df = scoring.get_product_details(test_features, benefits_df)
    print("\nДетали для продуктов:")
    print(details_df[['client_code', 'travel_sum_taxi', 'premium_percent', 'deposit_rate']].head())


if __name__ == "__main__":
    main()
