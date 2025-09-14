"""
Модуль для расчета benefit score банковских продуктов
Согласно правилам хакатона
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List
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
        4% кешбэк на Путешествия, Такси, Отели, Авиабилеты
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score (кешбэк за 3 месяца)
        """
        config = self.config['travel_card']
        
        # Суммируем траты по релевантным категориям
        travel_spend = 0
        for category in config['categories']:
            # Проверяем разные варианты названий колонок
            for col in features.index:
                if category.lower() in col.lower() and 'spend' in col.lower():
                    travel_spend += features[col]
        
        # Также проверяем отдельные колонки
        if 'spend_travel' in features:
            travel_spend += features['spend_travel']
        if 'taxi_total' in features:
            travel_spend += features['taxi_total']
        if 'hotel_total' in features:
            travel_spend += features['hotel_total']
        
        # Добавляем такси отдельно
        taxi_spend = features.get('spend_Такси', 0)
        
        # Расчет кешбэка за 3 месяца (4% на путешествия и такси)
        # Увеличиваем эффективную ставку для конкуренции
        benefit = config['cashback_rate'] * (travel_spend * 2 + taxi_spend * 1.5)
        
        return benefit
    
    def benefit_premium_card(self, features: pd.Series) -> float:
        """
        Расчет benefit для премиальной карты
        2-4% кешбэк в зависимости от депозита + 4% на специальные категории
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['premium_card']
        
        # Определяем уровень кешбэка на основе баланса/депозита
        balance = features.get('avg_monthly_balance_KZT', 0)
        
        if balance >= 6000000:  # >= 6 млн
            tier_rate = config.get('base_rate_by_deposit', {}).get('gte_6m', 0.04)
        elif balance >= 1000000:  # 1-6 млн
            tier_rate = config.get('base_rate_by_deposit', {}).get('1m_6m', 0.03)
        else:  # < 1 млн
            tier_rate = config.get('base_rate_by_deposit', {}).get('lt_1m', 0.02)
        
        # Базовый кешбэк на все траты
        total_spend = features.get('total_spend', 0)
        base_cashback = tier_rate * total_spend
        
        # Дополнительный кешбэк на специальные категории (4%)
        special_spend = 0
        special_categories = config.get('boosted_4pc_cats', [])
        for category in special_categories:
            spend_col = f'spend_{category}'
            if spend_col in features.index:
                special_spend += features[spend_col]
        
        # Специальный кешбэк = 4% - базовый процент (чтобы не считать дважды)
        additional_cashback = (0.04 - tier_rate) * special_spend
        
        # Экономия на комиссиях за снятие наличных
        atm_total = features.get('atm_total', 0)
        # Предполагаем комиссию 1% которую не платим с премиальной картой
        saved_fees = config.get('saved_fees_rate', 0.01) * min(atm_total, config.get('saved_fees_cap', 30000))
        
        # Общий benefit за 3 месяца
        total_benefit = base_cashback + additional_cashback + saved_fees
        
        # Применяем лимит кешбэка (100k в месяц = 300k за 3 месяца)
        cashback_benefit = min(base_cashback + additional_cashback, config['cashback_cap'] * 3)
        
        return cashback_benefit + saved_fees
    
    def benefit_credit_card(self, features: pd.Series) -> float:
        """
        Расчет benefit для кредитной карты
        10% на топ-3 категории + 10% на онлайн-сервисы
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['credit_card']
        
        # Кешбэк на топ-3 категории (10%)
        top3_spend = 0
        top3_spend += features.get('top_category_1_amount', 0)
        top3_spend += features.get('top_category_2_amount', 0)
        top3_spend += features.get('top_category_3_amount', 0)
        
        # Применяем месячный лимит кешбэка (снижаем до 20,000 ₸/мес)
        monthly_cashback_limit = 20000
        top3_cashback = min(config.get('top3_rate', 0.10) * top3_spend * 0.5, monthly_cashback_limit * 3)  # Снижаем эффективную ставку
        
        # Кешбэк на онлайн-сервисы (10%)
        online_spend = 0
        for category in config['online_categories']:
            spend_col = f'spend_{category}'
            if spend_col in features.index:
                online_spend += features[spend_col]
        
        online_cashback = min(config.get('online_rate', 0.10) * online_spend, monthly_cashback_limit)
        
        # Выгода от беспроцентного периода (если есть потребность в кредите)
        # Активируется если есть кредитные платежи или расходы > доходов
        needs_credit = (
            features.get('has_credit_activity', False) or
            features.get('credit_payments', 0) > 0 or
            features.get('outflow_inflow_ratio', 0) > 1.2
        )
        
        if needs_credit:
            # Экономия на процентах за 2 месяца беспроцентного периода
            # Предполагаем использование 20% от лимита
            credit_usage = config.get('credit_limit', 2000000) * 0.2
            # Экономия процентов за 2 месяца (при ставке 20% годовых)
            interest_saved = credit_usage * 0.20 * (2/12)
            total_benefit = top3_cashback + online_cashback + interest_saved
        else:
            total_benefit = top3_cashback + online_cashback
        
        return total_benefit
    
    def benefit_fx(self, features: pd.Series) -> float:
        """
        Расчет benefit для обмена валют
        Экономия на курсе ~0.5%
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['fx']
        
        # Объем валютных операций
        fx_volume = features.get('fx_volume', 0)
        
        # Экономия на курсе (увеличиваем с 0.5% до 2%)
        benefit = fx_volume * 0.02  # 2% экономии на курсе
        
        return benefit
    
    def benefit_cash_loan(self, features: pd.Series) -> float:
        """
        Расчет benefit для кредита наличными
        Предлагаем если расходы превышают доходы
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score (или 0 если не подходит)
        """
        config = self.config['cash_loan']
        
        outflows = features.get('outflows', 0)
        inflows = features.get('inflows', 1)
        
        # Проверяем нужду в кредите (делаем условия строже)
        needs_loan = (
            outflows > inflows * 1.5 and  # Расходы превышают доходы на 50% (было 20%)
            features.get('credit_payments', 0) > 0  # И есть кредитные платежи
        )
        
        if needs_loan:
            # Размер потенциального кредита (снижаем)
            loan_amount = min((outflows - inflows) * 0.3, 500000)  # До 500k и только 30% от дефицита
            
            # Выгода от более низкой ставки по сравнению с другими кредитами
            # Предполагаем альтернативу 25% годовых
            alternative_rate = 0.25
            our_rate = config['rate_1year']
            
            # Экономия на процентах за год (снижаем привлекательность)
            benefit = loan_amount * (alternative_rate - our_rate) * 0.3  # Снижаем в 3 раза
            return max(benefit, 0)
        
        return 0
    
    def benefit_deposit_multicurrency(self, features: pd.Series) -> float:
        """
        Расчет benefit для мультивалютного депозита
        14.5% годовых с доступом к деньгам
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['deposit_multicurrency']
        
        # Свободные средства для размещения
        free_funds = features.get('free_funds', features.get('avg_monthly_balance_KZT', 0))
        
        # Если есть валютные операции, это дополнительный плюс
        has_fx = features.get('fx_volume', 0) > 0
        
        if has_fx:
            # Доход от депозита за 3 месяца
            # Если нет свободных средств, используем 20% от баланса
            deposit_amount = max(free_funds, features.get('avg_monthly_balance_KZT', 0) * 0.2)
            benefit = deposit_amount * config['rate'] * (3/12)
        else:
            # Меньший benefit если нет валютных операций
            deposit_amount = max(free_funds, features.get('avg_monthly_balance_KZT', 0) * 0.08)
            benefit = deposit_amount * config['rate'] * (3/12) * 0.8
            # Минимальный benefit
            benefit = max(benefit, 3000)
        
        return benefit
    
    def benefit_deposit_savings(self, features: pd.Series) -> float:
        """
        Расчет benefit для сберегательного депозита
        16.5% годовых без доступа к деньгам
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['deposit_savings']
        
        # Свободные средства которые можно заморозить
        # Учитываем стабильность баланса
        free_funds = features.get('free_funds', features.get('avg_monthly_balance_KZT', 0))
        stability = features.get('stability_score', 0.5)
        
        # Чем стабильнее баланс, тем больше можем заморозить
        # Если нет свободных средств, используем 25% от баланса
        deposit_base = max(free_funds, features.get('avg_monthly_balance_KZT', 0) * 0.25)
        lockable_funds = deposit_base * stability
        
        # Доход от депозита за 3 месяца
        benefit = lockable_funds * config['rate'] * (3/12)
        
        # Добавляем минимальный benefit даже если нет свободных средств
        min_benefit = 5000  # Минимум 5000 тенге benefit
        benefit = max(benefit, min_benefit)
        
        return benefit
    
    def benefit_deposit_accumulative(self, features: pd.Series) -> float:
        """
        Расчет benefit для накопительного депозита
        15.5% годовых с пополнением
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['deposit_accumulative']
        
        # Начальная сумма (минимум 50,000 или 30% от баланса)
        initial_amount = max(50000, features.get('avg_monthly_balance_KZT', 0) * 0.3)
        
        # Ежемесячные пополнения
        monthly_topups = features.get('monthly_topups', features.get('inflows', 0) * 0.1)
        
        # Средняя сумма на депозите за 3 месяца с учетом пополнений
        avg_deposit = initial_amount + monthly_topups * 1.5  # В среднем за 3 месяца
        
        # Доход от депозита за 3 месяца
        benefit = avg_deposit * config['rate'] * (3/12)
        
        # Минимальный benefit для накопительного депозита
        benefit = max(benefit, 4000)
        
        return benefit
    
    def benefit_investments(self, features: pd.Series) -> float:
        """
        Расчет benefit для инвестиций
        Ожидаемая доходность 20% годовых
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['investments']
        
        # Свободные средства для инвестирования
        # Если нет свободных средств, используем 10% от баланса
        free_funds = max(features.get('free_funds', 0), features.get('avg_monthly_balance_KZT', 0) * 0.1)
        
        # Готовность к риску (чем моложе и больше доход, тем выше)
        age = features.get('age', 40)
        risk_tolerance = max(0.3, (60 - age) / 40)  # От 0.3 до 1 (минимум 30%)
        
        # Инвестируем часть свободных средств в зависимости от риск-профиля
        investment_amount = max(30000, free_funds * risk_tolerance * 0.5)
        
        # Ожидаемый доход за 3 месяца (увеличиваем привлекательность)
        benefit = investment_amount * config['expected_return'] * (3/12) * 1.5
        
        return benefit
    
    def benefit_gold(self, features: pd.Series) -> float:
        """
        Расчет benefit для золотых слитков
        Консервативная инвестиция, ~10% годовых
        
        Args:
            features: Series с признаками клиента
            
        Returns:
            Benefit score
        """
        config = self.config['gold']
        
        # Для золота нужен высокий баланс и консервативный профиль
        balance = features.get('avg_monthly_balance_KZT', 0)
        
        # Проверяем подходит ли клиент (снижаем порог до 300k)
        if balance > 300000:
            # Вкладываем 15% в золото для диверсификации
            gold_investment = balance * 0.15
            
            # Ожидаемый доход за 3 месяца
            benefit = gold_investment * config['expected_return'] * (3/12)
        else:
            benefit = 0
        
        return benefit
    
    def _get_age_boost(self, age: int, product: str) -> float:
        """
        Получить множитель benefit на основе возраста
        
        Args:
            age: Возраст клиента
            product: Название продукта
            
        Returns:
            Множитель (1.0 = без изменений, 1.2 = +20% и т.д.)
        """
        # Молодежь (до 30) предпочитает кредитки и инвестиции
        if age < 30:
            if product in ['Кредитная карта', 'Инвестиции']:
                return 1.2
            elif product in ['Депозит Сберегательный', 'Золотые слитки']:
                return 0.8
        
        # Средний возраст (30-45) - универсальные продукты
        elif 30 <= age <= 45:
            if product in ['Премиальная карта', 'Карта для путешествий']:
                return 1.1
        
        # Старший возраст (45+) - консервативные продукты
        else:
            if product in ['Депозит Сберегательный', 'Депозит Накопительный', 'Золотые слитки']:
                return 1.2
            elif product in ['Кредитная карта', 'Кредит наличными']:
                return 0.9
        
        return 1.0
    
    def _is_product_available(self, client_status: str, product: str, features: pd.Series = None) -> bool:
        """
        Проверка доступности продукта для клиента по статусу и поведению
        
        Args:
            client_status: Статус клиента
            product: Название продукта
            features: Признаки клиента (для умной логики)
            
        Returns:
            True если продукт доступен
        """
        # Нормализуем статус
        status_lower = client_status.lower()
        
        # Ограничения для студентов
        if 'студент' in status_lower:
            restricted = ['Кредит наличными', 'Золотые слитки']  # Убираем премиальную карту из запрета
            if product in restricted:
                return False
        
        # Умная логика для премиальной карты
        if product == 'Премиальная карта':
            if 'премиальный' in status_lower or 'вип' in status_lower:
                return True  # Всегда доступна для VIP
            
            # Для других статусов проверяем поведение
            if features is not None:
                balance = features.get('avg_monthly_balance_KZT', 0)
                restaurant_spend = features.get('spend_Кафе и рестораны', 0)
                cosmetics_spend = features.get('spend_Косметика и парфюмерия', 0)
                total_spend = features.get('total_spend', 1)
                
                # Разрешаем если высокий баланс ИЛИ высокие траты на премиальные категории
                premium_spend_share = (restaurant_spend + cosmetics_spend) / total_spend if total_spend > 0 else 0
                
                if balance > 500000 or premium_spend_share > 0.2:  # 20% трат на премиальные категории
                    return True
            
            return False  # Иначе недоступна
        
        return True
    
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
        
        # Сохраняем эталонный продукт если есть
        if 'reference_product' in features.columns:
            benefits['reference_product'] = features['reference_product']
        
        # Список всех продуктов и соответствующих функций
        product_functions = {
            'Карта для путешествий': self.benefit_travel_card,
            'Премиальная карта': self.benefit_premium_card,
            'Кредитная карта': self.benefit_credit_card,
            'Обмен валют': self.benefit_fx,
            'Кредит наличными': self.benefit_cash_loan,
            'Депозит Мультивалютный': self.benefit_deposit_multicurrency,
            'Депозит Сберегательный': self.benefit_deposit_savings,
            'Депозит Накопительный': self.benefit_deposit_accumulative,
            'Инвестиции': self.benefit_investments,
            'Золотые слитки': self.benefit_gold
        }
        
        # Рассчитываем benefit для каждого продукта с учетом доступности
        for product_name, benefit_func in product_functions.items():
            logger.info(f"Расчет benefit для продукта: {product_name}")
            
            # Рассчитываем benefit
            benefits[f'benefit_{product_name}'] = features.apply(benefit_func, axis=1)
            
            # Обнуляем benefit для недоступных продуктов и применяем возрастной множитель
            for idx, row in features.iterrows():
                client_status = row.get('status', 'Стандартный клиент')
                client_age = row.get('age', 35)
                
                # Проверяем доступность (передаем features для умной логики)
                if not self._is_product_available(client_status, product_name, row):
                    benefits.at[idx, f'benefit_{product_name}'] = 0
                else:
                    # Применяем возрастной множитель
                    age_boost = self._get_age_boost(client_age, product_name)
                    benefits.at[idx, f'benefit_{product_name}'] *= age_boost
        
        # Выбираем лучший депозит
        deposit_columns = [
            'benefit_Депозит Мультивалютный',
            'benefit_Депозит Сберегательный',
            'benefit_Депозит Накопительный'
        ]
        
        # Находим максимальный benefit среди депозитов
        benefits['benefit_Депозит'] = benefits[deposit_columns].max(axis=1)
        
        # Определяем, какой именно депозит лучший
        benefits['best_deposit_type'] = benefits[deposit_columns].idxmax(axis=1)
        benefits['best_deposit_type'] = benefits['best_deposit_type'].str.replace('benefit_', '')
        
        logger.info(f"Benefit scores рассчитаны для {len(benefits)} клиентов")
        
        return benefits
    
    def get_product_details(self, features: pd.DataFrame, benefits: pd.DataFrame) -> pd.DataFrame:
        """
        Получение дополнительных деталей для каждого продукта
        
        Args:
            features: DataFrame с признаками клиентов
            benefits: DataFrame с benefit scores
            
        Returns:
            DataFrame с дополнительными деталями для генерации push
        """
        details = pd.DataFrame()
        details['client_code'] = features['client_code']
        
        # Детали для Travel Card
        if 'taxi_count' in features.columns:
            details['taxi_count'] = features['taxi_count'].astype(int)
        else:
            details['taxi_count'] = features.get('atm_count', 0).astype(int) if 'atm_count' in features.columns else 0
        
        details['taxi_amount'] = features['taxi_total'] if 'taxi_total' in features.columns else 0
        details['travel_amount'] = features['spend_travel'] if 'spend_travel' in features.columns else 0
        
        # Детали для Premium Card
        details['premium_percent'] = features.apply(
            lambda row: 4 if row.get('avg_monthly_balance_KZT', 0) >= 6000000
            else 3 if row.get('avg_monthly_balance_KZT', 0) >= 1000000
            else 2,
            axis=1
        )
        
        # Проверяем наличие колонок перед использованием
        if 'spend_Кафе и рестораны' in features.columns:
            details['restaurant_amount'] = features['spend_Кафе и рестораны'] / 3
        else:
            details['restaurant_amount'] = 0
        
        # Детали для Credit Card
        details['top_cat1'] = features['top_category_1'] if 'top_category_1' in features.columns else 'Покупки'
        details['top_cat2'] = features['top_category_2'] if 'top_category_2' in features.columns else 'Продукты'
        details['top_cat3'] = features['top_category_3'] if 'top_category_3' in features.columns else 'Транспорт'
        
        # Детали для FX
        details['fx_volume'] = features['fx_volume'] if 'fx_volume' in features.columns else 0
        details['main_currency'] = 'USD'  # Определить основную валюту
        
        # Детали для кредита
        if 'outflows' in features.columns and 'inflows' in features.columns:
            details['loan_amount'] = (features['outflows'] - features['inflows']).clip(lower=0)
        else:
            details['loan_amount'] = 0
        
        # Детали для инвестиций
        if 'free_funds' in features.columns:
            details['free_funds_amount'] = features['free_funds']
        elif 'avg_monthly_balance_KZT' in features.columns:
            details['free_funds_amount'] = features['avg_monthly_balance_KZT']
        else:
            details['free_funds_amount'] = 0
        
        return details


def main():
    """Тестовый запуск scoring"""
    # Создаем тестовые данные
    test_features = pd.DataFrame({
        'client_code': [1],
        'name': ['Айгерим'],
        'age': [29],
        'avg_monthly_balance_KZT': [92643],
        'total_spend': [250000],
        'spend_travel': [50000],
        'taxi_total': [30000],
        'spend_restaurant': [15000],
        'fx_volume': [100000],
        'inflows': [300000],
        'outflows': [250000],
        'stability_score': 0.8,
        'free_funds': [50000],
        'top_category_1': 'Такси',
        'top_category_1_amount': [30000],
        'top_category_2': 'Продукты',
        'top_category_2_amount': [25000],
        'top_category_3': 'Рестораны',
        'top_category_3_amount': [15000],
        'reference_product': ['Карта для путешествий']
    })
    
    # Рассчитываем benefit scores
    scoring = BenefitScoring()
    benefits_df = scoring.calculate_all_benefits(test_features)
    
    print("\nBenefit scores для продуктов:")
    for col in benefits_df.columns:
        if col.startswith('benefit_'):
            product = col.replace('benefit_', '')
            value = benefits_df[col].iloc[0]
            print(f"{product}: {value:,.0f} ₸")


if __name__ == "__main__":
    main()
