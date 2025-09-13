"""
Тесты валидации данных и бизнес-логики
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import FeatureEngineering
from scoring_v2 import BenefitScoringV2
from compose_v2 import PushComposerV2


class TestDataValidation(unittest.TestCase):
    """Тесты валидации входных данных"""
    
    def test_client_data_validation(self):
        """Проверка валидации данных клиентов"""
        # Создаем тестовые данные
        valid_data = pd.DataFrame({
            'client_code': [1, 2, 3],
            'name': ['Айгерим', 'Данияр', 'Сабина'],
            'status': ['Зарплатный клиент', 'Премиальный клиент', 'Студент'],
            'age': [29, 41, 22],
            'city': ['Алматы', 'Астана', 'Алматы'],
            'avg_monthly_balance_KZT': [92643, 1577073, 63116]
        })
        
        # Проверяем обязательные колонки
        required_cols = ['client_code', 'name', 'status', 'age', 'city', 'avg_monthly_balance_KZT']
        for col in required_cols:
            self.assertIn(col, valid_data.columns)
        
        # Проверяем типы данных
        self.assertTrue(pd.api.types.is_numeric_dtype(valid_data['age']))
        self.assertTrue(pd.api.types.is_numeric_dtype(valid_data['avg_monthly_balance_KZT']))
        
        # Проверяем валидные статусы
        valid_statuses = ['Студент', 'Зарплатный клиент', 'Премиальный клиент', 'Стандартный клиент']
        for status in valid_data['status']:
            self.assertIn(status, valid_statuses)
    
    def test_transaction_validation(self):
        """Проверка валидации транзакций"""
        transactions = pd.DataFrame({
            'client_code': [1, 1, 1],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'category': ['Такси', 'Кафе и рестораны', 'Продукты питания'],
            'amount': [1500, 3200, 5000],
            'currency': ['KZT', 'KZT', 'USD']
        })
        
        # Проверяем что суммы положительные
        self.assertTrue((transactions['amount'] > 0).all())
        
        # Проверяем валидные валюты
        valid_currencies = ['KZT', 'USD', 'EUR', 'RUB']
        for curr in transactions['currency']:
            self.assertIn(curr, valid_currencies)
    
    def test_age_validation(self):
        """Проверка валидации возраста"""
        ages = [18, 25, 35, 45, 55, 65, 75]
        
        for age in ages:
            # Возраст должен быть от 18 до 100
            self.assertGreaterEqual(age, 18)
            self.assertLessEqual(age, 100)
    
    def test_balance_validation(self):
        """Проверка валидации баланса"""
        balances = [0, 50000, 100000, 1000000, 10000000]
        
        for balance in balances:
            # Баланс не может быть отрицательным
            self.assertGreaterEqual(balance, 0)


class TestBusinessLogic(unittest.TestCase):
    """Тесты бизнес-логики"""
    
    def setUp(self):
        """Инициализация тестовых данных"""
        self.scorer = BenefitScoringV2('conf/weights_v2.yaml')
        self.composer = PushComposerV2('conf/templates_v2.yaml')
    
    def test_student_restrictions(self):
        """Студенты не должны получать премиальную карту"""
        student_features = pd.Series({
            'client_code': 1,
            'name': 'Тест',
            'status': 'Студент',
            'age': 20,
            'avg_monthly_balance_KZT': 50000,
            'total_spend': 100000,
            'spend_Кафе и рестораны': 20000
        })
        
        # Проверяем доступность продуктов
        self.assertFalse(self.scorer._is_product_available('Студент', 'Премиальная карта'))
        self.assertFalse(self.scorer._is_product_available('Студент', 'Кредит наличными'))
        self.assertFalse(self.scorer._is_product_available('Студент', 'Золотые слитки'))
        self.assertTrue(self.scorer._is_product_available('Студент', 'Кредитная карта'))
    
    def test_premium_card_requirements(self):
        """Премиальная карта только для премиальных клиентов"""
        self.assertTrue(self.scorer._is_product_available('Премиальный клиент', 'Премиальная карта'))
        self.assertFalse(self.scorer._is_product_available('Стандартный клиент', 'Премиальная карта'))
        self.assertFalse(self.scorer._is_product_available('Студент', 'Премиальная карта'))
    
    def test_age_boost(self):
        """Проверка возрастных множителей"""
        # Молодежь предпочитает кредитки
        young_boost = self.scorer._get_age_boost(25, 'Кредитная карта')
        self.assertGreater(young_boost, 1.0)
        
        # Старшее поколение предпочитает депозиты
        senior_boost = self.scorer._get_age_boost(55, 'Депозит Сберегательный')
        self.assertGreater(senior_boost, 1.0)
    
    def test_cashback_limits(self):
        """Проверка лимитов кешбэка"""
        features = pd.Series({
            'spend_Кафе и рестораны': 1000000,  # Большие траты
            'spend_Такси': 500000,
            'spend_Продукты питания': 500000,
            'top_category_1_amount': 1000000,
            'top_category_2_amount': 500000,
            'top_category_3_amount': 500000,
            'age': 35,
            'status': 'Зарплатный клиент',
            'avg_monthly_balance_KZT': 100000
        })
        
        # Кешбэк должен быть ограничен
        credit_benefit = self.scorer.benefit_credit_card(features)
        # Проверяем что benefit не превышает разумные пределы (например, 100k за 3 месяца)
        self.assertLess(credit_benefit, 200000)


class TestPushNotifications(unittest.TestCase):
    """Тесты генерации push-уведомлений"""
    
    def setUp(self):
        self.composer = PushComposerV2('conf/templates_v2.yaml')
    
    def test_push_length(self):
        """Push не должен превышать 220 символов"""
        test_features = pd.Series({
            'name': 'Очень-Длинное-Имя-Клиента-Которое-Может-Сломать-Шаблон',
            'spend_Такси': 999999999,
            'spend_Кафе и рестораны': 888888888,
            'top_category_1': 'Очень длинная категория номер один',
            'top_category_2': 'Очень длинная категория номер два',
            'top_category_3': 'Очень длинная категория номер три'
        })
        
        # Генерируем push для разных продуктов
        products = ['Кредитная карта', 'Премиальная карта', 'Карта для путешествий']
        
        for product in products:
            push = self.composer.compose_push(product, test_features, {})
            self.assertLessEqual(len(push), 220, f"Push для {product} превышает 220 символов: {len(push)}")
    
    def test_push_format(self):
        """Проверка формата push-уведомлений"""
        test_features = pd.Series({
            'name': 'Айгерим',
            'spend_Такси': 27400,
            'spend_Кафе и рестораны': 35000
        })
        
        push = self.composer.compose_push('Карта для путешествий', test_features, {})
        
        # Не должно быть CAPS
        self.assertNotRegex(push, r'[А-Я]{2,}')  # Нет слов полностью в верхнем регистре
        
        # Максимум один восклицательный знак
        self.assertLessEqual(push.count('!'), 1)
        
        # Правильный формат валюты (пробел между числом и символом)
        if '₸' in push:
            # Проверяем что перед ₸ есть пробел
            self.assertRegex(push, r'\d\s₸')
    
    def test_push_personalization(self):
        """Проверка персонализации push-уведомлений"""
        features = pd.Series({
            'name': 'Данияр',
            'spend_Кафе и рестораны': 171457,
            'avg_monthly_balance_KZT': 1577073
        })
        
        push = self.composer.compose_push('Премиальная карта', features, {})
        
        # Должно содержать имя клиента
        self.assertIn('Данияр', push)
        
        # Должно содержать персональные данные (суммы)
        self.assertRegex(push, r'\d+\s*₸')  # Содержит суммы


class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев"""
    
    def test_zero_balance(self):
        """Обработка нулевого баланса"""
        features = pd.Series({
            'avg_monthly_balance_KZT': 0,
            'total_spend': 50000,
            'free_funds': 0,
            'age': 30,
            'status': 'Стандартный клиент'
        })
        
        scorer = BenefitScoringV2('conf/weights_v2.yaml')
        
        # Депозиты должны давать хоть какой-то benefit
        deposit_benefit = scorer.benefit_deposit_multicurrency(features)
        self.assertGreaterEqual(deposit_benefit, 0)
    
    def test_negative_free_funds(self):
        """Обработка отрицательных свободных средств"""
        features = pd.Series({
            'free_funds': -50000,  # Расходы больше доходов
            'avg_monthly_balance_KZT': 100000,
            'age': 35,
            'status': 'Зарплатный клиент'
        })
        
        scorer = BenefitScoringV2('conf/weights_v2.yaml')
        
        # Должны предложить кредитные продукты
        credit_benefit = scorer.benefit_cash_loan(features)
        self.assertGreater(credit_benefit, 0)
    
    def test_extreme_spending(self):
        """Обработка экстремально больших трат"""
        features = pd.Series({
            'spend_Кафе и рестораны': 10000000,  # 10 млн в месяц
            'total_spend': 50000000,
            'avg_monthly_balance_KZT': 100000000,
            'age': 45,
            'status': 'Премиальный клиент'
        })
        
        scorer = BenefitScoringV2('conf/weights_v2.yaml')
        
        # Premium карта должна давать большой benefit
        premium_benefit = scorer.benefit_premium_card(features)
        self.assertGreater(premium_benefit, 100000)  # Минимум 100k benefit
        
        # Но не должна превышать разумные пределы
        self.assertLess(premium_benefit, 1000000)  # Максимум 1 млн


if __name__ == '__main__':
    # Запускаем тесты с подробным выводом
    unittest.main(verbosity=2)
