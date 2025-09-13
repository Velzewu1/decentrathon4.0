"""
Тесты для модуля compose
"""
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем директорию src в path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from compose import PushComposer


class TestPushComposer(unittest.TestCase):
    """Тесты для класса PushComposer"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.composer = PushComposer()
        
        # Создаем тестовые данные клиента
        self.test_features = pd.Series({
            'client_code': 1,
            'name': 'Иван',
            'status': 'Премиальный клиент',
            'avg_monthly_balance': 500000,
            'spend_Такси': 27400,
            'spend_Рестораны': 35000,
            'spend_Отели': 80000,
            'spend_Путешествия': 150000,
            'spend_Онлайн-сервисы': 5000,
            'spend_Ювелирные изделия': 50000,
            'fx_volume': 150000,
            'outflow_inflow_ratio': 1.5,
            'top_category_1': 'Путешествия',
            'top_category_2': 'Отели',
            'top_category_3': 'Рестораны',
            'deposit_type': 'Мультивалютный депозит'
        })
        
        self.test_details = pd.Series({
            'client_code': 1,
            'travel_n_taxi': 12,
            'travel_sum_taxi': 27400,
            'travel_total_spend': 257400,
            'premium_percent': 4,
            'premium_restaurant_spend': 35000,
            'credit_cat1': 'Путешествия',
            'credit_cat2': 'Отели',
            'credit_cat3': 'Рестораны',
            'deposit_rate': 14.5,
            'fx_volume': 150000,
            'loan_amount': 100000,
            'investment_min_amount': 100000,
            'investment_rate': 15,
            'gold_min_amount': 50000
        })
    
    def test_format_currency(self):
        """Тест форматирования валюты"""
        # Проверяем форматирование чисел
        self.assertEqual(self.composer._format_currency(1000), "1 000")
        self.assertEqual(self.composer._format_currency(27400), "27 400")
        self.assertEqual(self.composer._format_currency(1500000), "1 500 000")
        self.assertEqual(self.composer._format_currency(0), "0")
    
    def test_get_month_name(self):
        """Тест получения названия месяца"""
        self.assertEqual(self.composer._get_month_name(1), "январе")
        self.assertEqual(self.composer._get_month_name(8), "августе")
        self.assertEqual(self.composer._get_month_name(12), "декабре")
        self.assertEqual(self.composer._get_month_name(13), "этом месяце")  # некорректный месяц
    
    def test_get_period_name(self):
        """Тест получения названия периода"""
        self.assertEqual(self.composer._get_period_name(3), "3 месяца")
        self.assertEqual(self.composer._get_period_name(6), "полгода")
        self.assertEqual(self.composer._get_period_name(12), "год")
        self.assertEqual(self.composer._get_period_name(5), "5 месяцев")
    
    def test_validate_push(self):
        """Тест валидации push-уведомлений"""
        # Тест на CAPS LOCK
        push_caps = "ПРИВЕТ МИР"
        validated = self.composer._validate_push(push_caps)
        self.assertNotEqual(validated, push_caps)
        self.assertTrue(validated[0].isupper())
        self.assertFalse(validated[1:].isupper())
        
        # Тест на множественные восклицательные знаки
        push_exclamations = "Привет!! Мир!!!"
        validated = self.composer._validate_push(push_exclamations)
        self.assertEqual(validated.count('!'), 1)
        
        # Тест на обычный текст
        push_normal = "Привет, мир! Как дела?"
        validated = self.composer._validate_push(push_normal)
        self.assertEqual(validated, push_normal)
    
    def test_shorten_push(self):
        """Тест укорачивания push-уведомлений"""
        # Создаем длинное сообщение
        long_push = "А" * 250  # Больше максимальной длины (220)
        shortened = self.composer._shorten_push(long_push)
        
        self.assertLessEqual(len(shortened), 220)
        self.assertTrue(shortened.endswith("... Открыть сейчас."))
        
        # Короткое сообщение не должно измениться
        short_push = "Короткое сообщение"
        shortened = self.composer._shorten_push(short_push)
        self.assertEqual(shortened, short_push)
    
    def test_compose_push_travel_card(self):
        """Тест генерации push для карты путешествий"""
        product = "Карта для путешествий"
        benefit = 10000
        
        push = self.composer.compose_push(
            self.test_features,
            self.test_details,
            product,
            benefit
        )
        
        # Проверяем, что push содержит нужные элементы
        self.assertIn("Иван", push)
        self.assertIn("такси", push.lower())
        self.assertIn("27 400", push)
        self.assertIn("10 000", push)
        self.assertLessEqual(len(push), 220)
    
    def test_compose_push_premium_card(self):
        """Тест генерации push для премиальной карты"""
        product = "Премиальная карта"
        benefit = 20000
        
        push = self.composer.compose_push(
            self.test_features,
            self.test_details,
            product,
            benefit
        )
        
        # Проверяем, что push содержит нужные элементы
        self.assertIn("Иван", push)
        self.assertTrue("4%" in push or "кешбэк" in push.lower())
        self.assertLessEqual(len(push), 220)
    
    def test_compose_push_credit_card(self):
        """Тест генерации push для кредитной карты"""
        product = "Кредитная карта"
        benefit = 15000
        
        push = self.composer.compose_push(
            self.test_features,
            self.test_details,
            product,
            benefit
        )
        
        # Проверяем, что push содержит топ категории
        self.assertIn("Иван", push)
        self.assertTrue(
            "Путешествия" in push or 
            "Отели" in push or 
            "Рестораны" in push or
            "10%" in push
        )
        self.assertLessEqual(len(push), 220)
    
    def test_compose_push_deposit(self):
        """Тест генерации push для депозита"""
        product = "Депозит"
        benefit = 72500
        
        push = self.composer.compose_push(
            self.test_features,
            self.test_details,
            product,
            benefit
        )
        
        # Проверяем, что push содержит процентную ставку
        self.assertIn("Иван", push)
        self.assertTrue("14" in push or "15" in push or "16" in push)
        self.assertIn("%", push)
        self.assertLessEqual(len(push), 220)
    
    def test_compose_push_fx(self):
        """Тест генерации push для валютного счета"""
        product = "Валютный счет"
        benefit = 1500
        
        push = self.composer.compose_push(
            self.test_features,
            self.test_details,
            product,
            benefit
        )
        
        # Проверяем, что push содержит информацию о валюте
        self.assertIn("Иван", push)
        self.assertTrue("валют" in push.lower() or "150 000" in push)
        self.assertLessEqual(len(push), 220)
    
    def test_compose_push_fallback(self):
        """Тест fallback шаблона при отсутствии данных"""
        # Создаем минимальные данные
        minimal_features = pd.Series({'name': 'Тест'})
        minimal_details = pd.Series({})
        
        push = self.composer.compose_push(
            minimal_features,
            minimal_details,
            "Неизвестный продукт",
            5000
        )
        
        # Проверяем, что используется fallback шаблон
        self.assertIn("Тест", push)
        self.assertIn("5 000", push)
        self.assertIn("Оформить", push)
        self.assertLessEqual(len(push), 220)
    
    def test_generate_all_pushes(self):
        """Тест генерации push-уведомлений для всех клиентов"""
        # Создаем данные для нескольких клиентов
        features_df = pd.DataFrame([
            self.test_features.to_dict(),
            {
                'client_code': 2,
                'name': 'Мария',
                'status': 'Студент',
                'avg_monthly_balance': 50000,
                'spend_Такси': 3500,
                'spend_Рестораны': 0,
                'spend_Онлайн-сервисы': 5000,
                'fx_volume': 10000,
                'top_category_1': 'Продукты',
                'top_category_2': 'Транспорт',
                'top_category_3': 'Онлайн-сервисы'
            }
        ])
        
        details_df = pd.DataFrame([
            self.test_details.to_dict(),
            {
                'client_code': 2,
                'travel_n_taxi': 2,
                'travel_sum_taxi': 3500,
                'premium_percent': 2,
                'credit_cat1': 'Продукты',
                'credit_cat2': 'Транспорт',
                'credit_cat3': 'Онлайн-сервисы',
                'deposit_rate': 16.5,
                'fx_volume': 10000,
                'loan_amount': 20000
            }
        ])
        
        ranked_products = pd.DataFrame({
            'client_code': [1, 2],
            'rank': [1, 1],
            'product': ['Карта для путешествий', 'Депозит'],
            'benefit': [10000, 8000],
            'deposit_type': [None, 'Сберегательный депозит']
        })
        
        # Генерируем push-уведомления
        result = self.composer.generate_all_pushes(features_df, details_df, ranked_products)
        
        # Проверяем результат
        self.assertEqual(len(result), 2)
        self.assertIn('client_code', result.columns)
        self.assertIn('product', result.columns)
        self.assertIn('push_notification', result.columns)
        
        # Проверяем, что push-уведомления не пустые
        for push in result['push_notification']:
            self.assertIsNotNone(push)
            self.assertGreater(len(push), 0)
            self.assertLessEqual(len(push), 220)


class TestPushComposerEdgeCases(unittest.TestCase):
    """Тесты граничных случаев для compose"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.composer = PushComposer()
    
    def test_empty_name(self):
        """Тест с пустым именем клиента"""
        features = pd.Series({'name': ''})
        details = pd.Series({})
        
        push = self.composer.compose_push(features, details, "Продукт", 1000)
        
        # Должен использовать значение по умолчанию
        self.assertIn("Клиент", push)
    
    def test_special_characters_in_name(self):
        """Тест со специальными символами в имени"""
        features = pd.Series({'name': 'Анна-Мария О\'Брайен'})
        details = pd.Series({})
        
        push = self.composer.compose_push(features, details, "Продукт", 1000)
        
        # Имя должно корректно отображаться
        self.assertIn("Анна-Мария О'Брайен", push)
    
    def test_zero_benefit(self):
        """Тест с нулевым benefit"""
        features = pd.Series({'name': 'Тест'})
        details = pd.Series({})
        
        push = self.composer.compose_push(features, details, "Продукт", 0)
        
        # Push должен быть сгенерирован даже с нулевым benefit
        self.assertIn("Тест", push)
        self.assertIn("0", push)
    
    def test_very_large_numbers(self):
        """Тест с очень большими числами"""
        features = pd.Series({
            'name': 'Тест',
            'spend_Такси': 999999999,
            'fx_volume': 1000000000
        })
        details = pd.Series({
            'travel_sum_taxi': 999999999,
            'fx_volume': 1000000000
        })
        
        push = self.composer.compose_push(
            features, details, 
            "Карта для путешествий", 
            40000000
        )
        
        # Проверяем, что большие числа корректно форматируются
        self.assertIn("999", push)
        self.assertLessEqual(len(push), 220)
    
    def test_missing_template_fields(self):
        """Тест с отсутствующими полями для шаблона"""
        features = pd.Series({'name': 'Тест'})
        details = pd.Series({})
        
        # Должен использовать fallback шаблон
        push = self.composer.compose_push(
            features, details,
            "Карта для путешествий",
            5000
        )
        
        # Проверяем, что push сгенерирован
        self.assertIn("Тест", push)
        self.assertIn("5 000", push)


if __name__ == '__main__':
    unittest.main()
