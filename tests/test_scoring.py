"""
Тесты для модуля scoring
"""
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем директорию src в path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from scoring import BenefitScoring


class TestBenefitScoring(unittest.TestCase):
    """Тесты для класса BenefitScoring"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.scoring = BenefitScoring()
        
        # Создаем тестовые данные клиента
        self.test_features = pd.Series({
            'client_code': 1,
            'name': 'Тест',
            'avg_monthly_balance': 500000,
            'total_spend': 300000,
            'spend_Такси': 10000,
            'spend_Путешествия': 50000,
            'spend_Отели': 30000,
            'spend_Билеты': 10000,
            'spend_Рестораны': 20000,
            'spend_Ювелирные изделия': 40000,
            'spend_Косметика и парфюмерия': 15000,
            'spend_Онлайн-сервисы': 5000,
            'spend_Едим дома': 3000,
            'fx_volume': 100000,
            'inflows': 400000,
            'outflows': 300000,
            'atm_total': 50000,
            'has_credit_activity': False,
            'top_category_1_amount': 50000,
            'top_category_2_amount': 40000,
            'top_category_3_amount': 30000,
            'stability_score': 0.8,
            'monthly_topups': 50000,
            'free_funds': 600000
        })
    
    def test_benefit_travel_card(self):
        """Тест расчета benefit для карты путешествий"""
        # Сумма трат: Такси (10k) + Путешествия (50k) + Отели (30k) + Билеты (10k) = 100k
        # Benefit: 0.04 * 100k = 4000
        benefit = self.scoring.benefit_travel_card(self.test_features)
        self.assertEqual(benefit, 4000)
        
        # Тест с cap
        features_high = self.test_features.copy()
        features_high['spend_Путешествия'] = 3000000
        benefit_high = self.scoring.benefit_travel_card(features_high)
        self.assertEqual(benefit_high, 100000)  # cap
    
    def test_benefit_premium_card(self):
        """Тест расчета benefit для премиальной карты"""
        # Total spend = 300k, tier = 2%
        # Base cashback: 0.02 * 300k = 6000
        # Special categories: Рестораны (20k) + Ювелирные (40k) + Косметика (15k) = 75k
        # Special cashback: 0.04 * 75k = 3000
        # Saved fees: 0.01 * 50k (atm) = 500
        # Total: 6000 + 3000 + 500 = 9500
        benefit = self.scoring.benefit_premium_card(self.test_features)
        self.assertEqual(benefit, 9500)
    
    def test_benefit_credit_card(self):
        """Тест расчета benefit для кредитной карты"""
        # Без кредитной активности должен вернуть 0
        benefit = self.scoring.benefit_credit_card(self.test_features)
        self.assertEqual(benefit, 0)
        
        # С кредитной активностью
        features_with_credit = self.test_features.copy()
        features_with_credit['has_credit_activity'] = True
        # Top3: 50k + 40k + 30k = 120k, cashback: 0.1 * 120k = 12000
        # Online: 5k + 3k = 8k, cashback: 0.1 * 8k = 800
        # Total: 12800
        benefit_with_credit = self.scoring.benefit_credit_card(features_with_credit)
        self.assertEqual(benefit_with_credit, 12800)
    
    def test_benefit_fx(self):
        """Тест расчета benefit для валютного счета"""
        # FX volume = 100k
        # Benefit: 0.01 * 100k = 1000
        benefit = self.scoring.benefit_fx(self.test_features)
        self.assertEqual(benefit, 1000)
    
    def test_benefit_cash_loan(self):
        """Тест расчета benefit для кредита наличными"""
        # outflows (300k) < 1.5 * inflows (400k * 1.5 = 600k) - не подходит
        benefit = self.scoring.benefit_cash_loan(self.test_features)
        self.assertEqual(benefit, float('-inf'))
        
        # Создаем условия для активации
        features_loan = self.test_features.copy()
        features_loan['outflows'] = 700000
        features_loan['inflows'] = 400000
        features_loan['avg_monthly_balance'] = 40000
        # Benefit: 0.1 * (700k - 400k) = 30000
        benefit_loan = self.scoring.benefit_cash_loan(features_loan)
        self.assertEqual(benefit_loan, 30000)
    
    def test_benefit_deposits(self):
        """Тест расчета benefit для депозитов"""
        # Мультивалютный: 0.145 * (500k + 100k) = 87000
        benefit_multi = self.scoring.benefit_deposit_multicurrency(self.test_features)
        self.assertEqual(benefit_multi, 87000)
        
        # Сберегательный: 0.165 * 500k * 0.8 = 66000
        benefit_savings = self.scoring.benefit_deposit_savings(self.test_features)
        self.assertEqual(benefit_savings, 66000)
        
        # Накопительный: 0.155 * (500k + 50k) = 85250
        benefit_accum = self.scoring.benefit_deposit_accumulative(self.test_features)
        self.assertEqual(benefit_accum, 85250)
    
    def test_benefit_investments(self):
        """Тест расчета benefit для инвестиций"""
        # free_funds (600k) > 100k threshold
        # Benefit: 0.01 * 500k = 5000
        benefit = self.scoring.benefit_investments(self.test_features)
        self.assertEqual(benefit, 5000)
        
        # Недостаточно свободных средств
        features_low = self.test_features.copy()
        features_low['free_funds'] = 50000
        benefit_low = self.scoring.benefit_investments(features_low)
        self.assertEqual(benefit_low, 0)
    
    def test_benefit_gold(self):
        """Тест расчета benefit для золота"""
        # avg_balance (500k) > 100k и есть траты на ювелирку (40k)
        # Benefit: 0.005 * 500k = 2500
        benefit = self.scoring.benefit_gold(self.test_features)
        self.assertEqual(benefit, 2500)
        
        # Без трат на ювелирку
        features_no_jewelry = self.test_features.copy()
        features_no_jewelry['spend_Ювелирные изделия'] = 0
        benefit_no_jewelry = self.scoring.benefit_gold(features_no_jewelry)
        self.assertEqual(benefit_no_jewelry, 0)
    
    def test_calculate_all_benefits(self):
        """Тест расчета всех benefit scores"""
        # Создаем DataFrame с несколькими клиентами
        features_df = pd.DataFrame([self.test_features.to_dict()])
        
        benefits = self.scoring.calculate_all_benefits(features_df)
        
        # Проверяем структуру результата
        self.assertIn('client_code', benefits.columns)
        self.assertIn('benefit_Карта для путешествий', benefits.columns)
        self.assertIn('benefit_Премиальная карта', benefits.columns)
        self.assertIn('benefit_Депозит', benefits.columns)
        self.assertIn('best_deposit_type', benefits.columns)
        
        # Проверяем, что выбран лучший депозит
        self.assertEqual(len(benefits), 1)
        self.assertEqual(benefits.iloc[0]['client_code'], 1)
        
        # Лучший депозит должен быть мультивалютным (87000)
        self.assertEqual(benefits.iloc[0]['benefit_Депозит'], 87000)
        self.assertEqual(benefits.iloc[0]['best_deposit_type'], 'Мультивалютный депозит')


class TestBenefitScoringEdgeCases(unittest.TestCase):
    """Тесты граничных случаев для scoring"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.scoring = BenefitScoring()
    
    def test_empty_features(self):
        """Тест с пустыми признаками"""
        empty_features = pd.Series({})
        
        # Все функции должны возвращать 0 или -inf для пустых данных
        self.assertEqual(self.scoring.benefit_travel_card(empty_features), 0)
        self.assertEqual(self.scoring.benefit_premium_card(empty_features), 0)
        self.assertEqual(self.scoring.benefit_fx(empty_features), 0)
    
    def test_zero_values(self):
        """Тест с нулевыми значениями"""
        zero_features = pd.Series({
            'avg_monthly_balance': 0,
            'total_spend': 0,
            'fx_volume': 0,
            'inflows': 0,
            'outflows': 0,
            'has_credit_activity': False,
            'stability_score': 0,
            'free_funds': 0
        })
        
        # Проверяем, что функции корректно обрабатывают нули
        self.assertEqual(self.scoring.benefit_travel_card(zero_features), 0)
        self.assertEqual(self.scoring.benefit_premium_card(zero_features), 0)
        self.assertEqual(self.scoring.benefit_fx(zero_features), 0)
        self.assertEqual(self.scoring.benefit_investments(zero_features), 0)
        self.assertEqual(self.scoring.benefit_gold(zero_features), 0)
    
    def test_tier_thresholds(self):
        """Тест уровней кешбэка для премиальной карты"""
        features = pd.Series({'total_spend': 0, 'atm_total': 0})
        
        # Tier 1: < 500k
        features['total_spend'] = 100000
        benefit1 = self.scoring.benefit_premium_card(features)
        self.assertEqual(benefit1, 2000)  # 0.02 * 100k
        
        # Tier 2: 500k - 1M
        features['total_spend'] = 600000
        benefit2 = self.scoring.benefit_premium_card(features)
        self.assertEqual(benefit2, 15000)  # 0.025 * 600k
        
        # Tier 3: 1M - 2M
        features['total_spend'] = 1500000
        benefit3 = self.scoring.benefit_premium_card(features)
        self.assertEqual(benefit3, 45000)  # 0.03 * 1.5M
        
        # Tier 4: > 2M
        features['total_spend'] = 2500000
        benefit4 = self.scoring.benefit_premium_card(features)
        self.assertEqual(benefit4, 100000)  # 0.04 * 2.5M, но cap = 100k


if __name__ == '__main__':
    unittest.main()
