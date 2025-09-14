"""
Улучшенный модуль ранжирования продуктов с бизнес-правилами
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
import random

logger = logging.getLogger(__name__)


class ProductRanker:
    """Улучшенный класс для ранжирования продуктов"""
    
    def __init__(self, config_path: str = "conf/weights.yaml"):
        """
        Инициализация ранжировщика
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Приоритеты продуктов для tie-breaking
        self.product_priorities = {
            'Премиальная карта': 1,
            'Карта для путешествий': 2,
            'Кредитная карта': 3,
            'Депозит Сберегательный': 4,
            'Депозит Накопительный': 5,
            'Депозит Мультивалютный': 6,
            'Инвестиции': 7,
            'Обмен валют': 8,
            'Золотые слитки': 9,
            'Кредит наличными': 10
        }
        
        # Минимальная квота для каждого продукта (% от всех рекомендаций)
        self.product_quotas = {
            'Карта для путешествий': 0.10,  # Минимум 10%
            'Премиальная карта': 0.15,
            'Кредитная карта': 0.15,
            'Депозит Сберегательный': 0.05,
            'Депозит Накопительный': 0.05,
            'Депозит Мультивалютный': 0.05,
            'Инвестиции': 0.05,
            'Обмен валют': 0.10,
            'Золотые слитки': 0.02,
            'Кредит наличными': 0.05
        }
        
        # Счетчики для отслеживания квот
        self.product_counts = {product: 0 for product in self.product_quotas.keys()}
        self.total_recommendations = 0
    
    def rank_products_for_clients(self, 
                                  benefits_df: pd.DataFrame,
                                  features_df: pd.DataFrame,
                                  top_n: int = 4) -> pd.DataFrame:
        """
        Ранжирование продуктов для всех клиентов с учетом бизнес-правил
        
        Args:
            benefits_df: DataFrame с benefit scores
            features_df: DataFrame с признаками клиентов
            top_n: Количество топ продуктов для каждого клиента
            
        Returns:
            DataFrame с ранжированными рекомендациями
        """
        logger.info("Ранжирование продуктов с бизнес-правилами")
        
        recommendations = []
        
        for idx, row in benefits_df.iterrows():
            client_code = row['client_code']
            client_features = features_df[features_df['client_code'] == client_code].iloc[0]
            
            # Получаем топ продукты для клиента
            client_recommendations = self._rank_for_client(
                row, client_features, top_n
            )
            
            recommendations.extend(client_recommendations)
        
        # Применяем квоты для разнообразия
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = self._apply_quotas(recommendations_df, benefits_df, features_df)
        
        logger.info(f"Ранжирование завершено для {len(benefits_df)} клиентов")
        return recommendations_df
    
    def _rank_for_client(self, 
                         benefits: pd.Series,
                         features: pd.Series,
                         top_n: int = 4) -> List[Dict]:
        """
        Ранжирование продуктов для одного клиента
        
        Args:
            benefits: Series с benefit scores клиента
            features: Series с признаками клиента
            top_n: Количество топ продуктов
            
        Returns:
            Список рекомендаций
        """
        client_code = benefits['client_code']
        
        # Извлекаем benefit scores
        benefit_cols = [col for col in benefits.index if col.startswith('benefit_')]
        product_benefits = {}
        
        for col in benefit_cols:
            product = col.replace('benefit_', '')
            score = benefits[col]
            
            # Применяем бизнес-правила для корректировки score
            adjusted_score = self._apply_business_rules(
                product, score, features
            )
            
            product_benefits[product] = adjusted_score
        
        # Депозиты обрабатываются как обычные продукты
        
        # Сортируем по score и приоритету
        sorted_products = sorted(
            product_benefits.items(),
            key=lambda x: (
                -x[1],  # По убыванию benefit
                self.product_priorities.get(x[0], 999)  # По приоритету при равных benefit
            )
        )
        
        # Формируем рекомендации
        recommendations = []
        for rank, (product, score) in enumerate(sorted_products[:top_n], 1):
            if score > 0:  # Только продукты с положительным benefit
                recommendations.append({
                    'client_code': client_code,
                    'product': product,
                    'benefit': score,
                    'rank': rank
                })
                
                # Обновляем счетчики
                if rank == 1:  # Считаем только главный продукт
                    self.product_counts[product] = self.product_counts.get(product, 0) + 1
                    self.total_recommendations += 1
        
        return recommendations
    
    def _apply_business_rules(self, 
                              product: str,
                              score: float,
                              features: pd.Series) -> float:
        """
        Применение бизнес-правил для корректировки score
        
        Args:
            product: Название продукта
            score: Исходный benefit score
            features: Признаки клиента
            
        Returns:
            Скорректированный score
        """
        adjusted_score = score
        
        # Правило 1: Усиливаем недопредставленные продукты
        if self.total_recommendations > 0:
            current_share = self.product_counts.get(product, 0) / self.total_recommendations
            target_share = self.product_quotas.get(product, 0.05)
            
            if current_share < target_share:
                # Усиливаем продукт если он недопредставлен
                boost = 1 + (target_share - current_share) * 2
                adjusted_score *= boost
        
        # Правило 2: Карта для путешествий только если такси/путешествия в топе
        if product == 'Карта для путешествий':
            taxi_spend = features.get('spend_Такси', 0)
            travel_spend = features.get('spend_Путешествия', 0)
            total_spend = features.get('total_spend', 1)
            
            # Доля трат на такси/путешествия от общих трат
            travel_share = (taxi_spend + travel_spend) / total_spend if total_spend > 0 else 0
            
            if travel_share > 0.25:  # Больше 25% трат на такси/путешествия
                adjusted_score *= 1.5
            elif travel_share < 0.15:  # Меньше 15% - сильно снижаем
                adjusted_score *= 0.3
        
        # Правило 3: Депозиты для тех, у кого есть свободные средства
        if 'Депозит' in product:
            free_funds = features.get('free_funds', 0)
            balance = features.get('avg_monthly_balance_KZT', 0)
            if free_funds > 0 or balance > 500000:
                adjusted_score *= 1.3  # Усиливаем на 30%
        
        # Правило 4: Инвестиции для молодежи
        if product == 'Инвестиции':
            age = features.get('age', 40)
            if age < 35:
                adjusted_score *= 1.4  # Усиливаем на 40%
        
        # Правило 5: Обмен валют для тех, кто делает валютные операции
        if product == 'Обмен валют':
            fx_volume = features.get('fx_volume', 0)
            if fx_volume > 100000:
                adjusted_score *= 1.5  # Усиливаем на 50%
        
        # Правило 6: Премиальная карта для высоких трат на рестораны/косметику
        if product == 'Премиальная карта':
            restaurant_spend = features.get('spend_Кафе и рестораны', 0)
            cosmetics_spend = features.get('spend_Косметика и парфюмерия', 0)
            total_spend = features.get('total_spend', 1)
            
            # Доля трат на рестораны + косметику
            premium_share = (restaurant_spend + cosmetics_spend) / total_spend if total_spend > 0 else 0
            
            if premium_share > 0.25:  # Больше 25% на премиальные категории
                adjusted_score *= 2.0  # Сильно усиливаем
        
        # Правило 7: Мультивалютный депозит только при FX операциях
        if product == 'Депозит Мультивалютный':
            fx_volume = features.get('fx_volume', 0)
            if fx_volume < 50000:  # Мало валютных операций
                adjusted_score *= 0.2  # Сильно снижаем
        
        # Правило 8: Снижаем кредит наличными (уже сделано в scoring)
        if product == 'Кредит наличными':
            adjusted_score *= 0.5  # Дополнительно снижаем на 50%
        
        return adjusted_score
    
    def _apply_quotas(self, 
                      recommendations_df: pd.DataFrame,
                      benefits_df: pd.DataFrame,
                      features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение квот для обеспечения разнообразия продуктов
        
        Args:
            recommendations_df: DataFrame с рекомендациями
            benefits_df: DataFrame с benefit scores
            features_df: DataFrame с признаками
            
        Returns:
            Скорректированный DataFrame
        """
        # Берем только топ-1 рекомендации
        top1_recommendations = recommendations_df[recommendations_df['rank'] == 1].copy()
        
        # Проверяем квоты
        total_clients = len(top1_recommendations)
        product_distribution = top1_recommendations['product'].value_counts()
        
        for product, min_quota in self.product_quotas.items():
            current_count = product_distribution.get(product, 0)
            min_count = int(total_clients * min_quota)
            
            if current_count < min_count:
                # Нужно добавить этот продукт некоторым клиентам
                needed = min_count - current_count
                
                # Находим клиентов, которым можно заменить рекомендацию
                candidates = self._find_replacement_candidates(
                    top1_recommendations, benefits_df, product, needed
                )
                
                # Заменяем рекомендации
                for client_code in candidates:
                    top1_recommendations.loc[
                        top1_recommendations['client_code'] == client_code, 
                        'product'
                    ] = product
                    
                    logger.debug(f"Заменена рекомендация для клиента {client_code} на {product}")
        
        return top1_recommendations
    
    def _find_replacement_candidates(self,
                                    recommendations_df: pd.DataFrame,
                                    benefits_df: pd.DataFrame,
                                    target_product: str,
                                    needed_count: int) -> List[int]:
        """
        Найти клиентов для замены рекомендации
        
        Args:
            recommendations_df: Текущие рекомендации
            benefits_df: Benefit scores
            target_product: Целевой продукт
            needed_count: Сколько нужно заменить
            
        Returns:
            Список client_code для замены
        """
        candidates = []
        
        # Находим клиентов с overrepresented продуктами
        product_counts = recommendations_df['product'].value_counts()
        overrepresented = product_counts[product_counts > len(recommendations_df) * 0.3]
        
        if len(overrepresented) > 0:
            # Берем клиентов с самым популярным продуктом
            most_common_product = overrepresented.index[0]
            potential_clients = recommendations_df[
                recommendations_df['product'] == most_common_product
            ]['client_code'].values
            
            # Проверяем, что у них есть хотя бы минимальный benefit для target_product
            for client_code in potential_clients[:needed_count]:
                client_benefits = benefits_df[benefits_df['client_code'] == client_code]
                if len(client_benefits) > 0:
                    target_benefit = client_benefits.iloc[0].get(f'benefit_{target_product}', 0)
                    if target_benefit > 0:
                        candidates.append(client_code)
        
        return candidates[:needed_count]


if __name__ == "__main__":
    # Тестирование
    print("Тестирование улучшенного ранжирования...")
    
    import sys
    sys.path.append('src')
    from etl import DataLoader
    from features import FeatureEngineering
    from scoring import BenefitScoring
    
    # Загружаем данные
    loader = DataLoader()
    clients_df, transactions_df = loader.load_all_data('data/clients.csv', 'data')
    
    # Создаем признаки
    fe = FeatureEngineering('conf/weights.yaml')
    features = fe.create_features(transactions_df)
    
    # Считаем benefits
    scorer = BenefitScoring('conf/weights.yaml')
    benefits = scorer.calculate_all_benefits(features)
    
    # Ранжируем с новыми правилами
    ranker = ProductRanker('conf/weights.yaml')
    recommendations = ranker.rank_products_for_clients(benefits, features, top_n=1)
    
    print("\nРаспределение продуктов:")
    print(recommendations['product'].value_counts())
    print(f"\nВсего уникальных продуктов: {len(recommendations['product'].unique())}")
