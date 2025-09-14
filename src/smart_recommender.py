"""
Умный рекомендательный модуль на основе реального поведения клиентов
"""
import pandas as pd
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class SmartRecommender:
    """Умный рекомендатор на основе анализа поведения"""
    
    def __init__(self):
        """Инициализация рекомендатора"""
        pass
    
    def recommend_for_clients(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация рекомендаций на основе реального поведения
        
        Args:
            features_df: DataFrame с признаками клиентов
            
        Returns:
            DataFrame с рекомендациями
        """
        logger.info("Генерация умных рекомендаций на основе поведения")
        
        recommendations = []
        
        for _, client in features_df.iterrows():
            product, reasoning = self._analyze_client_behavior(client)
            
            recommendations.append({
                'client_code': client['client_code'],
                'product': product,
                'reasoning': reasoning
            })
        
        return pd.DataFrame(recommendations)
    
    def _analyze_client_behavior(self, client: pd.Series) -> Tuple[str, str]:
        """
        Анализ поведения одного клиента для выбора продукта
        
        Args:
            client: Series с данными клиента
            
        Returns:
            Tuple (продукт, обоснование)
        """
        # Собираем данные о тратах
        spending_analysis = self._analyze_spending_patterns(client)
        
        # Данные клиента
        status = client.get('status', 'Стандартный клиент')
        balance = client.get('avg_monthly_balance_KZT', 0)
        fx_volume = client.get('fx_volume', 0)
        total_spend = client.get('total_spend', 0)
        free_funds = client.get('free_funds', 0)
        age = client.get('age', 35)
        
        # ПРАВИЛО 1: Премиальная карта для VIP с ресторанами/косметикой
        if ('премиальный' in status.lower() or 'вип' in status.lower()) and spending_analysis['premium_share'] > 15:
            return 'Премиальная карта', f"VIP статус + {spending_analysis['premium_share']:.1f}% на рестораны/косметику"
        
        # ПРАВИЛО 2: Карта для путешествий только при доминировании такси/путешествий
        if spending_analysis['travel_share'] > 25:  # Больше 25% трат
            return 'Карта для путешествий', f"Высокие траты на такси/путешествия ({spending_analysis['travel_share']:.1f}%)"
        
        # ПРАВИЛО 3: Обмен валют только при реальных FX операциях
        if fx_volume > 100000:
            return 'Обмен валют', f"Активные валютные операции ({fx_volume:,.0f} ₸)"
        
        # ПРАВИЛО 4: Депозиты только при высоком балансе И свободных средствах
        if balance > 1000000 and free_funds > 100000:
            if fx_volume > 50000:
                return 'Депозит Мультивалютный', f"Высокий баланс + FX операции"
            elif spending_analysis['stability_score'] > 0.7:
                return 'Депозит Сберегательный', f"Высокий стабильный баланс"
            else:
                return 'Депозит Накопительный', f"Высокий баланс, активные траты"
        
        # ПРАВИЛО 5: Инвестиции для молодежи с высоким балансом
        if age < 35 and balance > 500000:
            return 'Инвестиции', f"Молодой возраст + достаточный баланс"
        
        # ПРАВИЛО 6: Золото для консерваторов с очень высоким балансом
        if age > 45 and balance > 2000000:
            return 'Золотые слитки', f"Зрелый возраст + очень высокий баланс"
        
        # ПРАВИЛО 7: Кредит наличными только при острой нужде
        outflows = client.get('outflows', 0)
        inflows = client.get('inflows', 1)
        if outflows > inflows * 2 and client.get('credit_payments', 0) > 0:
            return 'Кредит наличными', f"Острая нужда в средствах"
        
        # ПРАВИЛО 8: Премиальная карта для высоких трат на премиальные категории
        if spending_analysis['premium_share'] > 20 and balance > 300000:
            return 'Премиальная карта', f"Высокие траты на рестораны/косметику ({spending_analysis['premium_share']:.1f}%)"
        
        # ПРАВИЛО 9: Кредитная карта - универсальное решение для активных трат
        if total_spend > 1000000:  # Активные траты
            return 'Кредитная карта', f"Активные траты по разным категориям"
        
        # ПРАВИЛО 10: По умолчанию - кредитная карта
        return 'Кредитная карта', "Универсальное решение"
    
    def _analyze_spending_patterns(self, client: pd.Series) -> Dict:
        """
        Анализ паттернов трат клиента
        
        Args:
            client: Series с данными клиента
            
        Returns:
            Словарь с анализом трат
        """
        # Собираем траты по категориям
        category_spending = {}
        total_spend = 0
        
        for col in client.index:
            if col.startswith('spend_') and client[col] > 0:
                category = col.replace('spend_', '')
                # Исключаем технические категории
                if category not in ['Оплата картой', 'Переводы', 'Снятие наличных', 'Обмен валют']:
                    category_spending[category] = client[col]
                    total_spend += client[col]
        
        # Анализируем доли трат
        restaurant_spend = category_spending.get('Кафе и рестораны', 0)
        cosmetics_spend = category_spending.get('Косметика и парфюмерия', 0)
        taxi_spend = category_spending.get('Такси', 0)
        travel_spend = category_spending.get('Путешествия', 0)
        
        premium_relevant = restaurant_spend + cosmetics_spend
        travel_relevant = taxi_spend + travel_spend
        
        analysis = {
            'total_spend': total_spend,
            'premium_share': (premium_relevant / total_spend * 100) if total_spend > 0 else 0,
            'travel_share': (travel_relevant / total_spend * 100) if total_spend > 0 else 0,
            'top_categories': sorted(category_spending.items(), key=lambda x: x[1], reverse=True)[:3],
            'stability_score': client.get('stability_score', 0.5)
        }
        
        return analysis


if __name__ == "__main__":
    # Тестирование
    import sys
    sys.path.append('src')
    from etl import DataLoader
    from features import FeatureEngineering
    
    # Загружаем данные
    loader = DataLoader()
    clients_df, transactions_df = loader.load_all_data('data/clients.csv', 'data')
    
    fe = FeatureEngineering('conf/weights.yaml')
    features = fe.create_features(transactions_df)
    
    # Генерируем умные рекомендации
    recommender = SmartRecommender()
    smart_recommendations = recommender.recommend_for_clients(features)
    
    print("УМНЫЕ РЕКОМЕНДАЦИИ ДЛЯ ПЕРВЫХ 10 КЛИЕНТОВ:")
    print("="*60)
    
    names = ['Айгерим', 'Данияр', 'Сабина', 'Тимур', 'Камилла', 'Аян', 'Руслан', 'Мадина', 'Арман', 'Карина']
    
    for i in range(10):
        client_id = i + 1
        rec = smart_recommendations[smart_recommendations['client_code'] == client_id].iloc[0]
        print(f"{client_id}. {names[i]:<10} -> {rec['product']:<25} ({rec['reasoning']})")
    
    # Проверяем распределение
    print(f"\nРаспределение продуктов:")
    product_dist = smart_recommendations['product'].value_counts()
    for product, count in product_dist.items():
        print(f"  {product:<30} {count:>3}")
    
    print(f"\nВсего уникальных продуктов: {len(product_dist)}")
