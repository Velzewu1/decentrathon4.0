"""
Модуль для ранжирования и выбора Top-4 продуктов
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


class ProductRanking:
    """Класс для ранжирования и выбора топ продуктов"""
    
    def __init__(self, config_path: str = "conf/weights.yaml"):
        """
        Инициализация
        
        Args:
            config_path: путь к файлу конфигурации
        """
        self.config = self._load_config(config_path)
        self.product_priority = self.config['product_priority']
        self.top_n = self.config['general']['top_products_count']
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _get_priority_score(self, product_name: str) -> int:
        """
        Получение приоритета продукта (чем меньше число, тем выше приоритет)
        
        Args:
            product_name: название продукта
            
        Returns:
            Приоритет продукта
        """
        # Ищем продукт в списке приоритетов
        for priority, name in self.product_priority.items():
            if name == product_name:
                return int(priority)
        
        # Если продукт не найден, возвращаем низкий приоритет
        return 999
    
    def rank_products(self, benefits: pd.DataFrame) -> pd.DataFrame:
        """
        Ранжирование продуктов для каждого клиента
        
        Args:
            benefits: DataFrame с benefit scores
            
        Returns:
            DataFrame с ранжированными продуктами
        """
        logger.info("Ранжирование продуктов для клиентов")
        
        result_list = []
        
        # Обрабатываем каждого клиента
        for idx, row in benefits.iterrows():
            client_code = row['client_code']
            
            # Собираем все продукты с их benefit scores
            products = []
            
            # Проходим по всем колонкам с benefit
            for col in benefits.columns:
                if col.startswith('benefit_') and col != 'benefit_Депозит':
                    product_name = col.replace('benefit_', '')
                    benefit_value = row[col]
                    
                    # Пропускаем продукты с отрицательным или нулевым benefit
                    if benefit_value > 0 and not np.isinf(benefit_value):
                        products.append({
                            'product': product_name,
                            'benefit': benefit_value,
                            'priority': self._get_priority_score(product_name)
                        })
            
            # Добавляем депозит (уже выбран лучший)
            if row.get('benefit_Депозит', 0) > 0:
                best_deposit = row.get('best_deposit_type', 'Депозит')
                products.append({
                    'product': 'Депозит',
                    'benefit': row['benefit_Депозит'],
                    'priority': self._get_priority_score('Депозит'),
                    'deposit_type': best_deposit
                })
            
            # Сортируем продукты: сначала по benefit (убывание), потом по приоритету (возрастание)
            products_sorted = sorted(
                products,
                key=lambda x: (-x['benefit'], x['priority'])
            )
            
            # Выбираем топ-N продуктов
            top_products = products_sorted[:self.top_n]
            
            # Добавляем в результат
            for rank, product_info in enumerate(top_products, 1):
                result_list.append({
                    'client_code': client_code,
                    'rank': rank,
                    'product': product_info['product'],
                    'benefit': product_info['benefit'],
                    'deposit_type': product_info.get('deposit_type', None)
                })
        
        result_df = pd.DataFrame(result_list)
        
        logger.info(f"Ранжирование завершено для {len(benefits)} клиентов")
        
        return result_df
    
    def get_main_product(self, ranked_products: pd.DataFrame) -> pd.DataFrame:
        """
        Получение основного (первого) продукта для каждого клиента
        
        Args:
            ranked_products: DataFrame с ранжированными продуктами
            
        Returns:
            DataFrame с основным продуктом для каждого клиента
        """
        logger.info("Определение основного продукта для каждого клиента")
        
        # Выбираем продукт с рангом 1
        main_products = ranked_products[ranked_products['rank'] == 1].copy()
        
        # Переименовываем колонки
        main_products = main_products.rename(columns={
            'product': 'main_product',
            'benefit': 'main_benefit'
        })
        
        # Удаляем колонку rank
        main_products = main_products.drop(columns=['rank'])
        
        return main_products
    
    def get_recommendations_summary(self, ranked_products: pd.DataFrame, 
                                   features: pd.DataFrame) -> pd.DataFrame:
        """
        Создание сводной таблицы рекомендаций
        
        Args:
            ranked_products: DataFrame с ранжированными продуктами
            features: DataFrame с признаками клиентов
            
        Returns:
            DataFrame со сводкой рекомендаций
        """
        logger.info("Создание сводной таблицы рекомендаций")
        
        # Получаем основной продукт
        main_products = self.get_main_product(ranked_products)
        
        # Добавляем информацию о клиенте
        summary = main_products.merge(
            features[['client_code', 'name', 'status', 'city']],
            on='client_code',
            how='left'
        )
        
        # Создаем список всех рекомендованных продуктов для каждого клиента
        all_products = ranked_products.groupby('client_code').apply(
            lambda x: ', '.join(x.sort_values('rank')['product'].tolist())
        ).reset_index()
        all_products.columns = ['client_code', 'all_products']
        
        # Добавляем к сводке
        summary = summary.merge(all_products, on='client_code', how='left')
        
        # Добавляем суммарный benefit от топ-4 продуктов
        total_benefit = ranked_products.groupby('client_code')['benefit'].sum().reset_index()
        total_benefit.columns = ['client_code', 'total_benefit']
        
        summary = summary.merge(total_benefit, on='client_code', how='left')
        
        return summary
    
    def analyze_product_distribution(self, ranked_products: pd.DataFrame) -> pd.DataFrame:
        """
        Анализ распределения продуктов
        
        Args:
            ranked_products: DataFrame с ранжированными продуктами
            
        Returns:
            DataFrame со статистикой по продуктам
        """
        logger.info("Анализ распределения продуктов")
        
        # Считаем, сколько раз каждый продукт попал в топ
        product_stats = ranked_products.groupby('product').agg({
            'client_code': 'count',
            'benefit': ['mean', 'median', 'std'],
            'rank': 'mean'
        }).reset_index()
        
        # Переименовываем колонки
        product_stats.columns = [
            'product', 'count', 'benefit_mean', 
            'benefit_median', 'benefit_std', 'avg_rank'
        ]
        
        # Сортируем по количеству рекомендаций
        product_stats = product_stats.sort_values('count', ascending=False)
        
        # Добавляем процент клиентов
        total_clients = ranked_products['client_code'].nunique()
        product_stats['client_percentage'] = (product_stats['count'] / total_clients * 100).round(1)
        
        return product_stats


def main():
    """Тестовый запуск ranking"""
    from scoring import BenefitScoring
    
    # Создаем тестовые данные с benefit scores
    test_benefits = pd.DataFrame({
        'client_code': [1, 2, 3],
        'benefit_Карта для путешествий': [9560, 140, 6720],
        'benefit_Премиальная карта': [18640, 1155, 7800],
        'benefit_Кредитная карта': [0, 0, 0],
        'benefit_Валютный счет': [1500, 312, 2320],
        'benefit_Кредит наличными': [float('-inf'), 2350, float('-inf')],
        'benefit_Депозит': [94250, 8550, 36000],
        'best_deposit_type': ['Мультивалютный депозит', 'Сберегательный депозит', 'Мультивалютный депозит'],
        'benefit_Инвестиции': [5000, 0, 0],
        'benefit_Золото': [2500, 0, 0]
    })
    
    # Создаем тестовые признаки
    test_features = pd.DataFrame({
        'client_code': [1, 2, 3],
        'name': ['Иван', 'Мария', 'Петр'],
        'status': ['Премиальный клиент', 'Студент', 'Зарплатный клиент'],
        'city': ['Алматы', 'Астана', 'Шымкент']
    })
    
    # Ранжируем продукты
    ranking = ProductRanking()
    ranked_df = ranking.rank_products(test_benefits)
    
    print("\nРанжированные продукты:")
    print(ranked_df)
    
    # Получаем основные продукты
    main_products_df = ranking.get_main_product(ranked_df)
    print("\nОсновные продукты:")
    print(main_products_df)
    
    # Создаем сводку
    summary_df = ranking.get_recommendations_summary(ranked_df, test_features)
    print("\nСводка рекомендаций:")
    print(summary_df)
    
    # Анализируем распределение
    stats_df = ranking.analyze_product_distribution(ranked_df)
    print("\nСтатистика по продуктам:")
    print(stats_df)


if __name__ == "__main__":
    main()
