"""
Модуль для генерации push-уведомлений из шаблонов (версия 2.0)
Согласно требованиям хакатона
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PushComposerV2:
    """Класс для генерации push-уведомлений согласно TOV"""
    
    def __init__(self, config_path: str = "conf/weights_v2.yaml", 
                 templates_path: str = "conf/templates_v2.yaml"):
        """
        Инициализация
        
        Args:
            config_path: путь к файлу конфигурации
            templates_path: путь к файлу с шаблонами
        """
        self.config = self._load_config(config_path)
        self.templates = self._load_config(templates_path)
        self.max_length = 220  # Максимальная длина push
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Файл не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _format_currency(self, amount: float) -> str:
        """
        Форматирование суммы в валюте с пробелами
        
        Args:
            amount: сумма
            
        Returns:
            Отформатированная строка (2 490 ₸)
        """
        # Форматируем число с пробелами между разрядами
        formatted = f"{int(amount):,}".replace(",", " ")
        return formatted
    
    def _get_month_name(self, month_num: int) -> str:
        """
        Получение названия месяца в предложном падеже
        
        Args:
            month_num: номер месяца (1-12)
            
        Returns:
            Название месяца
        """
        months = self.templates.get('months', {})
        return months.get(month_num, 'этом месяце')
    
    def _select_template(self, product: str, features: pd.Series, details: pd.Series) -> Dict[str, Any]:
        """
        Выбор подходящего шаблона для продукта
        
        Args:
            product: название продукта
            features: признаки клиента
            details: детали для генерации
            
        Returns:
            Словарь с шаблоном и полями
        """
        # Маппинг продуктов на ключи в шаблонах
        if 'путешествий' in product.lower():
            templates = self.templates.get('travel_card', {})
            # Выбираем шаблон в зависимости от данных
            if details.get('taxi_count', 0) > 5:
                return templates.get('taxi_lover', {})
            elif features.get('spend_Отели', 0) > 0:
                return templates.get('hotel_user', {})
            else:
                return templates.get('frequent_traveler', {})
        
        elif 'премиальная' in product.lower():
            templates = self.templates.get('premium_card', {})
            if features.get('avg_monthly_balance', 0) > 1000000:
                return templates.get('high_balance', {})
            elif details.get('restaurant_amount', 0) > 10000:
                return templates.get('restaurant_spender', {})
            else:
                return templates.get('luxury_buyer', {})
        
        elif 'кредитная' in product.lower():
            templates = self.templates.get('credit_card', {})
            if features.get('top_category_1'):
                return templates.get('top_categories', {})
            else:
                return templates.get('online_active', {})
        
        elif 'обмен валют' in product.lower() or 'валют' in product.lower():
            templates = self.templates.get('fx', {})
            if details.get('fx_volume', 0) > 50000:
                return templates.get('active_trader', {})
            else:
                return templates.get('currency_user', {})
        
        elif 'кредит' in product.lower() and 'наличными' in product.lower():
            templates = self.templates.get('cash_loan', {})
            return templates.get('needs_funds', {})
        
        elif 'депозит' in product.lower() or 'вклад' in product.lower():
            templates = self.templates.get('deposits', {})
            deposit_type = features.get('best_deposit_type', '')
            if 'мультивалют' in deposit_type.lower():
                return templates.get('multicurrency', {})
            elif 'накопитель' in deposit_type.lower():
                return templates.get('accumulative', {})
            else:
                return templates.get('savings', {})
        
        elif 'инвестиц' in product.lower():
            templates = self.templates.get('investments', {})
            if features.get('free_funds', 0) > 100000:
                return templates.get('has_free_funds', {})
            else:
                return templates.get('start_investing', {})
        
        elif 'золот' in product.lower():
            templates = self.templates.get('gold', {})
            return templates.get('diversification', {})
        
        # Fallback
        return {}
    
    def compose_push(self, features: pd.Series, details: pd.Series, 
                    product: str, benefit: float) -> str:
        """
        Генерация push-уведомления для клиента
        
        Args:
            features: признаки клиента
            details: детали продукта
            product: название продукта
            benefit: benefit score
            
        Returns:
            Текст push-уведомления
        """
        # Выбираем шаблон
        template_info = self._select_template(product, features, details)
        
        if not template_info or 'template' not in template_info:
            # Fallback шаблон
            name = features.get('name', 'Клиент')
            return f"{name}, откройте {product} и получите выгоду. Узнать подробнее."
        
        # Подготавливаем данные для шаблона
        template_data = {
            'name': features.get('name', 'Клиент'),
            'benefit': benefit,
            'month': self._get_month_name(datetime.now().month - 1),
        }
        
        # Добавляем специфичные данные
        if 'taxi' in template_info.get('template', ''):
            template_data['n_taxi'] = int(details.get('taxi_count', 0))
            template_data['taxi_amount'] = details.get('taxi_amount', 0)
        
        if 'travel_amount' in template_info.get('required_fields', []):
            template_data['travel_amount'] = details.get('travel_amount', 0)
        
        if 'percent' in template_info.get('required_fields', []):
            template_data['percent'] = details.get('premium_percent', 2)
        
        if 'restaurant_amount' in template_info.get('required_fields', []):
            template_data['restaurant_amount'] = details.get('restaurant_amount', 0)
        
        if 'cat1' in template_info.get('required_fields', []):
            template_data['cat1'] = details.get('top_cat1', 'Покупки')
            template_data['cat2'] = details.get('top_cat2', 'Продукты')
            template_data['cat3'] = details.get('top_cat3', 'Транспорт')
        
        if 'fx_volume' in template_info.get('required_fields', []):
            template_data['fx_volume'] = details.get('fx_volume', 0)
        
        if 'currency' in template_info.get('required_fields', []):
            template_data['currency'] = details.get('main_currency', 'долларах')
        
        if 'amount' in template_info.get('required_fields', []):
            if 'loan' in product.lower():
                template_data['amount'] = min(details.get('loan_amount', 100000), 1000000)
            else:
                template_data['amount'] = details.get('free_funds_amount', 0)
        
        # Заполняем шаблон
        try:
            push_text = template_info['template'].format(**template_data)
        except (KeyError, ValueError) as e:
            logger.warning(f"Ошибка при заполнении шаблона: {e}")
            name = features.get('name', 'Клиент')
            return f"{name}, откройте {product} и получите выгоду. Узнать подробнее."
        
        # Проверяем длину
        if len(push_text) > self.max_length:
            # Укорачиваем
            push_text = push_text[:self.max_length-20] + '... Узнать подробнее.'
        
        return push_text
    
    def generate_all_pushes(self, features: pd.DataFrame, details: pd.DataFrame,
                           ranked_products: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация push-уведомлений для всех клиентов
        
        Args:
            features: DataFrame с признаками клиентов
            details: DataFrame с деталями продуктов
            ranked_products: DataFrame с ранжированными продуктами
            
        Returns:
            DataFrame с push-уведомлениями
        """
        logger.info("Генерация push-уведомлений для всех клиентов")
        
        result_list = []
        
        # Получаем основной продукт для каждого клиента (rank=1)
        main_products = ranked_products[ranked_products['rank'] == 1].copy()
        
        for idx, row in main_products.iterrows():
            client_code = row['client_code']
            product = row['product']
            benefit = row['benefit']
            
            # Получаем данные клиента
            client_features = features[features['client_code'] == client_code].iloc[0]
            client_details = details[details['client_code'] == client_code].iloc[0]
            
            # Если это депозит, добавляем тип
            if product == 'Депозит' and 'deposit_type' in row:
                client_features = client_features.copy()
                client_features['best_deposit_type'] = row['deposit_type']
            
            # Генерируем push
            push_text = self.compose_push(client_features, client_details, product, benefit)
            
            result_list.append({
                'client_code': client_code,
                'product': product,
                'push_notification': push_text
            })
        
        result_df = pd.DataFrame(result_list)
        
        logger.info(f"Сгенерировано {len(result_df)} push-уведомлений")
        
        return result_df
    
    
class FormattedNumber:
    """Класс для форматирования чисел в шаблонах"""
    def __init__(self, value):
        self.value = value
    
    def __format__(self, format_spec):
        # Форматируем число с пробелами для разделения тысяч
        if format_spec.endswith('f'):
            # Убираем спецификатор f
            precision = format_spec[:-1]
            if precision.startswith('.'):
                precision = int(precision[1:])
            else:
                precision = 0
            
            if precision == 0:
                formatted = f"{int(self.value):,}".replace(",", " ")
            else:
                formatted = f"{self.value:,.{precision}f}".replace(",", " ")
        else:
            formatted = f"{int(self.value):,}".replace(",", " ")
        
        return formatted


def main():
    """Тестовый запуск"""
    # Создаем тестовые данные
    test_features = pd.DataFrame({
        'client_code': [1],
        'name': ['Айгерим'],
        'avg_monthly_balance': [92643],
        'spend_Такси': [27400],
        'spend_Кафе и рестораны': [15000],
        'top_category_1': ['Такси'],
        'fx_volume': [50000],
        'free_funds': [30000]
    })
    
    test_details = pd.DataFrame({
        'client_code': [1],
        'taxi_count': [12],
        'taxi_amount': [27400],
        'restaurant_amount': [5000],
        'premium_percent': [2],
        'top_cat1': ['Такси'],
        'top_cat2': ['Продукты'],
        'top_cat3': ['Кафе']
    })
    
    test_ranked = pd.DataFrame({
        'client_code': [1],
        'rank': [1],
        'product': ['Карта для путешествий'],
        'benefit': [1100]
    })
    
    # Генерируем push
    composer = PushComposerV2()
    result = composer.generate_all_pushes(test_features, test_details, test_ranked)
    
    print("\nСгенерированный push:")
    print(result['push_notification'].iloc[0])


if __name__ == "__main__":
    main()
