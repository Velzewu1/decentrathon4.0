"""
Модуль для генерации push-уведомлений из шаблонов
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import random
import locale
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PushComposer:
    """Класс для генерации push-уведомлений"""
    
    def __init__(self, config_path: str = "conf/weights.yaml", 
                 templates_path: str = "conf/templates.yaml"):
        """
        Инициализация
        
        Args:
            config_path: путь к файлу конфигурации
            templates_path: путь к файлу с шаблонами
        """
        self.config = self._load_config(config_path)
        self.templates = self._load_config(templates_path)
        self.max_length = self.config['general']['push_max_length']
        
        # Настройка форматирования чисел
        try:
            locale.setlocale(locale.LC_ALL, 'ru_RU.UTF-8')
        except:
            try:
                locale.setlocale(locale.LC_ALL, 'Russian_Russia.1251')
            except:
                pass  # Используем стандартную локаль
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Файл не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _format_currency(self, amount: float) -> str:
        """
        Форматирование суммы в валюте
        
        Args:
            amount: сумма
            
        Returns:
            Отформатированная строка
        """
        # Форматируем число с пробелами между разрядами
        formatted = f"{amount:,.0f}".replace(",", " ")
        return formatted
    
    def _get_month_name(self, month_num: int) -> str:
        """
        Получение названия месяца в предложном падеже
        
        Args:
            month_num: номер месяца (1-12)
            
        Returns:
            Название месяца
        """
        return self.templates['months'].get(month_num, 'этом месяце')
    
    def _get_period_name(self, months: int) -> str:
        """
        Получение названия периода
        
        Args:
            months: количество месяцев
            
        Returns:
            Название периода
        """
        return self.templates['periods'].get(months, f'{months} месяцев')
    
    def _select_template(self, product: str, features: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """
        Выбор подходящего шаблона для продукта
        
        Args:
            product: название продукта
            features: признаки клиента
            
        Returns:
            Кортеж (шаблон, требуемые поля)
        """
        # Маппинг продуктов на ключи в шаблонах
        product_mapping = {
            'Карта для путешествий': 'travel_card',
            'Премиальная карта': 'premium_card',
            'Кредитная карта': 'credit_card',
            'Депозит': 'deposits',
            'Мультивалютный депозит': 'deposits',
            'Сберегательный депозит': 'deposits',
            'Накопительный депозит': 'deposits',
            'Валютный счет': 'fx',
            'Кредит наличными': 'cash_loan',
            'Инвестиции': 'investments',
            'Золото': 'gold'
        }
        
        template_key = product_mapping.get(product)
        if not template_key:
            logger.warning(f"Шаблон не найден для продукта: {product}")
            return None, None
        
        product_templates = self.templates.get(template_key, {})
        
        # Выбираем подходящий шаблон на основе доступных данных
        if template_key == 'travel_card':
            if features.get('spend_Такси', 0) > 0:
                return product_templates.get('taxi_focus'), 'taxi_focus'
            elif features.get('spend_Отели', 0) > 0:
                return product_templates.get('hotel_focus'), 'hotel_focus'
            else:
                return product_templates.get('travel_focus'), 'travel_focus'
        
        elif template_key == 'premium_card':
            if features.get('spend_Рестораны', 0) > 10000:
                return product_templates.get('restaurant_lover'), 'restaurant_lover'
            elif features.get('avg_monthly_balance', 0) > 300000:
                return product_templates.get('high_balance'), 'high_balance'
            else:
                return product_templates.get('luxury_spender'), 'luxury_spender'
        
        elif template_key == 'credit_card':
            if features.get('top_category_1', ''):
                return product_templates.get('top_categories'), 'top_categories'
            elif features.get('spend_Онлайн-сервисы', 0) > 0:
                return product_templates.get('online_services'), 'online_services'
            else:
                return product_templates.get('installments'), 'installments'
        
        elif template_key == 'deposits':
            deposit_type = features.get('deposit_type', 'Сберегательный депозит')
            if 'Мультивалютный' in deposit_type:
                return product_templates.get('multicurrency'), 'multicurrency'
            elif 'Накопительный' in deposit_type:
                return product_templates.get('accumulative'), 'accumulative'
            else:
                return product_templates.get('savings'), 'savings'
        
        elif template_key == 'fx':
            if features.get('fx_volume', 0) > 100000:
                return product_templates.get('active_trader'), 'active_trader'
            else:
                return product_templates.get('currency_saver'), 'currency_saver'
        
        elif template_key == 'cash_loan':
            if features.get('outflow_inflow_ratio', 0) > 2:
                return product_templates.get('urgent_needs'), 'urgent_needs'
            else:
                return product_templates.get('quick_approval'), 'quick_approval'
        
        elif template_key == 'investments':
            if features.get('free_funds', 0) > 500000:
                return product_templates.get('diversify'), 'diversify'
            else:
                return product_templates.get('start_investing'), 'start_investing'
        
        elif template_key == 'gold':
            if features.get('avg_monthly_balance', 0) > 1000000:
                return product_templates.get('luxury_protection'), 'luxury_protection'
            else:
                return product_templates.get('stability'), 'stability'
        
        # Возвращаем первый доступный шаблон
        if product_templates:
            first_key = list(product_templates.keys())[0]
            return product_templates[first_key], first_key
        
        return None, None
    
    def _prepare_template_data(self, features: pd.Series, details: pd.Series, 
                               product: str, benefit: float) -> Dict[str, Any]:
        """
        Подготовка данных для заполнения шаблона
        
        Args:
            features: признаки клиента
            details: детали продукта
            product: название продукта
            benefit: benefit score
            
        Returns:
            Словарь с данными для шаблона
        """
        # Функция для создания форматированной строки с пробелами
        def format_with_spaces(value):
            """Создаем класс-обертку для числа с кастомным форматированием"""
            class FormattedNumber:
                def __init__(self, val):
                    self.val = val
                    
                def __format__(self, format_spec):
                    # Форматируем число
                    formatted = format(self.val, format_spec.replace(',', ''))
                    # Заменяем точку на пробел для разделителя тысяч
                    if '.' not in formatted:
                        # Добавляем пробелы для разделения тысяч
                        parts = []
                        for i, char in enumerate(reversed(formatted)):
                            if i > 0 and i % 3 == 0:
                                parts.append(' ')
                            parts.append(char)
                        return ''.join(reversed(parts))
                    return formatted
            
            return FormattedNumber(value)
        
        # Базовые данные
        data = {
            'name': features.get('name', 'Клиент'),
            'benefit': format_with_spaces(benefit),
            'month': self._get_month_name(datetime.now().month - 1),  # предыдущий месяц
            'period': self._get_period_name(3),
        }
        
        # Добавляем специфичные данные для каждого продукта
        if 'Карта для путешествий' in product:
            data.update({
                'n_taxi': int(details.get('travel_n_taxi', 0)),
                'sum_taxi': format_with_spaces(details.get('travel_sum_taxi', 0)),
                'travel_spend': format_with_spaces(details.get('travel_total_spend', 0)),
                'hotel_spend': format_with_spaces(features.get('spend_Отели', 0)),
            })
        
        elif 'Премиальная' in product:
            data.update({
                'percent': details.get('premium_percent', 2),
                'restaurant_spend': format_with_spaces(details.get('premium_restaurant_spend', 0)),
            })
        
        elif 'Кредитная' in product:
            data.update({
                'cat1': details.get('credit_cat1', 'Покупки'),
                'cat2': details.get('credit_cat2', 'Продукты'),
                'cat3': details.get('credit_cat3', 'Транспорт'),
            })
        
        elif 'Депозит' in product or 'депозит' in product.lower():
            data.update({
                'rate': details.get('deposit_rate', 15),
            })
        
        elif 'Валютный' in product:
            data.update({
                'fx_volume': format_with_spaces(details.get('fx_volume', 0)),
            })
        
        elif 'Кредит' in product:
            data.update({
                'amount': format_with_spaces(details.get('loan_amount', 100000)),
            })
        
        elif 'Инвестиции' in product:
            data.update({
                'min_amount': format_with_spaces(details.get('investment_min_amount', 100000)),
                'rate': details.get('investment_rate', 15),
            })
        
        elif 'Золото' in product:
            data.update({
                'min_amount': format_with_spaces(details.get('gold_min_amount', 50000)),
            })
        
        return data
    
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
        template_info, template_key = self._select_template(product, features)
        
        if not template_info:
            # Fallback шаблон
            name = features.get('name', 'Клиент')
            if name == '':
                name = 'Клиент'
            return f"{name}, откройте {product} и получите выгоду до {self._format_currency(benefit)} ₸. Оформить сейчас."
        
        # Подготавливаем данные
        template_data = self._prepare_template_data(features, details, product, benefit)
        
        # Заполняем шаблон
        try:
            push_text = template_info['template'].format(**template_data)
        except (KeyError, ValueError) as e:
            logger.warning(f"Ошибка форматирования шаблона {template_key}: {e}")
            # Используем fallback
            name = features.get('name', 'Клиент')
            if name == '':
                name = 'Клиент'
            return f"{name}, откройте {product} и получите выгоду до {self._format_currency(benefit)} ₸. Оформить сейчас."
        
        # Проверяем длину
        if len(push_text) > self.max_length:
            logger.warning(f"Push превышает максимальную длину: {len(push_text)} > {self.max_length}")
            # Пробуем укоротить
            push_text = self._shorten_push(push_text)
        
        # Валидация
        push_text = self._validate_push(push_text)
        
        return push_text
    
    def _shorten_push(self, push_text: str) -> str:
        """
        Укорачивание push-уведомления
        
        Args:
            push_text: исходный текст
            
        Returns:
            Укороченный текст
        """
        # Удаляем лишние пробелы
        push_text = ' '.join(push_text.split())
        
        # Если все еще длинный, обрезаем и добавляем CTA
        if len(push_text) > self.max_length:
            push_text = push_text[:self.max_length-20] + '... Открыть сейчас.'
        
        return push_text
    
    def _validate_push(self, push_text: str) -> str:
        """
        Валидация push-уведомления согласно правилам
        
        Args:
            push_text: текст уведомления
            
        Returns:
            Валидированный текст
        """
        # Проверка на CAPS LOCK
        if push_text.isupper():
            push_text = push_text.capitalize()
        
        # Проверка на множественные восклицательные знаки
        while '!!' in push_text:
            push_text = push_text.replace('!!', '!')
        
        # Проверка на количество восклицательных знаков
        if push_text.count('!') > 1:
            # Оставляем только первый
            parts = push_text.split('!')
            push_text = parts[0] + '!' + ''.join(parts[1:]).replace('!', '.')
        
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
            
            # Если это депозит, используем конкретный тип
            if product == 'Депозит' and 'deposit_type' in row:
                deposit_type = row['deposit_type']
                if deposit_type:
                    # Добавляем тип депозита в features для правильного выбора шаблона
                    client_features = features[features['client_code'] == client_code].iloc[0].copy()
                    client_features['deposit_type'] = deposit_type
                else:
                    client_features = features[features['client_code'] == client_code].iloc[0]
            else:
                client_features = features[features['client_code'] == client_code].iloc[0]
            
            client_details = details[details['client_code'] == client_code].iloc[0]
            
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


def main():
    """Тестовый запуск генерации push-уведомлений"""
    
    # Создаем тестовые данные
    test_features = pd.DataFrame({
        'client_code': [1, 2, 3],
        'name': ['Рамазан', 'Алия', 'Петр'],
        'status': ['Премиальный клиент', 'Студент', 'Зарплатный клиент'],
        'avg_monthly_balance': [500000, 50000, 200000],
        'spend_Такси': [27400, 3500, 8000],
        'spend_Рестораны': [35000, 0, 12000],
        'spend_Отели': [80000, 0, 60000],
        'spend_Путешествия': [150000, 0, 100000],
        'spend_Онлайн-сервисы': [3000, 5000, 0],
        'fx_volume': [150000, 31200, 232000],
        'outflow_inflow_ratio': [float('inf'), 2.57, float('inf')],
        'top_category_1': ['Путешествия', 'Продукты', 'Путешествия'],
        'top_category_2': ['Отели', 'Образование', 'Отели'],
        'top_category_3': ['Рестораны', 'Онлайн-сервисы', 'Продукты'],
        'deposit_type': ['Мультивалютный депозит', 'Сберегательный депозит', 'Мультивалютный депозит']
    })
    
    test_details = pd.DataFrame({
        'client_code': [1, 2, 3],
        'travel_n_taxi': [12, 2, 4],
        'travel_sum_taxi': [27400, 3500, 8000],
        'travel_total_spend': [266400, 3500, 168000],
        'premium_percent': [4, 2, 3],
        'premium_restaurant_spend': [35000, 0, 12000],
        'credit_cat1': ['Путешествия', 'Продукты', 'Путешествия'],
        'credit_cat2': ['Отели', 'Образование', 'Отели'],
        'credit_cat3': ['Рестораны', 'Онлайн-сервисы', 'Продукты'],
        'deposit_rate': [14.5, 16.5, 14.5],
        'fx_volume': [150000, 31200, 232000],
        'loan_amount': [362000, 23500, 195000],
        'investment_min_amount': [100000, 100000, 100000],
        'investment_rate': [15, 15, 15],
        'gold_min_amount': [50000, 50000, 50000]
    })
    
    test_ranked = pd.DataFrame({
        'client_code': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        'rank': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
        'product': ['Депозит', 'Премиальная карта', 'Карта для путешествий', 'Инвестиции',
                   'Депозит', 'Кредит наличными', 'Премиальная карта', 'Валютный счет',
                   'Депозит', 'Премиальная карта', 'Карта для путешествий', 'Валютный счет'],
        'benefit': [94250, 18640, 9560, 5000, 8550, 2350, 1155, 312, 36000, 7800, 6720, 2320],
        'deposit_type': ['Мультивалютный депозит', None, None, None,
                         'Сберегательный депозит', None, None, None,
                         'Мультивалютный депозит', None, None, None]
    })
    
    # Генерируем push-уведомления
    composer = PushComposer()
    pushes_df = composer.generate_all_pushes(test_features, test_details, test_ranked)
    
    print("\nСгенерированные push-уведомления:")
    for idx, row in pushes_df.iterrows():
        print(f"\nКлиент {row['client_code']} - {row['product']}:")
        print(f"  {row['push_notification']}")
        print(f"  Длина: {len(row['push_notification'])} символов")


if __name__ == "__main__":
    main()
