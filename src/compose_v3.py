"""
Улучшенный модуль композиции push-уведомлений (версия 3.0)
Исправлены баги с форматированием, расчетами и соответствием продуктов
"""
import pandas as pd
import numpy as np
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class FormattedNumber:
    """Класс для правильного форматирования чисел с запятой и пробелами"""
    
    def __init__(self, value: float):
        self.value = float(value) if pd.notna(value) else 0
    
    def __format__(self, format_spec: str) -> str:
        if format_spec == ',.0f':
            # Форматируем с пробелами как разделителями разрядов
            formatted = f"{self.value:,.0f}"
            # Заменяем запятые на пробелы для разрядов
            formatted = formatted.replace(',', ' ')
            return formatted
        elif format_spec == ',.1f':
            # Для процентов с одним знаком после запятой
            formatted = f"{self.value:.1f}"
            # Заменяем точку на запятую
            formatted = formatted.replace('.', ',')
            return formatted
        else:
            return str(self.value)


class PushComposerV3:
    """Улучшенный класс для генерации push-уведомлений"""
    
    def __init__(self, config_path: str = "conf/weights_v3.yaml", 
                 templates_path: str = "conf/templates_v3.yaml"):
        """
        Инициализация композера
        
        Args:
            config_path: Путь к конфигурации
            templates_path: Путь к шаблонам
        """
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        with open(templates_path, 'r', encoding='utf-8') as f:
            self.templates = yaml.safe_load(f)
    
    def compose_push_for_clients(self, 
                                recommendations_df: pd.DataFrame,
                                features_df: pd.DataFrame,
                                details_df: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация push-уведомлений для всех клиентов
        
        Args:
            recommendations_df: DataFrame с рекомендациями
            features_df: DataFrame с признаками
            details_df: DataFrame с деталями продуктов
            
        Returns:
            DataFrame с push-уведомлениями
        """
        logger.info("Генерация push-уведомлений для всех клиентов")
        
        results = []
        
        for _, rec in recommendations_df.iterrows():
            client_code = rec['client_code']
            product = rec['product']
            benefit = rec.get('benefit', 0)
            
            # Получаем признаки клиента
            client_features = features_df[features_df['client_code'] == client_code]
            if len(client_features) == 0:
                logger.warning(f"Не найдены признаки для клиента {client_code}")
                continue
            
            features = client_features.iloc[0]
            
            # Получаем детали продукта
            client_details = details_df[details_df['client_code'] == client_code] if len(details_df) > 0 else pd.DataFrame()
            details = client_details.iloc[0] if len(client_details) > 0 else pd.Series()
            
            # Генерируем push
            push = self.compose_push(product, features, details, benefit)
            
            if push:  # Только если push успешно сгенерирован
                results.append({
                    'client_code': client_code,
                    'product': product,
                    'push_notification': push
                })
            else:
                logger.warning(f"Не удалось сгенерировать push для клиента {client_code}, продукт {product}")
        
        logger.info(f"Сгенерировано {len(results)} push-уведомлений")
        return pd.DataFrame(results)
    
    def compose_push(self, 
                     product: str, 
                     features: pd.Series, 
                     details: pd.Series = None,
                     benefit: float = 0) -> Optional[str]:
        """
        Генерация push-уведомления для продукта
        
        Args:
            product: Название продукта
            features: Признаки клиента
            details: Детали продукта
            benefit: Benefit score
            
        Returns:
            Текст push-уведомления или None если не удалось сгенерировать
        """
        if details is None:
            details = pd.Series()
        
        try:
            # Нормализуем название продукта
            product_key = self._normalize_product_name(product)
            
            if product_key == 'travel_card':
                return self._compose_travel_card_push(features, details, benefit)
            elif product_key == 'premium_card':
                return self._compose_premium_card_push(features, details)
            elif product_key == 'credit_card':
                return self._compose_credit_card_push(features, details)
            elif product_key == 'fx':
                return self._compose_fx_push(features, details)
            elif product_key in ['deposit_savings', 'deposit_accumulative', 'deposit_multicurrency']:
                return self._compose_deposit_push(product_key, features, details)
            elif product_key == 'investments':
                return self._compose_investments_push(features, details)
            elif product_key == 'gold':
                return self._compose_gold_push(features, details)
            elif product_key == 'cash_loan':
                return self._compose_cash_loan_push(features, details)
            else:
                logger.warning(f"Неизвестный продукт: {product}")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при генерации push для продукта {product}: {e}")
            return None
    
    def _normalize_product_name(self, product: str) -> str:
        """Нормализация названия продукта к ключу в шаблонах"""
        product_mapping = {
            'Карта для путешествий': 'travel_card',
            'Премиальная карта': 'premium_card',
            'Кредитная карта': 'credit_card',
            'Обмен валют': 'fx',
            'Депозит Сберегательный': 'deposit_savings',
            'Депозит Накопительный': 'deposit_accumulative', 
            'Депозит Мультивалютный': 'deposit_multicurrency',
            'Депозит': 'deposit_savings',  # По умолчанию сберегательный
            'Инвестиции': 'investments',
            'Золотые слитки': 'gold',
            'Кредит наличными': 'cash_loan'
        }
        return product_mapping.get(product, product.lower().replace(' ', '_'))
    
    def _compose_travel_card_push(self, features: pd.Series, details: pd.Series, benefit: float) -> Optional[str]:
        """Генерация push для карты путешествий"""
        name = features.get('name', 'Клиент')
        
        # Получаем данные о тратах на такси и путешествия
        taxi_spend = features.get('spend_Такси', 0)
        travel_spend = features.get('spend_Путешествия', 0)
        
        total_spend = taxi_spend + travel_spend
        
        # Проверяем минимальный порог трат
        min_threshold = self.config.get('travel_card', {}).get('min_spend_threshold', 1000)
        if total_spend < min_threshold:
            logger.debug(f"Траты на такси/путешествия слишком малы: {total_spend}")
            return None
        
        # Рассчитываем реальный кешбэк
        cashback_rate = self.config.get('travel_card', {}).get('cashback_rate', 0.04)
        real_cashback = total_spend * cashback_rate
        
        # Проверяем что кешбэк реалистичен
        max_cashback = total_spend * 0.04  # Не больше 4% от трат
        if real_cashback > max_cashback or real_cashback <= 0:
            logger.debug(f"Нереалистичный кешбэк: {real_cashback} при тратах {total_spend}")
            return None
        
        # Определяем количество поездок (примерно)
        avg_taxi_trip = 1500  # Средняя поездка на такси
        taxi_count = max(1, int(taxi_spend / avg_taxi_trip)) if taxi_spend > 0 else 0
        
        template = self.templates['travel_card']['template']
        
        try:
            push = template.format(
                name=name,
                month="августе",
                taxi_count=taxi_count,
                taxi_amount=self._format_amount(total_spend),
                cashback_amount=self._format_amount(real_cashback)
            )
            return self._validate_and_fix_push(push)
        except Exception as e:
            logger.error(f"Ошибка форматирования push для карты путешествий: {e}")
            return None
    
    def _compose_premium_card_push(self, features: pd.Series, details: pd.Series) -> str:
        """Генерация push для премиальной карты"""
        name = features.get('name', 'Клиент')
        restaurant_spend = features.get('spend_Кафе и рестораны', 0)
        balance = features.get('avg_monthly_balance_KZT', 0)
        
        # Выбираем подходящий шаблон
        if restaurant_spend > 50000:  # Часто ходит в рестораны
            template = self.templates['premium_card']['restaurant_lover']['template']
            monthly_restaurant = restaurant_spend / 3  # За месяц
            
            push = template.format(
                name=name,
                restaurant_amount=self._format_amount(monthly_restaurant)
            )
        else:  # Высокий баланс
            template = self.templates['premium_card']['high_balance']['template']
            push = template.format(name=name)
        
        return self._validate_and_fix_push(push)
    
    def _compose_credit_card_push(self, features: pd.Series, details: pd.Series) -> str:
        """Генерация push для кредитной карты"""
        name = features.get('name', 'Клиент')
        
        # Получаем топ-3 категории
        top_cats = []
        for i in range(1, 4):
            cat = features.get(f'top_category_{i}', '')
            if cat and cat not in ['Оплата картой', 'Переводы', 'Снятие наличных']:
                top_cats.append(cat)
        
        # Дополняем до 3 категорий если нужно
        default_cats = ['Продукты питания', 'Кафе и рестораны', 'Такси']
        while len(top_cats) < 3:
            for cat in default_cats:
                if cat not in top_cats:
                    top_cats.append(cat)
                    break
            if len(top_cats) >= 3:
                break
        
        template = self.templates['credit_card']['template']
        push = template.format(
            name=name,
            top_category_1=top_cats[0] if len(top_cats) > 0 else 'Продукты питания',
            top_category_2=top_cats[1] if len(top_cats) > 1 else 'Кафе и рестораны', 
            top_category_3=top_cats[2] if len(top_cats) > 2 else 'Такси'
        )
        
        return self._validate_and_fix_push(push)
    
    def _compose_fx_push(self, features: pd.Series, details: pd.Series) -> Optional[str]:
        """Генерация push для обмена валют"""
        name = features.get('name', 'Клиент')
        fx_volume = features.get('fx_volume', 0)
        
        # Проверяем минимальный объем
        min_volume = self.config.get('fx', {}).get('min_volume_threshold', 50000)
        if fx_volume < min_volume:
            logger.debug(f"FX объем слишком мал: {fx_volume}")
            return None
        
        template = self.templates['fx']['template']
        push = template.format(
            name=name,
            fx_volume=self._format_amount(fx_volume)
        )
        
        return self._validate_and_fix_push(push)
    
    def _compose_deposit_push(self, product_key: str, features: pd.Series, details: pd.Series) -> str:
        """Генерация push для депозитов"""
        name = features.get('name', 'Клиент')
        template = self.templates[product_key]['template']
        
        push = template.format(name=name)
        return self._validate_and_fix_push(push)
    
    def _compose_investments_push(self, features: pd.Series, details: pd.Series) -> str:
        """Генерация push для инвестиций"""
        name = features.get('name', 'Клиент')
        template = self.templates['investments']['template']
        
        push = template.format(name=name)
        return self._validate_and_fix_push(push)
    
    def _compose_gold_push(self, features: pd.Series, details: pd.Series) -> str:
        """Генерация push для золотых слитков"""
        name = features.get('name', 'Клиент')
        template = self.templates['gold']['template']
        
        push = template.format(name=name)
        return self._validate_and_fix_push(push)
    
    def _compose_cash_loan_push(self, features: pd.Series, details: pd.Series) -> str:
        """Генерация push для кредита наличными"""
        name = features.get('name', 'Клиент')
        template = self.templates['cash_loan']['template']
        
        # Определяем лимит кредита на основе доходов
        inflows = features.get('inflows', 0)
        loan_limit = min(inflows * 2, 1000000)  # До 2х доходов, но не больше 1 млн
        
        push = template.format(
            name=name,
            loan_limit=self._format_amount(loan_limit)
        )
        
        return self._validate_and_fix_push(push)
    
    def _format_amount(self, amount: float) -> str:
        """Форматирование суммы с правильными разделителями"""
        if pd.isna(amount) or amount == 0:
            return "0 ₸"
        
        formatted_num = FormattedNumber(amount)
        return f"{formatted_num:,.0f} ₸"
    
    def _format_percentage(self, value: float) -> str:
        """Форматирование процентов с запятой"""
        formatted_num = FormattedNumber(value)
        return f"{formatted_num:,.1f}%"
    
    def _validate_and_fix_push(self, push: str) -> str:
        """Валидация и исправление push-уведомления"""
        if not push:
            return ""
        
        # Исправляем название карты на консистентное
        push = push.replace('Премиум-карта', 'Премиальная карта')
        push = push.replace('премиум-карта', 'премиальная карта')
        
        # Исправляем проценты с точки на запятую
        push = re.sub(r'(\d+)\.(\d+)%', r'\1,\2%', push)
        
        # Убираем множественные пробелы
        push = re.sub(r'\s+', ' ', push)
        
        # Обрезаем до максимальной длины
        max_length = 220
        if len(push) > max_length:
            push = push[:max_length-3] + '...'
        
        return push.strip()


if __name__ == "__main__":
    # Тестирование
    composer = PushComposerV3()
    
    # Тестовые данные
    test_features = pd.Series({
        'name': 'Тестовый Клиент',
        'spend_Такси': 50000,
        'spend_Путешествия': 30000,
        'spend_Кафе и рестораны': 75000,
        'top_category_1': 'Продукты питания',
        'top_category_2': 'Кафе и рестораны',
        'top_category_3': 'Такси'
    })
    
    # Тест карты путешествий
    push = composer.compose_push('Карта для путешествий', test_features, benefit=3200)
    print(f"Travel card push: {push}")
    
    # Тест премиальной карты
    push = composer.compose_push('Премиальная карта', test_features)
    print(f"Premium card push: {push}")
    
    print("Тестирование завершено!")
