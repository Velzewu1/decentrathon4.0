"""
Чистый LLM композер push-уведомлений (только OpenAI, без fallback)
"""
import pandas as pd
import numpy as np
import yaml
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла (с обработкой ошибок)
try:
    load_dotenv()
except UnicodeDecodeError:
    pass  # Игнорируем ошибки кодировки
except FileNotFoundError:
    pass  # .env файл не обязателен

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PushComposer:
    """Чистый LLM композер push-уведомлений"""
    
    def __init__(self, config_path: str = "conf/weights.yaml", 
                 api_key: Optional[str] = None):
        """Инициализация (только LLM режим)"""
        self.config = self._load_config(config_path)
        
        # Настройка OpenAI API (ОБЯЗАТЕЛЬНО)
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Отладка: проверяем переменные окружения
            logger.info("🔍 Проверяем переменные окружения...")
            
            # Проверяем все варианты
            api_key = os.getenv('OPENAI_API_KEY')
            
            logger.info(f"🔑 API ключ из окружения: {'FOUND' if api_key else 'NOT FOUND'}")
            
            if api_key:
                logger.info(f"🔍 Ключ начинается с: {api_key[:10]}...")
            
            if not api_key:
                # Попробуем прочитать .env файл напрямую
                try:
                    with open('.env', 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.info(f"📄 .env файл найден, размер: {len(content)} символов")
                        
                        # Ищем ключ вручную
                        for line in content.split('\n'):
                            if 'OPENAI_API_KEY' in line and '=' in line:
                                api_key = line.split('=', 1)[1].strip()
                                logger.info(f"🔍 Найден ключ в .env: {api_key[:10]}...")
                                break
                                
                except Exception as e:
                    logger.error(f"Ошибка чтения .env: {e}")
                
                if not api_key:
                    raise ValueError("""
❌ OPENAI_API_KEY не найден!

📝 Проверьте файл .env в корне проекта:
OPENAI_API_KEY=your_api_key_here

🔑 Получить ключ: https://platform.openai.com/api-keys
                    """)
            
            self.client = OpenAI(api_key=api_key)
            logger.info("✅ LLM режим активирован (OpenAI GPT-4o-mini)")
        
        self.max_length = 220
        self.model = "gpt-4o-mini"
        self.generation_cache = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _format_currency(self, amount: float) -> str:
        """Форматирование валюты: 2 490 ₸"""
        if pd.isna(amount) or amount <= 0:
            return None
        formatted = f"{int(amount):,}".replace(',', ' ')
        return f"{formatted} ₸"
    
    def _create_system_prompt(self) -> str:
        """Финальный системный промпт с учетом всех замечаний"""
        return """Ты пишешь персональные банковские push по принципу "персонализированная лаконичность".

🎯 ПРИНЦИП:
• КОНКРЕТНЫЕ ЦИФРЫ - суммы, проценты, количество
• ЛИЧНОЕ НАБЛЮДЕНИЕ - "вы часто...", "у вас много..."
• ПРЕДЛОЖЕНИЕ ПРОДУКТА - что конкретно получит
• ПРОСТОЙ CTA - одно слово

📏 ФОРМАТ: 180-220 символов, без КАПСА, МАКС 1 "!"

⚠️ КРИТИЧНО ИЗБЕГАЙ:
• НЕ пиши "ваш депозит", "ваша карта" - у клиента её ещё нет!
• НЕ пиши "вы получили кешбэк" - это ложь!
• Пиши "депозит принесет", "карта вернет", "могли бы получить", "даст"
• Премиальная: пиши "до 4%" (не фиксированные 3%)
• Продукт "Депозит" не существует - пиши "Депозит Сберегательный"
• Депозиты: пиши "депозит под X% годовых", а НЕ "ваш депозит"

📝 ПРИМЕРЫ ПРАВИЛЬНЫХ PUSH:
• "Айгерим, в августе вы потратили 67 400 ₸ на рестораны. Кредитная карта вернет до 10% кешбэка с ваших любимых категорий. Оформить карту."
• "Данияр, ваш баланс 4,2 млн ₸ даёт право на до 4% кешбэк с премиальной карты. Плюс бесплатные снятия по миру. Оформить."
• "Камилла, 190 поездок на такси за 463 000 ₸. Карта для путешествий вернула бы 4% кешбэка с поездок и дала бы доступ к VIP-залам. Оформить."

Отвечай ТОЛЬКО текстом push-уведомления."""

    def _create_user_prompt(self, client_data: Dict[str, Any], product: str, 
                           details: Dict[str, Any]) -> str:
        """Создание персонализированного промпта"""
        name = client_data.get('name', 'Клиент')
        age = client_data.get('age', 30)
        style = "живо и просто" if age < 30 else "вежливо и дружелюбно"
        
        prompt = f"""Клиент: {name} ({age} лет)
Продукт: {product}
Стиль: {style}

Данные для персонализации:
"""
        
        # Персонализированные данные по продукту
        if product == "Карта для путешествий":
            taxi_spend = client_data.get('spend_Такси', 0)
            travel_spend = client_data.get('spend_Путешествия', 0)
            total_travel = taxi_spend + travel_spend
            
            if total_travel > 0:
                trip_count = max(1, int(total_travel / 1500))
                cashback = total_travel * 0.04
                travel_str = self._format_currency(total_travel)
                cashback_str = self._format_currency(cashback)
                prompt += f"Наблюдение: {trip_count} поездок на {travel_str}. Потенциальный кешбэк: {cashback_str}."
            else:
                prompt += f"Продукт: Карта для путешествий. 4% кешбэк с поездок, VIP-залы."
            
        elif product == "Премиальная карта":
            restaurant_spend = client_data.get('spend_Кафе и рестораны', 0)
            cosmetics_spend = client_data.get('spend_Косметика и парфюмерия', 0)
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            monthly_premium = (restaurant_spend + cosmetics_spend) / 3
            
            if monthly_premium > 0:
                monthly_str = self._format_currency(monthly_premium)
                prompt += f"Наблюдение: {monthly_str}/мес на рестораны/косметику. Премиальная карта: до 4% кешбэка."
            else:
                prompt += f"Продукт: Премиальная карта. До 4% кешбэка, VIP-обслуживание."
            
        elif product == "Кредитная карта":
            total_spend = client_data.get('total_spend', 0)
            monthly_spend = total_spend / 3
            top_cat1 = client_data.get('top_category_1', 'продукты')
            top_cat2 = client_data.get('top_category_2', 'рестораны')
            
            if monthly_spend > 0:
                monthly_str = self._format_currency(monthly_spend)
                prompt += f"Наблюдение: {monthly_str}/мес на {top_cat1.lower()}, {top_cat2.lower()}. Кешбэк: до 10% с лимитом."
            else:
                prompt += f"Продукт: Кредитная карта. До 10% кешбэка, 62 дня без %."
            
        elif product == "Обмен валют":
            fx_volume = client_data.get('fx_volume', 0)
            
            if fx_volume > 0:
                fx_volume_str = self._format_currency(fx_volume)
                prompt += f"Наблюдение: валютные операции на {fx_volume_str}. Выгодный курс 24/7, автопокупка."
            else:
                prompt += f"Продукт: Обмен валют в приложении. Выгодный курс, целевой курс."
            
        elif product in ["Депозит Сберегательный", "Депозит Накопительный", "Депозит Мультивалютный"]:
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            free_funds = client_data.get('free_funds', 0)
            rate = 16.5 if 'Сберегательный' in product else 15.5 if 'Накопительный' in product else 14.5
            
            if free_funds > 0:
                monthly_income = free_funds * (rate/100) / 12
                free_funds_str = self._format_currency(free_funds)
                income_str = self._format_currency(monthly_income)
                prompt += f"Наблюдение: {free_funds_str} могут приносить доход. Потенциал: {income_str}/мес под {rate}%."
            else:
                prompt += f"Продукт: {product} под {rate}% годовых. Надежное размещение средств."
            
        elif product == "Инвестиции":
            free_funds = client_data.get('free_funds', 0)
            age = client_data.get('age', 35)
            
            if free_funds > 0:
                free_funds_str = self._format_currency(free_funds)
                prompt += f"Наблюдение: {free_funds_str} могут потенциально работать. Возраст: {age}. Порог входа: 6₸."
            else:
                prompt += f"Продукт: Инвестиции от 6₸ без комиссий. Подходит для начала."
            
        elif product == "Золотые слитки":
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            
            if balance > 0:
                balance_str = self._format_currency(balance)
                prompt += f"Наблюдение: баланс {balance_str}. Золото 999 пробы для диверсификации."
            else:
                prompt += f"Продукт: Золотые слитки 999 пробы. Диверсификация портфеля."
            
        elif product == "Кредит наличными":
            outflows = client_data.get('outflows', 0)
            inflows = client_data.get('inflows', 1)
            shortage = max(0, outflows - inflows)
            
            if shortage > 0:
                shortage_str = self._format_currency(shortage)
                limit = min(2000000, shortage * 2)
                limit_str = self._format_currency(limit)
                prompt += f"Наблюдение: дефицит {shortage_str}. Лимит: до {limit_str}, от 12%."
            else:
                prompt += f"Продукт: Кредит наличными от 12% без залога."
            
        elif product == "Депозит":  # Исправляем на справильное название
            prompt += f"Продукт: Депозит Сберегательный под 16,5% годовых. Надежное размещение."
            
        prompt += "\n\nНапиши простой, короткий и уникальный push."
        
        return prompt
    
    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str, 
                            max_retries: int = 3) -> Optional[str]:
        """Вызов LLM с повторными попытками"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=150,
                    temperature=0.8,  # Больше креативности
                    top_p=0.9,
                    frequency_penalty=0.5,  # Избегаем повторов
                    presence_penalty=0.3
                )
                
                generated_text = response.choices[0].message.content.strip()
                
                # Валидация длины
                if len(generated_text) > self.max_length:
                    sentences = generated_text.split('.')
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated + sentence + ".") <= self.max_length:
                            truncated += sentence + "."
                        else:
                            break
                    generated_text = truncated.rstrip('.')
                
                return generated_text
                
            except Exception as e:
                logger.error(f"Ошибка LLM (попытка {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Не удалось сгенерировать push через LLM после {max_retries} попыток: {e}")
    
    def generate_push(self, client_data: Dict[str, Any], product: str, 
                     details: Dict[str, Any]) -> str:
        """Генерация push через LLM (чистый режим)"""
        cache_key = f"{client_data.get('client_code')}_{product}"
        
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
        
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(client_data, product, details)
        
        logger.info(f"🤖 Генерируем LLM push для {client_data.get('name')} - {product}")
        
        generated_push = self._call_llm_with_retry(system_prompt, user_prompt)
        generated_push = self._clean_generated_text(generated_push)
        self.generation_cache[cache_key] = generated_push
        
        logger.info(f"✅ LLM push сгенерирован: {len(generated_push)} символов")
        return generated_push
    
    def _clean_generated_text(self, text: str) -> str:
        """Очистка сгенерированного текста"""
        text = text.strip('"\'')
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def compose_push_for_clients(self, features: pd.DataFrame, details: pd.DataFrame, 
                                ranked_products: pd.DataFrame) -> pd.DataFrame:
        """Генерация push через LLM для всех клиентов"""
        logger.info(f"🤖 Начинаем LLM генерацию push-уведомлений")
        
        # Берем только лучший продукт для каждого клиента
        best_products = ranked_products.groupby('client_code').first().reset_index()
        logger.info(f"Выбрано {len(best_products)} лучших продуктов")
        
        recommendations = []
        
        for _, row in best_products.iterrows():
            client_code = row['client_code']
            product = row['product']
            
            # Получаем данные клиента
            client_data = features[features['client_code'] == client_code].iloc[0].to_dict()
            
            # Получаем детали продукта
            product_details = details[details['client_code'] == client_code].iloc[0].to_dict()
            
            # Генерируем push через LLM
            push_text = self.generate_push(client_data, product, product_details)
            
            recommendations.append({
                'client_code': client_code,
                'product': product,
                'push_notification': push_text
            })
        
        result_df = pd.DataFrame(recommendations)
        
        logger.info(f"🏆 Сгенерировано {len(recommendations)} высококачественных LLM push-уведомлений")
        
        return result_df


def test_clean_llm_composer():
    """Тест чистого LLM композера"""
    test_client = {
        'client_code': 1,
        'name': 'Айгерим',
        'age': 29,
        'status': 'Зарплатный клиент',
        'avg_monthly_balance_KZT': 92643,
        'spend_Такси': 45000,
        'spend_Путешествия': 12000,
        'total_spend': 180000,
        'top_category_1': 'Продукты питания',
        'top_category_2': 'Кафе и рестораны'
    }
    
    test_details = {
        'travel_amount': 45000,
        'taxi_amount': 12000,
        'benefit': 2280
    }
    
    try:
        composer = PushComposer()
        
        # Тестируем разные продукты
        products = ["Карта для путешествий", "Кредитная карта", "Премиальная карта"]
        
        for product in products:
            push = composer.generate_push(test_client, product, test_details)
            print(f"\n{product}:")
            print(f"Push: {push}")
            print(f"Длина: {len(push)} символов")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    test_clean_llm_composer()
