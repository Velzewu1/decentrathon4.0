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
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("""
❌ OPENAI_API_KEY не найден!

📝 Создайте файл .env в корне проекта:
echo "OPENAI_API_KEY=your_api_key_here" > .env

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
        """Системный промпт для персонализированной лаконичности"""
        return """Ты пишешь персональные банковские push по принципу "персонализированная лаконичность".

🎯 ПРИНЦИП:
• КОНКРЕТНЫЕ ЦИФРЫ - суммы, проценты, количество
• ЛИЧНОЕ НАБЛЮДЕНИЕ - "вы часто...", "у вас много..."
• КОРОТКО О ВЫГОДЕ - что конкретно получит
• ПРОСТОЙ CTA - одно слово

📏 ФОРМАТ: 180-220 символов, без КАПСА, макс 1 "!"

📝 ПРИМЕРЫ ХОРОШИХ PUSH:
• "Айгерим, в августе вы потратили 67 400 ₸ на рестораны. Кредитка вернула бы 6 740 ₸ кешбэком. Оформить."
• "Данияр, ваш баланс 4,2 млн ₸ даёт право на 4% кешбэк с премиальной карты. Плюс бесплатные снятия. Оформить."
• "Камилла, 190 поездок на такси за 463 000 ₸. Тревел-карта вернула бы 18 520 ₸. Оформить."

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
            trip_count = max(1, int(total_travel / 1500))
            cashback = total_travel * 0.04
            
            prompt += f"Наблюдение: {trip_count} поездок на {self._format_currency(total_travel)}. Кешбэк: {self._format_currency(cashback)}."
            
        elif product == "Премиальная карта":
            restaurant_spend = client_data.get('spend_Кафе и рестораны', 0)
            cosmetics_spend = client_data.get('spend_Косметика и парфюмерия', 0)
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            monthly_premium = (restaurant_spend + cosmetics_spend) / 3
            tier = 4 if balance > 6000000 else 3 if balance > 1000000 else 2
            
            prompt += f"Наблюдение: {self._format_currency(monthly_premium)}/мес на рестораны/косметику. Кешбэк: {tier}%."
            
        elif product == "Кредитная карта":
            total_spend = client_data.get('total_spend', 0)
            monthly_spend = total_spend / 3
            potential_cashback = min(total_spend * 0.10, 90000)
            top_cat1 = client_data.get('top_category_1', 'продукты')
            top_cat2 = client_data.get('top_category_2', 'рестораны')
            
            prompt += f"Наблюдение: {self._format_currency(monthly_spend)}/мес на {top_cat1.lower()}, {top_cat2.lower()}. Кешбэк: {self._format_currency(potential_cashback)}."
            
        elif product == "Обмен валют":
            fx_volume = client_data.get('fx_volume', 0)
            monthly_fx = fx_volume / 3
            savings = fx_volume * 0.01
            
            prompt += f"Наблюдение: {self._format_currency(monthly_fx)}/мес валютных операций. Экономия: {self._format_currency(savings)}."
            
        elif product in ["Депозит Сберегательный", "Депозит Накопительный", "Депозит Мультивалютный"]:
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            free_funds = client_data.get('free_funds', 0)
            rate = 16.5 if 'Сберегательный' in product else 15.5 if 'Накопительный' in product else 14.5
            monthly_income = free_funds * (rate/100) / 12
            
            prompt += f"Наблюдение: {self._format_currency(free_funds)} лежат без дела. Доход: {self._format_currency(monthly_income)}/мес под {rate}%."
            
        elif product == "Инвестиции":
            free_funds = client_data.get('free_funds', 0)
            potential_return = free_funds * 0.20 / 12
            
            prompt += f"Наблюдение: {self._format_currency(free_funds)} могут работать. Потенциал: {self._format_currency(potential_return)}/мес."
            
        elif product == "Золотые слитки":
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            gold_allocation = balance * 0.15
            
            prompt += f"Баланс: {self._format_currency(balance)}. Золото 999 пробы для защиты капитала."
            
        elif product == "Кредит наличными":
            outflows = client_data.get('outflows', 0)
            inflows = client_data.get('inflows', 1)
            shortage = max(0, outflows - inflows)
            limit = min(2000000, shortage * 2)
            
            prompt += f"Лимит: {self._format_currency(limit)}. От 12% годовых, без залога."
            
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
