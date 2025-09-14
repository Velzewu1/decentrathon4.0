"""
Асинхронный LLM композер push-уведомлений (только OpenAI, без fallback)
Оптимизирован для быстрой генерации с помощью asyncio и AsyncOpenAI
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
import asyncio
from openai import AsyncOpenAI
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
    """Асинхронный LLM композер push-уведомлений"""
    
    def __init__(self, config_path: str = "conf/weights.yaml", 
                 api_key: Optional[str] = None):
        """Инициализация (только LLM режим)"""
        self.config = self._load_config(config_path)
        
        # Настройка AsyncOpenAI API (ОБЯЗАТЕЛЬНО)
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)
            logger.info("✅ Асинхронный LLM режим активирован (OpenAI GPT-4o-mini)")
        else:
            # Отладка: проверяем переменные окружения
            logger.info("🔍 Проверяем переменные окружения...")
            
            # Проверяем все варианты
            api_key = os.getenv('OPENAI_API_KEY')
            
            logger.info(f"🔑 API ключ из окружения: {'FOUND' if api_key else 'NOT FOUND'}")
            
            try:
                if not api_key:
                    raise ValueError("""
❌ OPENAI_API_KEY не найден!

🚀 РЕКОМЕНДУЕМЫЙ СПОСОБ (через аргумент):
python src/pipeline.py data/clients.csv --api-key your_api_key_here

📝 Альтернатива (через .env файл):
echo "OPENAI_API_KEY=your_api_key_here" > .env

🔑 Получить ключ: https://platform.openai.com/api-keys
                    """)
                self.client = AsyncOpenAI(api_key=api_key)
                logger.info("✅ Асинхронный LLM режим активирован (OpenAI GPT-4o-mini)")
            except UnicodeDecodeError as e:
                logger.error(f"Ошибка чтения .env: {e}")
                raise ValueError("""
❌ OPENAI_API_KEY не найден из-за ошибки кодировки .env файла!

🚀 РЕКОМЕНДУЕМОЕ РЕШЕНИЕ:
python src/pipeline.py data/clients.csv --api-key your_api_key_here

📝 Альтернатива (исправить .env):
Убедитесь, что .env файл сохранен в кодировке UTF-8.
echo "OPENAI_API_KEY=your_api_key_here" > .env

🔑 Получить ключ: https://platform.openai.com/api-keys
                """) from e
            except Exception as e:
                raise ValueError(f"❌ Ошибка инициализации AsyncOpenAI API: {e}") from e
        
        self.max_length = 220
        self.model = "gpt-4o-mini"
        self.generation_cache = {}
        
        # Настройки для асинхронности
        self.max_concurrent_requests = 10  # Максимум одновременных запросов
        self.request_delay = 0.1  # Задержка между запросами (секунды)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Ошибка загрузки конфигурации: {e}")
            return {}

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
• ТОЧНО НАЗЫВАЙ ПРОДУКТ: "Карта для путешествий" (НЕ "кредитная карта")!
• Премиальная: пиши "до 4%" (не фиксированные 3%)
• Временные рамки: ВСЕГДА "за последние 3 месяца" (единообразно)
• Кредит наличными: НЕ пиши "дефицит" - пиши "крупные расходы"
• Мультивалютный депозит ТОЛЬКО если есть FX операции!

📝 ПРИМЕРЫ ПРАВИЛЬНЫХ PUSH:
• "Айгерим, за последние 3 месяца вы потратили 67 400 ₸ на рестораны. Кредитная карта вернет до 10% кешбэка с ваших любимых категорий. Оформить карту."
• "Данияр, ваш средний баланс 4,2 млн ₸ дает право на до 4% кешбэк с премиальной карты. Плюс бесплатные снятия по миру. Оформить."
• "Камилла, 190 поездок за последние 3 месяца на 463 000 ₸. Карта для путешествий вернет до 4% кешбэка с поездок и даст доступ к VIP-залам. Оформить."

⚠️ ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА:
• ЕДИНООБРАЗИЕ: всегда "дает" (без ё), "на сберегательном депозите"
• FX: НЕ "обменивать" → "обменяйте валюту по выгодному курсу"
• ДЕПОЗИТЫ: "зарабатывайте на ставке" (НЕ "экономьте на ставках")

Отвечай ТОЛЬКО текстом push-уведомления."""

    def _format_currency(self, amount: float) -> Optional[str]:
        """Форматирование валюты: 2 490 ₸ (скрываем нули)"""
        if pd.isna(amount) or amount <= 0:
            return None  # Возвращаем None для нулевых сумм
        formatted = f"{int(amount):,}".replace(',', ' ')
        return f"{formatted} ₸"

    def _create_user_prompt(self, client_name: str, product: str, client_data: Dict[str, Any]) -> str:
        """Создание пользовательского промпта с персонализацией"""
        prompt = f"Клиент: {client_name}\n"
        
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
                prompt += f"Наблюдение: {trip_count} поездок за последние 3 месяца на {travel_str}. Карта для путешествий вернет до 4% кешбэка — это {cashback_str}."
            else:
                prompt += f"Продукт: Карта для путешествий. До 4% кешбэк с поездок, VIP-залы в аэропортах."
            
        elif product == "Премиальная карта":
            restaurant_spend = client_data.get('spend_Кафе и рестораны', 0)
            cosmetics_spend = client_data.get('spend_Косметика и парфюмерия', 0)
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            monthly_premium = (restaurant_spend + cosmetics_spend) / 3
            
            if monthly_premium > 0:
                monthly_str = self._format_currency(monthly_premium)
                prompt += f"Наблюдение: {monthly_str}/мес за последние 3 месяца на рестораны/косметику. Премиальная карта вернет до 4% кешбэка."
            else:
                prompt += f"Продукт: Премиальная карта. До 4% кешбэка, VIP-обслуживание в банке."
                
        elif product == "Кредитная карта":
            food_spend = client_data.get('spend_Продукты питания', 0)
            restaurant_spend = client_data.get('spend_Кафе и рестораны', 0)
            total_food = food_spend + restaurant_spend
            monthly_food = total_food / 3
            
            if total_food > 0:
                food_str = self._format_currency(total_food)
                monthly_str = self._format_currency(monthly_food)
                potential_cashback_range = f"до {self._format_currency(min(30000, total_food * 0.1))}"
                prompt += f"Наблюдение: {food_str} за последние 3 месяца на еду ({monthly_str}/мес). Кредитная карта вернет до 10% кешбэка — это {potential_cashback_range}."
            else:
                prompt += f"Продукт: Кредитная карта с кешбэком до 10% на популярные категории с лимитом 30 000 ₸/мес."
                
        elif product == "Обмен валют":
            fx_volume = client_data.get('fx_volume_KZT', 0)
            if fx_volume > 0:
                fx_str = self._format_currency(fx_volume)
                prompt += f"Наблюдение: валютные операции {fx_str}. FX: выгодный курс, без комиссии, автопокупка."
            else:
                prompt += f"Продукт: Обмен валют. Выгодный курс 24/7, целевой курс, без комиссий."
                
        elif product == "Кредит наличными":
            outflows = client_data.get('outflows', 0)
            inflows = client_data.get('inflows', 1)
            shortage = max(0, outflows - inflows)
            
            if shortage > 0:
                limit = min(2000000, shortage * 2)
                limit_str = self._format_currency(limit)
                prompt += f"Наблюдение: крупные расходы за последние 3 месяца. Кредит наличными поможет с лимитом до {limit_str} от 12% годовых."
            else:
                prompt += f"Продукт: Кредит наличными от 12% годовых без залога, до 2 000 000 ₸."
                
        elif product in ["Депозит Мультивалютный", "Депозит Сберегательный", "Депозит Накопительный"]:
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            fx_volume = client_data.get('fx_volume_KZT', 0)
            rates = {"Депозит Мультивалютный": "14,5%", 
                    "Депозит Сберегательный": "16,5%", 
                    "Депозит Накопительный": "15,5%"}
            rate = rates.get(product, "15%")
            
            if product == "Депозит Мультивалютный":
                # Всегда говорим про мультивалютный, если он назначен
                if fx_volume > 0:
                    fx_str = self._format_currency(fx_volume)
                    prompt += f"Наблюдение: валютные операции {fx_str} за последние 3 месяца. {product} под {rate} годовых — удобно для работы с валютами, доступ к средствам."
                else:
                    # Даже без FX говорим про мультивалютный, раз он назначен
                    balance_str = self._format_currency(balance) if balance > 0 else ""
                    prompt += f"Наблюдение: средний баланс {balance_str}. {product} под {rate} годовых — диверсификация валют, доступ к средствам."
            elif balance > 0:
                balance_str = self._format_currency(balance)
                prompt += f"Наблюдение: средний баланс {balance_str}. {product} под {rate} годовых — надежное размещение."
            else:
                prompt += f"Продукт: {product} под {rate} годовых. Надежное размещение средств."
                
        elif product == "Инвестиции":
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            
            if balance > 0:
                balance_str = self._format_currency(balance)
                prompt += f"Наблюдение: средний баланс {balance_str} за последние 3 месяца. Инвестиции от 6 ₸, без комиссий на старт — потенциальный рост капитала."
            else:
                prompt += f"Продукт: Инвестиции от 6 ₸. Потенциальный рост капитала, без комиссий на старт."
                
        elif product == "Золотые слитки":
            balance = client_data.get('avg_monthly_balance_KZT', 0)
            
            if balance > 0:
                balance_str = self._format_currency(balance)
                prompt += f"Наблюдение: средний баланс {balance_str}. Золотые слитки 999 пробы — надежная диверсификация портфеля."
            else:
                prompt += f"Продукт: Золотые слитки 999 пробы. Надежная диверсификация портфеля."
        
        return prompt

    async def _generate_push_async(self, client_name: str, product: str, client_data: Dict[str, Any]) -> str:
        """Асинхронная генерация одного push-уведомления"""
        cache_key = f"{client_name}_{product}_{hash(str(sorted(client_data.items())))}"
        
        if cache_key in self.generation_cache:
            logger.info(f"📋 Используем кэш для {client_name}")
            return self.generation_cache[cache_key]
        
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(client_name, product, client_data)
        
        try:
            # Асинхронный запрос к OpenAI
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=150,
                temperature=0.8,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            
            push_text = response.choices[0].message.content.strip()
            
            # Кэшируем результат
            self.generation_cache[cache_key] = push_text
            
            logger.info(f"✅ LLM push сгенерирован: {len(push_text)} символов")
            return push_text
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации push для {client_name}: {e}")
            return f"{client_name}, у нас есть отличное предложение для вас! Оформить {product.lower()}."

    async def _generate_push_batch(self, client_data_list: List[tuple]) -> List[str]:
        """Асинхронная генерация batch push-уведомлений"""
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        async def generate_with_semaphore(client_name, product, client_data):
            async with semaphore:
                logger.info(f"🤖 Генерируем LLM push для {client_name} - {product}")
                result = await self._generate_push_async(client_name, product, client_data)
                # Небольшая задержка между запросами
                await asyncio.sleep(self.request_delay)
                return result
        
        # Создаем задачи для всех клиентов
        tasks = [
            generate_with_semaphore(client_name, product, client_data)
            for client_name, product, client_data in client_data_list
        ]
        
        # Выполняем все задачи параллельно
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Обрабатываем исключения
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client_name, product, _ = client_data_list[i]
                logger.error(f"❌ Ошибка для {client_name}: {result}")
                processed_results.append(f"{client_name}, у нас есть отличное предложение для вас! Оформить {product.lower()}.")
            else:
                processed_results.append(result)
        
        return processed_results

    def compose_push_for_clients(self, features: pd.DataFrame, ranked_products: pd.DataFrame) -> pd.DataFrame:
        """Основная функция композиции push-уведомлений (асинхронная)"""
        logger.info("🤖 Начинаем асинхронную LLM генерацию push-уведомлений")
        
        # Выбираем лучший продукт для каждого клиента
        best_products = ranked_products.groupby('client_code').first().reset_index()
        logger.info(f"Выбрано {len(best_products)} лучших продуктов")
        
        # Подготавливаем данные для batch обработки
        client_data_list = []
        for _, row in best_products.iterrows():
            client_code = row['client_code']
            product = row['product']
            
            # Находим данные клиента
            client_features = features[features['client_code'] == client_code].iloc[0]
            client_name = client_features.get('name', f'Клиент {client_code}')
            
            # Подготавливаем данные клиента
            client_data = {
                'spend_Такси': client_features.get('spend_Такси', 0),
                'spend_Путешествия': client_features.get('spend_Путешествия', 0),
                'spend_Кафе и рестораны': client_features.get('spend_Кафе и рестораны', 0),
                'spend_Косметика и парфюмерия': client_features.get('spend_Косметика и парфюмерия', 0),
                'spend_Продукты питания': client_features.get('spend_Продукты питания', 0),
                'fx_volume_KZT': client_features.get('fx_volume_KZT', 0),
                'avg_monthly_balance_KZT': client_features.get('avg_monthly_balance_KZT', 0),
                'outflows': client_features.get('outflows', 0),
                'inflows': client_features.get('inflows', 1),
            }
            
            client_data_list.append((client_name, product, client_data))
        
        # Запускаем асинхронную генерацию
        start_time = time.time()
        
        # Создаем и запускаем event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        push_notifications = loop.run_until_complete(self._generate_push_batch(client_data_list))
        
        generation_time = time.time() - start_time
        logger.info(f"🏆 Сгенерировано {len(push_notifications)} высококачественных LLM push-уведомлений за {generation_time:.2f} секунд")
        
        # Создаем результат
        result = best_products.copy()
        result['push_notification'] = push_notifications
        
        # Проверяем консистентность продукт ↔ текст (без автозамены)
        for idx, row in result.iterrows():
            product = row['product']
            client_code = row['client_code']
            
            # Просто логируем для отладки
            if product == "Депозит Мультивалютный":
                client_features = features[features['client_code'] == client_code].iloc[0]
                fx_volume = client_features.get('fx_volume_KZT', 0)
                logger.info(f"✅ Клиент {client_code}: {product} (FX: {fx_volume:,.0f} ₸)")
        
        logger.info(f"🏆 Сгенерировано {len(result)} высококачественных LLM push-уведомлений")
        return result[['client_code', 'product', 'push_notification']]


def main():
    """Тестирование асинхронного композера"""
    print("🧪 Тестирование асинхронного LLM композера")
    
    # Создаем тестовые данные
    test_features = pd.DataFrame({
        'client_code': [1, 2, 3],
        'name': ['Айгерим', 'Данияр', 'Сабина'],
        'spend_Кафе и рестораны': [67400, 45600, 32100],
        'spend_Такси': [12300, 8900, 15600],
        'avg_monthly_balance_KZT': [850000, 1200000, 450000],
        'fx_volume_KZT': [0, 125000, 0],
    })
    
    test_ranked = pd.DataFrame({
        'client_code': [1, 2, 3],
        'product': ['Кредитная карта', 'Карта для путешествий', 'Премиальная карта'],
        'benefit': [25000, 18500, 12800]
    })
    
    # Тестируем композер
    composer = PushComposer()
    result = composer.compose_push_for_clients(test_features, test_ranked)
    
    print("\n📱 Сгенерированные push-уведомления:")
    for _, row in result.iterrows():
        print(f"\n👤 Клиент {row['client_code']}: {row['product']}")
        print(f"💬 {row['push_notification']}")
        print(f"📏 Длина: {len(row['push_notification'])} символов")


if __name__ == "__main__":
    main()