"""
Модуль для генерации персонализированных push-уведомлений через LLM API
Улучшенная версия с использованием OpenAI GPT для более качественной персонализации
"""
import pandas as pd
import numpy as np
import yaml
import openai
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

# Загружаем переменные окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMPushComposer:
    """Класс для генерации push-уведомлений через LLM API"""
    
    def __init__(self, config_path: str = "conf/weights_v2.yaml", 
                 api_key: Optional[str] = None):
        """
        Инициализация
        
        Args:
            config_path: путь к файлу конфигурации
            api_key: OpenAI API ключ
        """
        self.config = self._load_config(config_path)
        
        # Настройка OpenAI API
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Попробуем взять из переменной окружения
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API ключ не найден. Укажите его в параметре api_key или переменной OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        
        self.max_length = 220  # Максимальная длина push
        self.model = "gpt-4o-mini"  # Используем более экономичную модель
        
        # Кэш для избежания повторных запросов
        self.generation_cache = {}
        
        # Месяцы для форматирования
        self.months = {
            1: "январе", 2: "феврале", 3: "марте", 4: "апреле",
            5: "мае", 6: "июне", 7: "июле", 8: "августе",
            9: "сентябре", 10: "октябре", 11: "ноябре", 12: "декабре"
        }
        
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
            Отформатированная строка вида "2 490 ₸"
        """
        if pd.isna(amount) or amount == 0:
            return "0 ₸"
        
        # Форматируем с пробелами между разрядами
        formatted = f"{int(amount):,}".replace(',', ' ')
        return f"{formatted} ₸"
    
    def _create_system_prompt(self) -> str:
        """Создание системного промпта с правилами TOV"""
        return """Ты - эксперт по написанию персонализированных банковских push-уведомлений для казахстанского банка.

🎯 СТРОГО СЛЕДУЙ ШАБЛОНАМ ИЗ ТЗ ХАКАТОНА:

Карта для путешествий:
"{name}, в {месяц} у вас много поездок/такси. С тревел-картой часть расходов вернулась бы кешбэком. Хотите оформить?"

Премиальная карта:  
"{name}, у вас стабильно крупный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас."

Кредитная карта:
"{name}, ваши топ-категории — {cat1}, {cat2}, {cat3}. Кредитная карта даёт до 10% в любимых категориях и на онлайн-сервисы. Оформить карту."

Депозиты:
"{name}, у вас остаются свободные средства. Разместите их на вкладе — удобно копить и получать вознаграждение. Открыть вклад."

🚨 КРИТИЧЕСКИ ВАЖНО:
• ДЛИНА СТРОГО 180-220 СИМВОЛОВ
• Используй ТОЧНЫЕ шаблоны выше, адаптируя под данные клиента
• Имена клиентов с БОЛЬШОЙ буквы (Айгерим, Данияр)
• Идеальная грамматика без ошибок

TONE OF VOICE из ТЗ:
• На равных, просто и по-человечески; доброжелательно
• Обращение на «вы» с маленькой буквы, без драматизации
• Важное — в начало, без воды/канцеляризмов/пассивного залога
• Допустим лёгкий, ненавязчивый юмор
• Для молодёжи: меньше официоза, чуть живее (но без жаргона)

ФОРМАТ из ТЗ:
• БЕЗ КАПС; один восклицательный максимум (только по делу)
• Валюта: "2 490 ₸" (пробел между разрядами и перед символом)  
• CTA: глаголы действия («Оформить», «Открыть», «Настроить»)
• БЕЗ крикливых обещаний/давления

СТРУКТУРА:
1. Персональный контекст (наблюдение по тратам/поведению)
2. Польза/объяснение (как продукт решает задачу)
3. Призыв к действию

АЛГОРИТМ:
1. Выбери подходящий ШАБЛОН из ТЗ
2. Подставь реальные данные клиента
3. Проверь длину 180-220 символов
4. Проверь грамматику и TOV

Отвечай ТОЛЬКО текстом push-уведомления."""

    def _create_user_prompt(self, client_data: Dict[str, Any], product: str, 
                           details: Dict[str, Any]) -> str:
        """
        Создание пользовательского промпта с данными клиента
        
        Args:
            client_data: данные клиента
            product: название продукта  
            details: детали для персонализации
            
        Returns:
            Промпт для LLM
        """
        # Извлекаем ключевую информацию
        name = client_data.get('name', 'Клиент')
        age = client_data.get('age', 30)
        status = client_data.get('status', 'Клиент')
        balance = client_data.get('avg_monthly_balance_KZT', 0)
        
        # Определяем стиль общения по возрасту
        style_note = "Более живой стиль общения" if age < 30 else "Классический вежливый стиль"
        
        prompt = f"""Создай персонализированное push-уведомление для банковского продукта.

КЛИЕНТ:
• Имя: {name}
• Возраст: {age} лет  
• Статус: {status}
• Баланс: {self._format_currency(balance)}
• Стиль общения: {style_note}

ПРОДУКТ: {product}

ДАННЫЕ ДЛЯ ПЕРСОНАЛИЗАЦИИ:
"""
        
        # Добавляем специфичные данные по продукту
        if product == "Карта для путешествий":
            travel_amount = details.get('travel_amount', 0)
            taxi_amount = details.get('taxi_amount', 0)
            hotel_amount = details.get('hotel_amount', 0)
            
            prompt += f"""• Траты на путешествия: {self._format_currency(travel_amount)}
• Траты на такси: {self._format_currency(taxi_amount)}  
• Траты на отели: {self._format_currency(hotel_amount)}
• Потенциальный кешбэк: {self._format_currency(details.get('benefit', 0))}

КОНТЕКСТ: Клиент активно тратит на поездки и путешествия. Покажи конкретную выгоду от 4% кешбэка."""
            
        elif product == "Премиальная карта":
            restaurant_amount = details.get('restaurant_amount', 0)
            cashback_tier = details.get('cashback_tier', 2)
            free_atm_benefit = details.get('free_atm_benefit', 0)
            
            prompt += f"""• Траты в ресторанах: {self._format_currency(restaurant_amount)}
• Уровень кешбэка: {cashback_tier}%  
• Экономия на снятии наличных: {self._format_currency(free_atm_benefit)}
• Потенциальный кешбэк: {self._format_currency(details.get('benefit', 0))}

КОНТЕКСТ: У клиента высокий баланс и/или активные траты в премиум-категориях."""
            
        elif product == "Кредитная карта":
            top_categories = details.get('top_categories', [])
            online_amount = details.get('online_amount', 0)
            credit_activity = details.get('has_credit_activity', False)
            
            categories_str = ", ".join(top_categories[:3]) if top_categories else "разные категории"
            prompt += f"""• Топ-3 категории трат: {categories_str}
• Траты на онлайн-сервисы: {self._format_currency(online_amount)}
• Есть кредитная активность: {'Да' if credit_activity else 'Нет'}
• Потенциальный кешбэк: {self._format_currency(details.get('benefit', 0))}

КОНТЕКСТ: 10% кешбэк на топ-категории и онлайн-сервисы, беспроцентный период."""
            
        elif product == "Обмен валют":
            fx_volume = details.get('fx_volume', 0)
            main_currency = details.get('main_fx_currency', 'USD')
            
            prompt += f"""• Объем валютных операций: {self._format_currency(fx_volume)}
• Основная валюта: {main_currency}
• Экономия на курсе: {self._format_currency(details.get('benefit', 0))}

КОНТЕКСТ: Выгодный курс в приложении, автопокупка по целевому курсу."""
            
        elif product in ["Депозит Сберегательный", "Депозит Накопительный", "Депозит Мультивалютный"]:
            free_funds = details.get('free_funds', 0)
            rate = details.get('rate', 15.0)
            
            prompt += f"""• Свободные средства: {self._format_currency(free_funds)}
• Ставка по депозиту: {rate}%
• Потенциальный доход: {self._format_currency(details.get('benefit', 0))}

КОНТЕКСТ: Выгодное размещение свободных средств под проценты."""
            
        elif product == "Инвестиции":
            free_funds = details.get('free_funds', 0)
            expected_return = details.get('expected_return', 0)
            
            prompt += f"""• Свободные средства: {self._format_currency(free_funds)}
• Ожидаемая доходность: 20% годовых
• Потенциальный доход: {self._format_currency(expected_return)}

КОНТЕКСТ: Инвестиции с низким порогом входа от 6 тенге, без комиссий в первый год."""
            
        elif product == "Золотые слитки":
            balance_for_gold = details.get('balance', 0)
            jewelry_amount = details.get('jewelry_amount', 0)
            
            prompt += f"""• Текущий баланс: {self._format_currency(balance_for_gold)}
• Траты на ювелирку: {self._format_currency(jewelry_amount)}

КОНТЕКСТ: Диверсификация портфеля, защита от инфляции, золото 999 пробы."""
            
        elif product == "Кредит наличными":
            monthly_shortage = details.get('monthly_shortage', 0)
            available_limit = min(2000000, abs(monthly_shortage) * 12)
            
            prompt += f"""• Ежемесячная нехватка: {self._format_currency(monthly_shortage)}
• Доступный лимит: {self._format_currency(available_limit)}

КОНТЕКСТ: Быстрое решение финансовых потребностей, от 12% годовых."""
            
        prompt += "\n\nСоздай push-уведомление учитывая ВСЕ правила TOV и персональные данные."
        
        return prompt
    
    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str, 
                            max_retries: int = 3) -> Optional[str]:
        """
        Вызов LLM с повторными попытками
        
        Args:
            system_prompt: системный промпт
            user_prompt: пользовательский промпт  
            max_retries: максимальное количество попыток
            
        Returns:
            Сгенерированный текст или None
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7,  # Немного креативности
                    top_p=0.9,
                    frequency_penalty=0.3,  # Избегаем повторов
                    presence_penalty=0.3
                )
                
                generated_text = response.choices[0].message.content.strip()
                
                # Валидация длины
                if len(generated_text) > self.max_length:
                    logger.warning(f"Сгенерированный текст слишком длинный: {len(generated_text)} символов")
                    # Обрезаем до последнего предложения
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
                logger.error(f"Ошибка вызова LLM (попытка {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Экспоненциальная задержка
                else:
                    return None
    
    def generate_push(self, client_data: Dict[str, Any], product: str, 
                     details: Dict[str, Any]) -> str:
        """
        Генерация персонализированного push-уведомления
        
        Args:
            client_data: данные клиента
            product: название продукта
            details: детали для персонализации
            
        Returns:
            Текст push-уведомления
        """
        # Создаем ключ для кэша
        cache_key = f"{client_data.get('client_code')}_{product}"
        
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
        
        try:
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(client_data, product, details)
            
            logger.info(f"Генерируем push для клиента {client_data.get('name')} - {product}")
            
            generated_push = self._call_llm_with_retry(system_prompt, user_prompt)
            
            if generated_push:
                # Очистка от возможных артефактов
                generated_push = self._clean_generated_text(generated_push)
                
                # Кэшируем результат
                self.generation_cache[cache_key] = generated_push
                
                logger.info(f"✓ Push сгенерирован: {len(generated_push)} символов")
                return generated_push
            else:
                logger.error("Не удалось сгенерировать push через LLM")
                return self._fallback_push(client_data, product, details)
                
        except Exception as e:
            logger.error(f"Ошибка генерации push: {str(e)}")
            return self._fallback_push(client_data, product, details)
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Очистка сгенерированного текста от артефактов
        
        Args:
            text: исходный текст
            
        Returns:
            Очищенный текст
        """
        # Удаляем кавычки в начале и конце
        text = text.strip('"\'')
        
        # Удаляем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Проверяем, что текст заканчивается правильно
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def _fallback_push(self, client_data: Dict[str, Any], product: str, 
                      details: Dict[str, Any]) -> str:
        """
        Резервный механизм генерации push при сбое LLM
        
        Args:
            client_data: данные клиента
            product: название продукта
            details: детали для персонализации
            
        Returns:
            Базовый push-текст
        """
        name = client_data.get('name', 'Клиент')
        
        # Простые шаблоны как fallback
        fallback_templates = {
            "Карта для путешествий": f"{name}, с тревел-картой вернете 4% с поездок и такси. Оформить карту.",
            "Премиальная карта": f"{name}, премиальная карта даст повышенный кешбэк и привилегии. Оформить сейчас.",
            "Кредитная карта": f"{name}, кредитка вернет до 10% с топ-категорий. Оформить карту.",
            "Обмен валют": f"{name}, выгодный курс валют в приложении 24/7. Настроить обмен.",
            "Депозит Сберегательный": f"{name}, сберегательный депозит под 16,5% годовых. Открыть вклад.",
            "Депозит Накопительный": f"{name}, накопительный депозит под 15,5% с пополнением. Открыть вклад.", 
            "Депозит Мультивалютный": f"{name}, мультивалютный депозит под 14,5%. Защитите от курсовых колебаний. Открыть вклад.",
            "Инвестиции": f"{name}, инвестиции с порога от 6 тенге без комиссий. Открыть счет.",
            "Золотые слитки": f"{name}, золото 999 пробы для защиты капитала. Узнать подробнее.",
            "Кредит наличными": f"{name}, кредит наличными от 12% без залога. Узнать лимит."
        }
        
        return fallback_templates.get(product, f"{name}, выгодный банковский продукт для вас. Узнать подробнее.")
    
    def generate_all_pushes(self, features: pd.DataFrame, details: pd.DataFrame, 
                           ranked_products: pd.DataFrame) -> pd.DataFrame:
        """
        Генерация push-уведомлений только для лучшего продукта каждого клиента
        
        Args:
            features: признаки клиентов
            details: детали продуктов
            ranked_products: ранжированные продукты (топ-4 для каждого клиента)
            
        Returns:
            DataFrame с рекомендациями (по одной на клиента)
        """
        logger.info("Начинаем генерацию push-уведомлений через LLM")
        
        # Берем только лучший продукт для каждого клиента (первый в ранжировании)
        best_products = ranked_products.groupby('client_code').first().reset_index()
        
        logger.info(f"Выбрано {len(best_products)} лучших продуктов из {len(ranked_products)} возможных")
        
        recommendations = []
        
        for _, row in best_products.iterrows():
            client_code = row['client_code']
            product = row['product']
            
            # Получаем данные клиента
            client_data = features[features['client_code'] == client_code].iloc[0].to_dict()
            
            # Получаем детали продукта
            product_details = details[details['client_code'] == client_code].iloc[0].to_dict()
            
            # Генерируем push с retry логикой для качества 80%+
            push_text = None
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                attempts += 1
                try:
                    push_text = self.generate_push(client_data, product, product_details)
                    
                    # Пост-обработка для достижения 180-220 символов
                    if len(push_text) < 180:
                        logger.info(f"📏 Push слишком короткий ({len(push_text)} символов), удлиняем...")
                        push_text = self._extend_short_push(push_text, client_data, product)
                    
                    # Исправление грамматических ошибок
                    push_text = self._fix_grammar_errors(push_text)
                    
                    # Проверяем качество - если хорошо, выходим
                    if self._is_valid_push(push_text):
                        break
                        
                except Exception as e:
                    logger.error(f"Ошибка генерации push (попытка {attempts}): {e}")
                    
            # Если все попытки неудачны - используем fallback
            if not push_text or not self._is_valid_push(push_text):
                push_text = self._generate_fallback_push(client_data, product)
                logger.warning(f"Использован fallback для клиента {client_code}")
            
            recommendations.append({
                'client_code': client_code,
                'product': product,
                'push_notification': push_text
            })
        
        result_df = pd.DataFrame(recommendations)
        
        logger.info(f"✓ Сгенерировано {len(recommendations)} push-уведомлений через LLM")
        
        return result_df
    
    def _is_valid_push(self, push_text: str) -> bool:
        """Проверка качества push-уведомления"""
        if not push_text:
            return False
            
        length = len(push_text)
        exclamation_count = push_text.count('!')
        
        # Основные критерии валидации
        length_ok = 180 <= length <= 220
        exclamation_ok = exclamation_count <= 1
        no_caps = not any(word.isupper() and len(word) > 1 for word in push_text.split())
        
        return length_ok and exclamation_ok and no_caps
    
    def _extend_short_push(self, push_text: str, client_data: Dict[str, Any], product: str) -> str:
        """Автоматическое удлинение коротких push до 180-220 символов"""
        if len(push_text) >= 180:
            return push_text
            
        # Умные расширения в зависимости от продукта и возраста клиента
        age = client_data.get('age', 30)
        is_young = age < 30
        
        extensions = {
            'Кредитная карта': [
                " Оформление займёт всего пару минут в мобильном приложении.",
                " Беспроцентный период до 62 дней на все покупки.",
                " Управляйте лимитами и кешбэком прямо в приложении."
            ],
            'Премиальная карта': [
                " Дополнительно: VIP-обслуживание и консьерж-сервис 24/7.",
                " Бесплатные переводы и снятие наличных по всему миру.",
                " Приоритетное обслуживание во всех отделениях банка."
            ],
            'Карта для путешествий': [
                " Бонус: страховка в поездках и доступ к VIP-залам аэропортов.",
                " Скидки на отели, авиабилеты и аренду автомобилей до 15%.",
                " Без комиссий за операции в любой стране мира."
            ],
            'Депозит Накопительный': [
                " Проценты начисляются ежемесячно на основной счёт.",
                " Пополняйте в любое время без ограничений по сумме.",
                " Государственная защита вкладов до 5 млн ₸."
            ]
        }
        
        # Для молодежи - более живые формулировки
        if is_young:
            youth_extensions = [
                " Всё онлайн — никаких походов в банк и бумажек.",
                " Подключайте и управляйте прямо из приложения на телефоне.", 
                " Получите карту курьером или в ближайшем отделении."
            ]
            extensions[product] = extensions.get(product, []) + youth_extensions
        
        # Универсальные расширения
        universal = [
            " Узнайте подробности и оформите в мобильном приложении.",
            " Подключение займёт меньше 5 минут без справок.",
            " Воспользуйтесь выгодным предложением уже сегодня."
        ]
        
        # Выбираем лучшее расширение по длине
        product_extensions = extensions.get(product, universal)
        target_length = 190  # Середина диапазона 180-220
        
        for ext in product_extensions:
            new_push = push_text + ext
            if 180 <= len(new_push) <= 220:
                logger.info(f"📏 Push удлинён с {len(push_text)} до {len(new_push)} символов")
                return new_push
        
        # Если не подошло ни одно - берём первое
        new_push = push_text + product_extensions[0]
        return new_push[:220] if len(new_push) > 220 else new_push
    
    def _fix_grammar_errors(self, push_text: str) -> str:
        """Исправление типичных грамматических ошибок"""
        fixes = {
            'на каждую трата': 'на каждую трату',
            'на каждую покупка': 'на каждую покупку',
            'с каждой трата': 'с каждой тратой',
            'своими трата': 'своими тратами',
            'любимых трата': 'любимых тратах',
            'активные трата': 'активных тратах',
            'частые трата': 'частых тратах'
        }
        
        for wrong, correct in fixes.items():
            push_text = push_text.replace(wrong, correct)
            
        # Исправляем имена - первая буква должна быть заглавной
        words = push_text.split()
        if words and words[0].endswith(','):
            name = words[0][:-1]  # Убираем запятую
            if name.islower() and name.isalpha():
                words[0] = name.capitalize() + ','
                push_text = ' '.join(words)
        
        return push_text
    
    def _generate_fallback_push(self, client_data: Dict[str, Any], product: str) -> str:
        """Резервный генератор на основе шаблонов из ТЗ хакатона"""
        name = client_data['name']
        
        templates = {
            'Кредитная карта': f"{name}, оформите кредитную карту и получайте до 10% кешбэка в любимых категориях и на онлайн-сервисы. Беспроцентный период до 62 дней поможет управлять финансами. Оформить карту.",
            'Премиальная карта': f"{name}, у вас стабильно крупный остаток и траты в ресторанах. Премиальная карта даст повышенный кешбэк и бесплатные снятия. Оформить сейчас.",
            'Карта для путешествий': f"{name}, если у вас есть поездки и такси, с тревел-картой часть расходов вернулась бы кешбэком. Дополнительно страховка и VIP-залы. Хотите оформить?",
            'Депозит Накопительный': f"{name}, у вас остаются свободные средства. Разместите их на вкладе под 15,5% — удобно копить и получать вознаграждение. Открыть вклад.",
            'Депозит Сберегательный': f"{name}, разместите средства на сберегательном депозите под 16,5% годовых. Максимальный доход при готовности заморозить до конца срока. Открыть вклад.",
            'Обмен валют': f"{name}, если часто меняете валюту, в приложении выгодный курс без комиссии 24/7. Можно выставить целевой курс для авто-покупки. Настроить обмен.",
            'Инвестиции': f"{name}, попробуйте инвестиции с низким порогом входа от 6 ₸ и без комиссий на старт. Первый год пополнения и вывод бесплатно. Открыть счёт."
        }
        
        fallback = templates.get(product, f"{name}, у нас есть выгодное предложение по продукту {product}. Узнайте подробности в мобильном приложении или свяжитесь с нами для получения персональных условий.")
        
        # Убеждаемся что fallback попадает в диапазон 180-220
        if len(fallback) < 180:
            fallback += " Оформление займёт несколько минут."
        if len(fallback) > 220:
            fallback = fallback[:217] + "..."
            
        return fallback
    
    def validate_pushes(self, recommendations: pd.DataFrame) -> Dict[str, Any]:
        """
        Валидация сгенерированных push-уведомлений
        
        Args:
            recommendations: DataFrame с рекомендациями
            
        Returns:
            Статистика валидации
        """
        stats = {
            'total': len(recommendations),
            'too_long': 0,
            'too_short': 0,
            'below_180': 0,  # Новая категория для хакатона
            'has_caps': 0,
            'multiple_exclamations': 0,
            'valid': 0,
            'avg_length': 0
        }
        
        lengths = []
        
        for _, row in recommendations.iterrows():
            push = row['push_notification']
            length = len(push)
            lengths.append(length)
            
            # Проверка длины
            if length > 220:
                stats['too_long'] += 1
            elif length < 180:
                stats['below_180'] += 1
            elif length < 50:
                stats['too_short'] += 1
            
            # Проверка CAPS
            if any(word.isupper() and len(word) > 1 for word in push.split()):
                stats['has_caps'] += 1
            
            # Проверка восклицательных знаков
            if push.count('!') > 1:
                stats['multiple_exclamations'] += 1
            
            # Считаем валидными если соответствует ТЗ хакатона
            if 180 <= length <= 220 and push.count('!') <= 1:
                stats['valid'] += 1
        
        stats['avg_length'] = np.mean(lengths) if lengths else 0
        stats['validity_rate'] = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
        
        logger.info(f"Валидация: {stats['valid']}/{stats['total']} валидных push ({stats['validity_rate']:.1%})")
        logger.info(f"📏 Длина: средняя {stats['avg_length']:.0f}, коротких (<180): {stats['below_180']}, длинных (>220): {stats['too_long']}")
        
        return stats


def test_llm_composer():
    """Тест функциональности LLM композера"""
    # Тестовые данные
    test_client = {
        'client_code': 1,
        'name': 'Айгерим',
        'age': 29,
        'status': 'Зарплатный клиент',
        'avg_monthly_balance_KZT': 92643
    }
    
    test_details = {
        'travel_amount': 45000,
        'taxi_amount': 12000,
        'benefit': 2280
    }
    
    # API ключ из переменных окружения
    api_key = os.getenv('OPENAI_API_KEY')
    
    try:
        composer = LLMPushComposer(api_key=api_key)
        push = composer.generate_push(test_client, "Карта для путешествий", test_details)
        print(f"Сгенерированный push: {push}")
        print(f"Длина: {len(push)} символов")
        
    except Exception as e:
        print(f"Ошибка тестирования: {e}")


if __name__ == "__main__":
    test_llm_composer()
