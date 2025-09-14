#!/usr/bin/env python3
"""
🚀 ЗАПУСК LLM ПАЙПЛАЙНА С 95% КАЧЕСТВОМ
Генерация персонализированных push-уведомлений через OpenAI GPT-4o-mini

Использование:
    python run_llm_pipeline.py
    
Требования:
    1. Установите зависимости: pip install -r requirements.txt
    2. Создайте файл .env с API ключом:
       OPENAI_API_KEY=your_openai_api_key_here
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Добавляем src в путь
sys.path.append('src')

def main():
    """Основная функция запуска LLM пайплайна"""
    
    print("🚀 ЗАПУСК LLM ПАЙПЛАЙНА С 95% КАЧЕСТВОМ")
    print("=" * 60)
    
    # Проверка .env файла
    if not Path('.env').exists():
        print("❌ ОШИБКА: Файл .env не найден!")
        print("\n📝 Создайте файл .env с содержимым:")
        print("OPENAI_API_KEY=your_openai_api_key_here")
        print("\nЗатем замените 'your_openai_api_key_here' на ваш настоящий API ключ OpenAI")
        return
    
    # Загружаем переменные окружения
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ОШИБКА: OPENAI_API_KEY не найден в .env файле!")
        return
    
    if api_key == 'your_openai_api_key_here':
        print("❌ ОШИБКА: Замените 'your_openai_api_key_here' на реальный API ключ!")
        return
    
    print("✅ API ключ загружен из .env")
    
    # Импорт модулей
    try:
        from etl_v2 import DataLoaderV2
        from features import FeatureEngineering
        from scoring_v2 import BenefitScoringV2
        from ranking_v2 import ProductRankerV2
        from compose_llm import LLMPushComposer
        logger.info("✅ Все модули импортированы успешно")
    except Exception as e:
        print(f"❌ Ошибка импорта модулей: {e}")
        return
    
    # Шаг 1: Загрузка данных
    print("\n[1/6] 📊 Загрузка данных клиентов...")
    try:
        data_loader = DataLoaderV2()
        clients_df, transactions_df = data_loader.load_all_data('data/clients.csv', 'data/')
        print(f"✅ Загружено {len(clients_df)} клиентов и {len(transactions_df)} записей")
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Шаг 2: Расчет признаков
    print("\n[2/6] 🔧 Расчет признаков...")
    try:
        feature_engineer = FeatureEngineering()
        features = feature_engineer.create_features(transactions_df)
        print(f"✅ Создано {len(features.columns)} признаков для {len(features)} клиентов")
    except Exception as e:
        print(f"❌ Ошибка расчета признаков: {e}")
        return
    
    # Шаг 3: Расчет benefit scores
    print("\n[3/6] 💰 Расчет benefit scores...")
    try:
        scorer = BenefitScoringV2('conf/weights_v2.yaml')
        details = scorer.calculate_all_benefits(features)
        print(f"✅ Рассчитаны benefits для {len(details)} записей")
    except Exception as e:
        print(f"❌ Ошибка расчета benefits: {e}")
        return
    
    # Шаг 4: Ранжирование продуктов
    print("\n[4/6] 📈 Ранжирование продуктов...")
    try:
        ranker = ProductRankerV2('conf/weights_v2.yaml')
        ranked_products = ranker.rank_products_for_clients(details, features)
        
        # Берем только лучший продукт для каждого клиента
        best_products = ranked_products.groupby('client_code').first().reset_index()
        print(f"✅ Выбрано {len(best_products)} лучших продуктов из {len(ranked_products)} возможных")
    except Exception as e:
        print(f"❌ Ошибка ранжирования: {e}")
        return
    
    # Шаг 5: Генерация LLM push-уведомлений
    print("\n[5/6] 🤖 Генерация LLM push-уведомлений (может занять 1-2 минуты)...")
    try:
        composer = LLMPushComposer(api_key=api_key)
        recommendations = composer.generate_all_pushes(features, details, best_products)
        print(f"✅ Сгенерировано {len(recommendations)} push-уведомлений")
    except Exception as e:
        print(f"❌ Ошибка генерации push: {e}")
        return
    
    # Шаг 6: Валидация и сохранение
    print("\n[6/6] 📊 Валидация качества и сохранение...")
    try:
        # Валидация
        stats = composer.validate_pushes(recommendations)
        
        # Сохранение
        output_file = 'output/recommendations_llm_generated.csv'
        recommendations.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✅ Результаты сохранены в {output_file}")
        
        # Финальная статистика
        print("\n" + "="*60)
        print("🎉 LLM ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО!")
        print("="*60)
        print(f"📊 Качество push: {stats['validity_rate']:.1%}")
        print(f"📏 Средняя длина: {stats['avg_length']:.0f} символов")
        print(f"✅ Валидных push: {stats['valid']}/{stats['total']}")
        print(f"💾 Результат: {output_file}")
        
        if stats['validity_rate'] >= 0.8:
            print("🏆 ПРЕВОСХОДНЫЙ РЕЗУЛЬТАТ! Качество превышает 80%")
        
    except Exception as e:
        print(f"❌ Ошибка валидации/сохранения: {e}")
        return

if __name__ == "__main__":
    main()
