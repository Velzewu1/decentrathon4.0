#!/usr/bin/env python3
"""
Скрипт для тестирования LLM генерации push-уведомлений
"""
import sys
import os
from pathlib import Path

# Добавляем src в path
sys.path.append(str(Path(__file__).parent / 'src'))

from compose_llm import LLMPushComposer
import pandas as pd
import json

def test_single_generation():
    """Тест генерации одного push-уведомления"""
    
    print("🧪 Тест генерации одного push-уведомления")
    print("=" * 50)
    
    # API ключ
    api_key = os.getenv('OPENAI_API_KEY')
    
    try:
        # Создаем композер
        composer = LLMPushComposer(api_key=api_key)
        
        # Тестовые данные клиента
        test_clients = [
            {
                'client_code': 1,
                'name': 'Айгерим',
                'age': 29,
                'status': 'Зарплатный клиент',
                'avg_monthly_balance_KZT': 92643
            },
            {
                'client_code': 2,
                'name': 'Данияр', 
                'age': 41,
                'status': 'Премиальный клиент',
                'avg_monthly_balance_KZT': 1577073
            },
            {
                'client_code': 3,
                'name': 'Сабина',
                'age': 22,
                'status': 'Студент',
                'avg_monthly_balance_KZT': 63116
            }
        ]
        
        # Тестовые продукты и детали
        test_cases = [
            {
                'product': 'Карта для путешествий',
                'details': {
                    'travel_amount': 45000,
                    'taxi_amount': 27400,
                    'hotel_amount': 18000,
                    'benefit': 2280
                }
            },
            {
                'product': 'Премиальная карта',
                'details': {
                    'restaurant_amount': 152496,
                    'cashback_tier': 3,
                    'free_atm_benefit': 5000,
                    'benefit': 8500
                }
            },
            {
                'product': 'Кредитная карта',
                'details': {
                    'top_categories': ['Продукты питания', 'Кафе и рестораны', 'Такси'],
                    'online_amount': 25000,
                    'has_credit_activity': True,
                    'benefit': 7500
                }
            }
        ]
        
        results = []
        
        for i, client_data in enumerate(test_clients):
            test_case = test_cases[i % len(test_cases)]
            
            print(f"\n👤 Клиент: {client_data['name']} ({client_data['age']} лет)")
            print(f"🏦 Продукт: {test_case['product']}")
            
            # Генерируем push
            push_text = composer.generate_push(
                client_data, 
                test_case['product'], 
                test_case['details']
            )
            
            print(f"💬 Push: {push_text}")
            print(f"📏 Длина: {len(push_text)} символов")
            print(f"✅ Валиден: {'Да' if 50 <= len(push_text) <= 220 else 'Нет'}")
            
            results.append({
                'client': client_data['name'],
                'product': test_case['product'],
                'push': push_text,
                'length': len(push_text),
                'valid': 50 <= len(push_text) <= 220
            })
        
        # Статистика
        print("\n" + "=" * 50)
        print("📊 СТАТИСТИКА ГЕНЕРАЦИИ")
        print("=" * 50)
        
        total = len(results)
        valid = sum(1 for r in results if r['valid'])
        avg_length = sum(r['length'] for r in results) / total
        
        print(f"Всего сгенерировано: {total}")
        print(f"Валидных push: {valid} ({valid/total:.1%})")
        print(f"Средняя длина: {avg_length:.0f} символов")
        
        # Сохраняем результаты
        results_file = "test_llm_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Результаты сохранены в {results_file}")
        print("✅ Тест завершен успешно!")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bulk_generation():
    """Тест массовой генерации на реальных данных"""
    
    print("\n🧪 Тест массовой генерации push-уведомлений")
    print("=" * 50)
    
    # Проверяем наличие данных
    clients_file = Path("data/clients.csv")
    if not clients_file.exists():
        print("❌ Файл data/clients.csv не найден")
        return False
    
    try:
        # Загружаем клиентов
        clients_df = pd.read_csv(clients_file)
        print(f"📊 Загружено {len(clients_df)} клиентов")
        
        # Берем первых 5 для теста
        test_clients = clients_df.head(5)
        
        # API ключ
        api_key = os.getenv('OPENAI_API_KEY')
        
        composer = LLMPushComposer(api_key=api_key)
        
        # Тестируем разные продукты
        products = [
            'Карта для путешествий',
            'Премиальная карта', 
            'Кредитная карта',
            'Депозит Сберегательный',
            'Инвестиции'
        ]
        
        results = []
        
        for i, (_, client_row) in enumerate(test_clients.iterrows()):
            client_data = client_row.to_dict()
            product = products[i % len(products)]
            
            # Генерируем базовые детали
            details = {
                'benefit': 5000 + i * 1000,
                'free_funds': client_data['avg_monthly_balance_KZT'] * 0.3
            }
            
            print(f"\n👤 {client_data['name']} → 🏦 {product}")
            
            push_text = composer.generate_push(client_data, product, details)
            
            results.append({
                'client_code': client_data['client_code'],
                'client_name': client_data['name'],
                'product': product,
                'push_notification': push_text,
                'length': len(push_text)
            })
            
            print(f"💬 {push_text}")
            print(f"📏 {len(push_text)} символов")
        
        # Создаем DataFrame и сохраняем
        results_df = pd.DataFrame(results)
        results_file = "test_bulk_results.csv"
        results_df.to_csv(results_file, index=False, encoding='utf-8')
        
        print(f"\n💾 Результаты сохранены в {results_file}")
        
        # Валидация
        validation_stats = composer.validate_pushes(results_df)
        
        print(f"\n📊 Валидация: {validation_stats['validity_rate']:.1%} качественных push")
        print(f"📏 Средняя длина: {validation_stats['avg_length']:.0f} символов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка массовой генерации: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_templates():
    """Сравнение LLM генерации с шаблонной"""
    
    print("\n🧪 Сравнение LLM vs шаблоны")
    print("=" * 50)
    
    try:
        from compose_v2 import PushComposerV2
        
        # API ключ для LLM
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Создаем оба композера
        llm_composer = LLMPushComposer(api_key=api_key)
        template_composer = PushComposerV2()
        
        # Тестовый клиент
        test_client = {
            'client_code': 1,
            'name': 'Айгерим',
            'age': 29,
            'status': 'Зарплатный клиент',
            'avg_monthly_balance_KZT': 92643
        }
        
        test_details = {
            'taxi_amount': 27400,
            'travel_amount': 45000,
            'benefit': 2280,
            'n_taxi': 12,
            'month': 'августе'
        }
        
        product = 'Карта для путешествий'
        
        print(f"👤 Клиент: {test_client['name']}")
        print(f"🏦 Продукт: {product}")
        
        # LLM генерация
        print(f"\n🤖 LLM Push:")
        llm_push = llm_composer.generate_push(test_client, product, test_details)
        print(f"💬 {llm_push}")
        print(f"📏 Длина: {len(llm_push)} символов")
        
        # Шаблонная генерация
        print(f"\n📝 Template Push:")
        
        # Создаем Series для шаблонного композера
        client_series = pd.Series(test_client)
        details_series = pd.Series(test_details)
        
        template_push = template_composer._compose_travel_card(client_series, details_series)
        print(f"💬 {template_push}")
        print(f"📏 Длина: {len(template_push)} символов")
        
        print(f"\n📊 Сравнение:")
        print(f"LLM более персонализированный: {'Да' if len(llm_push) > len(template_push) else 'Нет'}")
        print(f"Разница в длине: {abs(len(llm_push) - len(template_push))} символов")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка сравнения: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Главная функция тестирования"""
    
    print("🚀 ТЕСТИРОВАНИЕ LLM ГЕНЕРАЦИИ PUSH-УВЕДОМЛЕНИЙ")
    print("=" * 60)
    
    # Проверяем наличие API ключа
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Переменная OPENAI_API_KEY не установлена, используем ключ из кода")
    
    success_count = 0
    total_tests = 3
    
    # Тест 1: Одиночная генерация
    if test_single_generation():
        success_count += 1
    
    # Тест 2: Массовая генерация
    if test_bulk_generation():
        success_count += 1
    
    # Тест 3: Сравнение с шаблонами
    if compare_with_templates():
        success_count += 1
    
    # Итоги
    print(f"\n" + "=" * 60)
    print(f"🎯 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print(f"=" * 60)
    print(f"Успешных тестов: {success_count}/{total_tests}")
    print(f"Процент успеха: {success_count/total_tests:.1%}")
    
    if success_count == total_tests:
        print("✅ Все тесты пройдены успешно!")
        print("🚀 LLM генератор готов к использованию!")
    else:
        print("⚠️  Некоторые тесты не прошли")
        print("🔧 Требуется дополнительная настройка")
    
    return success_count == total_tests


if __name__ == "__main__":
    main()
