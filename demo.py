"""
🏆 ДЕМОНСТРАЦИОННЫЙ СКРИПТ ДЛЯ ХАКАТОНА
Полный цикл: запуск пайплайна + создание визуализаций
"""
import subprocess
import sys
import time
from pathlib import Path

def run_demo(api_key: str):
    """Запускает полную демонстрацию"""
    
    print("🏆 ДЕМОНСТРАЦИЯ ПАЙПЛАЙНА ПЕРСОНАЛИЗИРОВАННЫХ PUSH-УВЕДОМЛЕНИЙ")
    print("=" * 80)
    print("🚀 Асинхронная обработка | 🤖 LLM генерация | 📊 Визуализация результатов")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Запуск основного пайплайна
        print("\n🔥 ШАГ 1: ЗАПУСК АСИНХРОННОГО ПАЙПЛАЙНА")
        print("-" * 50)
        
        pipeline_cmd = [
            "python", "src/pipeline.py", "data/clients.csv", 
            "--api-key", api_key, "--full", "--report"
        ]
        
        result = subprocess.run(pipeline_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Пайплайн выполнен успешно!")
            
            # Извлекаем время выполнения из вывода
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "Время выполнения:" in line:
                    print(f"⚡ {line.strip()}")
                    break
        else:
            print(f"❌ Ошибка пайплайна: {result.stderr}")
            return False
        
        # 2. Создание визуализаций
        print("\n🎨 ШАГ 2: СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
        print("-" * 50)
        
        viz_result = subprocess.run(["python", "src/visualizer.py"], 
                                   capture_output=True, text=True)
        
        if viz_result.returncode == 0:
            print("✅ Визуализации созданы успешно!")
        else:
            print(f"⚠️ Предупреждение: {viz_result.stderr}")
        
        # 3. Итоговая статистика
        total_time = time.time() - start_time
        print(f"\n🏆 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("=" * 50)
        print(f"⏱️ Общее время: {total_time:.2f} секунд")
        print(f"📁 Результаты: output/recommendations.csv")
        print(f"📊 Графики: output/visualizations/")
        
        # Проверяем созданные файлы
        output_files = list(Path("output").glob("**/*"))
        print(f"\n📂 СОЗДАННЫЕ ФАЙЛЫ ({len(output_files)}):")
        for file in sorted(output_files):
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"   📄 {file} ({size_kb:.1f} KB)")
        
        print(f"\n🎉 ГОТОВО К ДЕМОНСТРАЦИИ ЖЮРИ!")
        return True
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return False


def main():
    """Главная функция демонстрации"""
    
    if len(sys.argv) != 2:
        print("❌ Использование: python demo.py your_openai_api_key")
        print("🔑 Получить ключ: https://platform.openai.com/api-keys")
        return
    
    api_key = sys.argv[1]
    
    print("🔍 Проверка готовности...")
    
    # Проверяем наличие данных
    if not Path("data/clients.csv").exists():
        print("❌ Файл data/clients.csv не найден!")
        return
    
    print("✅ Данные найдены")
    print("✅ API ключ получен")
    print("✅ Все готово к демонстрации!")
    
    # Запускаем демонстрацию
    success = run_demo(api_key)
    
    if success:
        print("\n🏆 ДЕМОНСТРАЦИЯ УСПЕШНА! ПРОЕКТ ГОТОВ К ОЦЕНКЕ ЖЮРИ!")
    else:
        print("\n❌ Демонстрация не удалась")


if __name__ == "__main__":
    main()
