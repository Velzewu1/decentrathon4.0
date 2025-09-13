# 📁 Output Directory

Эта папка содержит все выходные файлы и результаты работы пайплайна персонализированных push-уведомлений.

## 📋 Типы файлов

### Основные результаты:
- **`recommendations.csv`** - Основной результат: client_code, product, push_notification
- **`recommendations_full.xlsx`** - Детальные результаты в Excel с несколькими листами
- **`recommendations_report.txt`** - Аналитический отчет с статистикой

### Структура файлов:

#### recommendations.csv
```csv
client_code,product,push_notification
1,Кредитная карта,"Айгерим, ваши топ-категории — ..."
2,Депозит Мультивалютный,"Данияр, вы платите в USD..."
```

#### recommendations_full.xlsx
- **Лист "Рекомендации"**: Основные результаты
- **Лист "Топ-4 продукта"**: Все топ-4 продукта для каждого клиента
- **Лист "Benefit Scores"**: Детальные benefit scores по всем продуктам
- **Лист "Статистика продуктов"**: Агрегированная статистика
- **Лист "Сводка по клиентам"**: Информация о клиентах с результатами

#### recommendations_report.txt
- Общая статистика обработки
- Распределение продуктов
- Топ клиенты по benefit
- Средние показатели
- Распределение по статусам клиентов

## 🚀 Как использовать

```bash
# Базовый запуск (результаты в output/)
python src/pipeline.py data/clients.csv

# С полными отчетами
python src/pipeline.py data/clients.csv --full --report

# Кастомное имя файла
python src/pipeline.py data/clients.csv -o output/my_results.csv
```

## 📊 Для хакатона

Основной файл для сдачи: **`recommendations.csv`**

Формат соответствует требованиям:
- client_code (int)
- product (string) 
- push_notification (string, ≤220 символов)
