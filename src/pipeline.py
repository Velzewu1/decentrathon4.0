"""
Главный пайплайн для генерации персонализированных push-уведомлений
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Добавляем текущую директорию в path
sys.path.append(str(Path(__file__).parent))

# Импортируем модули
from etl import DataLoader
from features import FeatureEngineering
from scoring import BenefitScoring
from ranking import ProductRanking
from compose import PushComposer
from export import ResultExporter

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RecommendationPipeline:
    """Основной пайплайн для генерации рекомендаций"""
    
    def __init__(self, config_path: str = "conf/weights.yaml",
                 templates_path: str = "conf/templates.yaml"):
        """
        Инициализация пайплайна
        
        Args:
            config_path: путь к файлу конфигурации
            templates_path: путь к файлу с шаблонами
        """
        logger.info("Инициализация пайплайна")
        
        self.data_loader = DataLoader(config_path)
        self.feature_eng = FeatureEngineering(config_path)
        self.scoring = BenefitScoring(config_path)
        self.ranking = ProductRanking(config_path)
        self.composer = PushComposer(config_path, templates_path)
        self.exporter = ResultExporter()
        
        logger.info("Пайплайн инициализирован успешно")
    
    def run(self, input_file: str, output_file: str = "recommendations.csv",
            export_full: bool = False, export_report: bool = False) -> Dict[str, Any]:
        """
        Запуск полного пайплайна
        
        Args:
            input_file: путь к входному CSV файлу
            output_file: путь к выходному файлу с рекомендациями
            export_full: экспортировать ли полные результаты в Excel
            export_report: создавать ли аналитический отчет
            
        Returns:
            Словарь с результатами и статистикой
        """
        logger.info("=" * 50)
        logger.info("ЗАПУСК ПАЙПЛАЙНА")
        logger.info("=" * 50)
        
        start_time = datetime.now()
        results = {}
        
        try:
            # Шаг 1: Загрузка и предобработка данных
            logger.info("\n[1/7] Загрузка и предобработка данных...")
            processed_data = self.data_loader.preprocess(input_file)
            results['processed_rows'] = len(processed_data)
            logger.info(f"✓ Обработано {len(processed_data)} транзакций")
            
            # Шаг 2: Создание признаков
            logger.info("\n[2/7] Расчет агрегированных признаков...")
            features = self.feature_eng.create_features(processed_data)
            results['total_clients'] = len(features)
            results['total_features'] = len(features.columns)
            logger.info(f"✓ Создано {len(features.columns)} признаков для {len(features)} клиентов")
            
            # Шаг 3: Расчет benefit scores
            logger.info("\n[3/7] Расчет benefit scores для продуктов...")
            benefits = self.scoring.calculate_all_benefits(features)
            product_columns = [col for col in benefits.columns if col.startswith('benefit_')]
            results['products_scored'] = len(product_columns)
            logger.info(f"✓ Рассчитаны scores для {len(product_columns)} продуктов")
            
            # Шаг 4: Получение деталей для продуктов
            logger.info("\n[4/7] Подготовка деталей продуктов...")
            details = self.scoring.get_product_details(features, benefits)
            logger.info("✓ Детали продуктов подготовлены")
            
            # Шаг 5: Ранжирование продуктов
            logger.info("\n[5/7] Ранжирование и выбор топ-4 продуктов...")
            ranked_products = self.ranking.rank_products(benefits)
            results['total_recommendations'] = len(ranked_products)
            logger.info(f"✓ Создано {len(ranked_products)} рекомендаций")
            
            # Шаг 6: Генерация push-уведомлений
            logger.info("\n[6/7] Генерация push-уведомлений...")
            recommendations = self.composer.generate_all_pushes(features, details, ranked_products)
            results['pushes_generated'] = len(recommendations)
            logger.info(f"✓ Сгенерировано {len(recommendations)} push-уведомлений")
            
            # Шаг 7: Экспорт результатов
            logger.info("\n[7/7] Экспорт результатов...")
            
            # Основной экспорт
            output_path = self.exporter.export_recommendations(recommendations, output_file)
            results['output_file'] = output_path
            logger.info(f"✓ Рекомендации сохранены в {output_path}")
            
            # Дополнительные экспорты
            if export_full:
                excel_path = self.exporter.export_full_results(
                    features, benefits, ranked_products, recommendations,
                    filename=output_file.replace('.csv', '_full.xlsx')
                )
                results['excel_file'] = excel_path
                logger.info(f"✓ Полные результаты сохранены в {excel_path}")
            
            if export_report:
                report_path = self.exporter.export_analysis_report(
                    features, benefits, ranked_products, recommendations,
                    filename=output_file.replace('.csv', '_report.txt')
                )
                results['report_file'] = report_path
                logger.info(f"✓ Аналитический отчет сохранен в {report_path}")
            
            # Расчет времени выполнения
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            results['execution_time'] = execution_time
            
            # Статистика по продуктам
            product_stats = ranked_products['product'].value_counts().to_dict()
            results['product_distribution'] = product_stats
            
            # Итоговое сообщение
            logger.info("\n" + "=" * 50)
            logger.info("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО")
            logger.info(f"Время выполнения: {execution_time:.2f} секунд")
            logger.info(f"Обработано клиентов: {results['total_clients']}")
            logger.info(f"Создано рекомендаций: {results['pushes_generated']}")
            logger.info("=" * 50)
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в пайплайне: {str(e)}")
            raise
    
    def validate_input(self, input_file: str) -> bool:
        """
        Валидация входного файла
        
        Args:
            input_file: путь к входному файлу
            
        Returns:
            True если файл валиден
        """
        logger.info(f"Валидация входного файла: {input_file}")
        
        # Проверка существования файла
        if not Path(input_file).exists():
            logger.error(f"Файл не найден: {input_file}")
            return False
        
        # Проверка формата
        if not input_file.endswith('.csv'):
            logger.error("Файл должен быть в формате CSV")
            return False
        
        # Проверка структуры
        try:
            df = pd.read_csv(input_file, nrows=5)
            required_columns = [
                'client_code', 'name', 'status', 'age', 'city',
                'avg_monthly_balance_KZT', 'date', 'category',
                'amount', 'currency', 'type', 'direction'
            ]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                logger.error(f"Отсутствуют обязательные колонки: {missing_columns}")
                return False
            
            logger.info("✓ Входной файл валиден")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при чтении файла: {str(e)}")
            return False
    
    def generate_sample_data(self, output_file: str = "data/sample_data.csv", 
                            n_clients: int = 10, n_months: int = 3) -> str:
        """
        Генерация тестовых данных
        
        Args:
            output_file: путь для сохранения тестовых данных
            n_clients: количество клиентов
            n_months: количество месяцев данных
            
        Returns:
            Путь к сгенерированному файлу
        """
        logger.info(f"Генерация тестовых данных для {n_clients} клиентов")
        
        np.random.seed(42)  # Для воспроизводимости
        
        # Списки для генерации
        names = ['Иван', 'Мария', 'Алексей', 'Елена', 'Дмитрий', 
                'Анна', 'Сергей', 'Ольга', 'Андрей', 'Наталья']
        statuses = ['Студент', 'Зарплатный клиент', 'Премиальный клиент', 'Стандартный клиент']
        cities = ['Алматы', 'Астана', 'Шымкент', 'Караганда', 'Актобе']
        categories = ['Продукты', 'Рестораны', 'Такси', 'Путешествия', 'Отели',
                     'Ювелирные изделия', 'Косметика и парфюмерия', 'Онлайн-сервисы',
                     'Едим дома', 'Смотрим дома', 'Играем дома', 'Транспорт', 
                     'Одежда', 'Электроника', 'Образование']
        currencies = ['KZT', 'KZT', 'KZT', 'KZT', 'USD', 'EUR']  # Больше KZT
        types_out = ['card_out', 'atm_withdrawal', 'p2p_out', 'fx_buy', 'fx_sell']
        types_in = ['salary_in', 'stipend_in', 'cashback_in', 'refund_in']
        
        data = []
        
        for client_id in range(1, n_clients + 1):
            # Генерируем профиль клиента
            name = names[client_id % len(names)]
            status = np.random.choice(statuses)
            age = np.random.randint(18, 65)
            city = np.random.choice(cities)
            
            # Баланс зависит от статуса
            if status == 'Премиальный клиент':
                avg_balance = np.random.randint(500000, 2000000)
            elif status == 'Зарплатный клиент':
                avg_balance = np.random.randint(150000, 500000)
            elif status == 'Стандартный клиент':
                avg_balance = np.random.randint(50000, 150000)
            else:  # Студент
                avg_balance = np.random.randint(10000, 50000)
            
            # Генерируем транзакции
            n_transactions = np.random.randint(20, 100)
            
            for _ in range(n_transactions):
                # Дата в пределах последних n_months месяцев
                days_ago = np.random.randint(0, n_months * 30)
                date = pd.Timestamp.now() - pd.Timedelta(days=days_ago)
                
                # Направление транзакции (больше расходов)
                direction = np.random.choice(['out', 'in'], p=[0.8, 0.2])
                
                if direction == 'out':
                    category = np.random.choice(categories)
                    transaction_type = np.random.choice(types_out)
                    
                    # Сумма зависит от категории
                    if category in ['Путешествия', 'Отели']:
                        amount = np.random.randint(50000, 300000)
                    elif category in ['Ювелирные изделия']:
                        amount = np.random.randint(30000, 200000)
                    elif category in ['Рестораны']:
                        amount = np.random.randint(5000, 30000)
                    elif category in ['Такси']:
                        amount = np.random.randint(1000, 10000)
                    else:
                        amount = np.random.randint(1000, 50000)
                else:
                    category = 'Доход'
                    transaction_type = np.random.choice(types_in)
                    amount = np.random.randint(10000, avg_balance)
                
                currency = np.random.choice(currencies)
                
                data.append({
                    'client_code': client_id,
                    'name': name,
                    'status': status,
                    'age': age,
                    'city': city,
                    'avg_monthly_balance_KZT': avg_balance,
                    'date': date.strftime('%Y-%m-%d'),
                    'category': category,
                    'amount': amount,
                    'currency': currency,
                    'type': transaction_type,
                    'direction': direction
                })
        
        # Создаем DataFrame и сохраняем
        df = pd.DataFrame(data)
        
        # Создаем директорию если не существует
        Path(output_file).parent.mkdir(exist_ok=True)
        
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"✓ Тестовые данные сохранены в {output_file}")
        logger.info(f"  Клиентов: {n_clients}")
        logger.info(f"  Транзакций: {len(df)}")
        
        return output_file


def main():
    """Главная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description='Пайплайн для генерации персонализированных push-уведомлений'
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default=None,
        help='Путь к входному CSV файлу с транзакциями'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='recommendations.csv',
        help='Путь к выходному файлу (по умолчанию: recommendations.csv)'
    )
    
    parser.add_argument(
        '--full',
        action='store_true',
        help='Экспортировать полные результаты в Excel'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Создать аналитический отчет'
    )
    
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Сгенерировать тестовые данные'
    )
    
    parser.add_argument(
        '--n-clients',
        type=int,
        default=10,
        help='Количество клиентов для генерации тестовых данных (по умолчанию: 10)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Только проверить входной файл без запуска пайплайна'
    )
    
    args = parser.parse_args()
    
    # Инициализация пайплайна
    pipeline = RecommendationPipeline()
    
    # Генерация тестовых данных если запрошено
    if args.generate_sample:
        sample_file = pipeline.generate_sample_data(
            n_clients=args.n_clients
        )
        if not args.input_file:
            args.input_file = sample_file
            print(f"\nИспользуем сгенерированные данные: {sample_file}")
    
    # Проверка наличия входного файла
    if not args.input_file:
        print("\nОшибка: не указан входной файл!")
        print("Используйте: python pipeline.py <input_file.csv>")
        print("Или сгенерируйте тестовые данные: python pipeline.py --generate-sample")
        return
    
    # Валидация входного файла
    if not pipeline.validate_input(args.input_file):
        print("\nВходной файл не прошел валидацию!")
        return
    
    if args.validate_only:
        print("\nВалидация успешна!")
        return
    
    # Запуск пайплайна
    try:
        results = pipeline.run(
            input_file=args.input_file,
            output_file=args.output,
            export_full=args.full,
            export_report=args.report
        )
        
        # Вывод результатов
        print("\n" + "=" * 50)
        print("РЕЗУЛЬТАТЫ")
        print("=" * 50)
        print(f"Обработано транзакций: {results['processed_rows']}")
        print(f"Обработано клиентов: {results['total_clients']}")
        print(f"Создано признаков: {results['total_features']}")
        print(f"Оценено продуктов: {results['products_scored']}")
        print(f"Создано рекомендаций: {results['total_recommendations']}")
        print(f"Сгенерировано push-уведомлений: {results['pushes_generated']}")
        print(f"Время выполнения: {results['execution_time']:.2f} сек")
        print(f"\nРезультаты сохранены в: {results['output_file']}")
        
        if 'excel_file' in results:
            print(f"Полные результаты: {results['excel_file']}")
        
        if 'report_file' in results:
            print(f"Аналитический отчет: {results['report_file']}")
        
        print("\n✓ Пайплайн завершен успешно!")
        
    except Exception as e:
        print(f"\n✗ Ошибка при выполнении пайплайна: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
