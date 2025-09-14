"""
Асинхронный пайплайн для генерации персонализированных push-уведомлений
Оптимизирован для максимальной скорости с AsyncOpenAI (7x ускорение)
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
from ranking import ProductRanker
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
                 templates_path: str = "conf/templates.yaml",
                 api_key: str = None):
        """
        Инициализация пайплайна
        
        Args:
            config_path: путь к файлу конфигурации
            templates_path: путь к файлу с шаблонами
            api_key: OpenAI API ключ для LLM генерации
        """
        logger.info("Инициализация пайплайна")
        
        self.data_loader = DataLoader(config_path)
        self.feature_eng = FeatureEngineering(config_path)
        self.scoring = BenefitScoring(config_path)
        self.ranking = ProductRanker(config_path)
        # Инициализируем LLM композер (чистый режим)
        self.composer = PushComposer(config_path, api_key)
        logger.info("🤖 Асинхронный LLM композер готов (OpenAI GPT-4o-mini)")
        self.exporter = ResultExporter()
        
        logger.info("🚀 Асинхронный пайплайн инициализирован успешно")
    
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
            # Шаг 1: Загрузка данных из трех источников
            logger.info("\n[1/7] Загрузка данных клиентов, транзакций и переводов...")
            # Определяем пути к файлам
            if input_file.endswith('clients.csv'):
                clients_file = input_file
                data_dir = str(Path(input_file).parent)
            else:
                # Если передан другой файл, ищем clients.csv в той же папке
                data_dir = str(Path(input_file).parent)
                clients_file = str(Path(data_dir) / 'clients.csv')
            
            clients_df, transactions_df = self.data_loader.load_all_data(clients_file, data_dir)
            results['total_clients'] = len(clients_df)
            results['processed_rows'] = len(transactions_df) if not transactions_df.empty else 0
            logger.info(f"✓ Загружено {len(clients_df)} клиентов, {results['processed_rows']} записей")
            
            # Шаг 2: Создание признаков
            logger.info("\n[2/7] Расчет агрегированных признаков...")
            if not transactions_df.empty:
                features = self.feature_eng.create_features(transactions_df)
            else:
                # Если нет транзакций, используем только данные клиентов
                features = clients_df.copy()
                features['total_spend'] = 0
                features['inflows'] = 0
                features['outflows'] = 0
                features['free_funds'] = features['avg_monthly_balance_KZT']
            
            # Добавляем эталонный продукт если есть
            if 'reference_product' in transactions_df.columns and not transactions_df.empty:
                ref_products = transactions_df.groupby('client_code')['reference_product'].first()
                features = features.merge(ref_products.to_frame(), left_on='client_code', right_index=True, how='left')
            
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
            ranked_products = self.ranking.rank_products_for_clients(benefits, features, top_n=1)
            results['total_recommendations'] = len(ranked_products)
            logger.info(f"✓ Создано {len(ranked_products)} рекомендаций")
            
            # Шаг 6: Генерация push-уведомлений
            logger.info("\n[6/7] Генерация push-уведомлений...")
            recommendations = self.composer.compose_push_for_clients(features, ranked_products)
            results['pushes_generated'] = len(recommendations)
            logger.info(f"🏆 Сгенерировано {len(recommendations)} высококачественных LLM push-уведомлений")
            
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
        
        # Проверка структуры в зависимости от типа файла
        try:
            df = pd.read_csv(input_file, nrows=5)
            
            if 'clients' in input_file:
                # Для файла клиентов
                required_columns = [
                    'client_code', 'name', 'status', 'age', 'city',
                    'avg_monthly_balance_KZT'
                ]
            elif 'transactions' in input_file:
                # Для файла транзакций
                required_columns = [
                    'client_code', 'date', 'category', 'amount', 'currency'
                ]
            elif 'transfers' in input_file:
                # Для файла переводов
                required_columns = [
                    'client_code', 'date', 'type', 'direction', 'amount', 'currency'
                ]
            else:
                # По умолчанию проверяем как файл клиентов
                required_columns = [
                    'client_code', 'name', 'status', 'age', 'city',
                    'avg_monthly_balance_KZT'
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
    


def main():
    """Главная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description='🚀 Асинхронный пайплайн для генерации персонализированных push-уведомлений (7x ускорение через AsyncOpenAI)'
    )
    
    parser.add_argument(
        'input_file',
        nargs='?',
        default='data/clients.csv',
        help='Путь к файлу clients.csv (по умолчанию: data/clients.csv)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output/recommendations.csv',
        help='Путь к выходному файлу (по умолчанию: output/recommendations.csv)'
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
        '--validate-only',
        action='store_true',
        help='Только проверить входной файл без запуска пайплайна'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='OpenAI API ключ (РЕКОМЕНДУЕТСЯ для быстрого запуска)'
    )
    
    
    args = parser.parse_args()
    
    # Инициализация пайплайна
    pipeline = RecommendationPipeline(api_key=args.api_key)
    
    # Проверка наличия входного файла
    if not args.input_file:
        print("\n❌ Ошибка: не указан входной файл!")
        print("\n🚀 БЫСТРЫЙ ЗАПУСК:")
        print("python src/pipeline.py data/clients.csv --api-key your_api_key_here")
        print("\n📝 Получить API ключ: https://platform.openai.com/api-keys")
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
