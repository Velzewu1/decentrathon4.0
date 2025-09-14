"""
Модуль для экспорта результатов в CSV
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultExporter:
    """Класс для экспорта результатов"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Инициализация экспортера
        
        Args:
            output_dir: директория для сохранения результатов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_recommendations(self, recommendations: pd.DataFrame, 
                              filename: str = "recommendations.csv") -> str:
        """
        Экспорт рекомендаций в CSV файл
        
        Args:
            recommendations: DataFrame с рекомендациями
            filename: имя файла для сохранения
            
        Returns:
            Путь к сохраненному файлу
        """
        logger.info(f"Экспорт рекомендаций в {filename}")
        
        # Проверяем необходимые колонки
        required_columns = ['client_code', 'product', 'push_notification']
        missing_columns = set(required_columns) - set(recommendations.columns)
        
        if missing_columns:
            raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
        
        # Выбираем только нужные колонки в правильном порядке
        export_df = recommendations[required_columns].copy()
        
        # Сортируем по client_code
        export_df = export_df.sort_values('client_code')
        
        # Определяем правильный путь для сохранения
        if Path(filename).is_absolute() or '/' in filename or '\\' in filename:
            # Если filename уже содержит путь, используем его как есть
            output_path = Path(filename)
        else:
            # Если это просто имя файла, добавляем к output_dir
            output_path = self.output_dir / filename
        
        # Создаем директорию если не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем в CSV
        export_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"Рекомендации сохранены в {output_path}")
        logger.info(f"Экспортировано {len(export_df)} записей")
        
        return str(output_path)
    
    def export_full_results(self, features: pd.DataFrame, 
                           benefits: pd.DataFrame,
                           ranked_products: pd.DataFrame,
                           recommendations: pd.DataFrame,
                           filename: str = "full_results.xlsx") -> str:
        """
        Экспорт полных результатов в Excel файл с несколькими листами
        
        Args:
            features: DataFrame с признаками клиентов
            benefits: DataFrame с benefit scores
            ranked_products: DataFrame с ранжированными продуктами
            recommendations: DataFrame с рекомендациями
            filename: имя файла для сохранения
            
        Returns:
            Путь к сохраненному файлу
        """
        logger.info(f"Экспорт полных результатов в {filename}")
        
        # Определяем правильный путь для сохранения
        if Path(filename).is_absolute() or '/' in filename or '\\' in filename:
            output_path = Path(filename)
        else:
            output_path = self.output_dir / filename
        
        # Создаем директорию если не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Создаем Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Лист 1: Основные рекомендации
            recommendations.to_excel(writer, sheet_name='Рекомендации', index=False)
            
            # Лист 2: Топ-4 продукта для каждого клиента
            top_products = self._prepare_top_products(ranked_products)
            top_products.to_excel(writer, sheet_name='Топ-4 продукта', index=False)
            
            # Лист 3: Benefit scores
            benefits_export = self._prepare_benefits(benefits, features)
            benefits_export.to_excel(writer, sheet_name='Benefit Scores', index=False)
            
            # Лист 4: Статистика по продуктам
            product_stats = self._calculate_product_stats(ranked_products, benefits)
            product_stats.to_excel(writer, sheet_name='Статистика продуктов', index=False)
            
            # Лист 5: Сводка по клиентам
            client_summary = self._prepare_client_summary(features, recommendations, ranked_products)
            client_summary.to_excel(writer, sheet_name='Сводка по клиентам', index=False)
        
        logger.info(f"Полные результаты сохранены в {output_path}")
        
        return str(output_path)
    
    def _prepare_top_products(self, ranked_products: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка таблицы с топ-4 продуктами для каждого клиента
        
        Args:
            ranked_products: DataFrame с ранжированными продуктами
            
        Returns:
            DataFrame с топ продуктами
        """
        # Pivot таблица для удобного отображения
        pivot_df = ranked_products.pivot_table(
            index='client_code',
            columns='rank',
            values=['product', 'benefit'],
            aggfunc='first'
        )
        
        # Переименовываем колонки
        pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
        
        # Переименовываем для читаемости
        rename_dict = {}
        for i in range(1, 5):
            rename_dict[f'product_{i}'] = f'Продукт {i}'
            rename_dict[f'benefit_{i}'] = f'Benefit {i}'
        
        pivot_df = pivot_df.rename(columns=rename_dict)
        
        # Упорядочиваем колонки
        column_order = ['client_code']
        for i in range(1, 5):
            column_order.extend([f'Продукт {i}', f'Benefit {i}'])
        
        # Выбираем только существующие колонки
        existing_columns = [col for col in column_order if col in pivot_df.columns]
        pivot_df = pivot_df[existing_columns]
        
        return pivot_df
    
    def _prepare_benefits(self, benefits: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка таблицы с benefit scores
        
        Args:
            benefits: DataFrame с benefit scores
            features: DataFrame с признаками клиентов
            
        Returns:
            DataFrame с benefit scores и информацией о клиентах
        """
        # Добавляем информацию о клиенте
        result = features[['client_code', 'name', 'status', 'city']].merge(
            benefits,
            on='client_code',
            how='left'
        )
        
        # Округляем benefit scores
        benefit_columns = [col for col in result.columns if col.startswith('benefit_')]
        for col in benefit_columns:
            result[col] = result[col].round(0)
        
        # Заменяем -inf на 0
        result = result.replace([np.inf, -np.inf], 0)
        
        return result
    
    def _calculate_product_stats(self, ranked_products: pd.DataFrame, 
                                benefits: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет статистики по продуктам
        
        Args:
            ranked_products: DataFrame с ранжированными продуктами
            benefits: DataFrame с benefit scores
            
        Returns:
            DataFrame со статистикой
        """
        # Считаем, сколько раз каждый продукт попал в топ
        stats = ranked_products.groupby('product').agg({
            'client_code': 'count',
            'benefit': ['mean', 'median', 'min', 'max'],
            'rank': 'mean'
        }).reset_index()
        
        # Переименовываем колонки
        stats.columns = [
            'Продукт', 
            'Количество рекомендаций',
            'Средний benefit',
            'Медианный benefit',
            'Мин. benefit',
            'Макс. benefit',
            'Средний ранг'
        ]
        
        # Округляем числовые значения
        numeric_columns = ['Средний benefit', 'Медианный benefit', 'Мин. benefit', 'Макс. benefit', 'Средний ранг']
        for col in numeric_columns:
            stats[col] = stats[col].round(0)
        
        # Сортируем по количеству рекомендаций
        stats = stats.sort_values('Количество рекомендаций', ascending=False)
        
        # Добавляем процент клиентов
        total_clients = ranked_products['client_code'].nunique()
        stats['% клиентов'] = (stats['Количество рекомендаций'] / total_clients * 100).round(1)
        
        return stats
    
    def _prepare_client_summary(self, features: pd.DataFrame, 
                               recommendations: pd.DataFrame,
                               ranked_products: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка сводки по клиентам
        
        Args:
            features: DataFrame с признаками клиентов
            recommendations: DataFrame с рекомендациями
            ranked_products: DataFrame с ранжированными продуктами
            
        Returns:
            DataFrame со сводкой
        """
        # Базовая информация о клиенте
        summary = features[['client_code', 'name', 'status', 'age', 'city', 
                           'avg_monthly_balance', 'total_spend']].copy()
        
        # Добавляем основной рекомендованный продукт
        main_product = recommendations[['client_code', 'product']].rename(
            columns={'product': 'Основной продукт'}
        )
        summary = summary.merge(main_product, on='client_code', how='left')
        
        # Добавляем суммарный benefit от топ-4
        total_benefit = ranked_products.groupby('client_code')['benefit'].sum().reset_index()
        total_benefit.columns = ['client_code', 'Суммарный benefit']
        summary = summary.merge(total_benefit, on='client_code', how='left')
        
        # Добавляем количество подходящих продуктов
        product_count = ranked_products.groupby('client_code').size().reset_index()
        product_count.columns = ['client_code', 'Количество продуктов']
        summary = summary.merge(product_count, on='client_code', how='left')
        
        # Округляем числовые значения
        summary['avg_monthly_balance'] = summary['avg_monthly_balance'].round(0)
        summary['total_spend'] = summary['total_spend'].round(0)
        summary['Суммарный benefit'] = summary['Суммарный benefit'].round(0)
        
        # Переименовываем колонки
        summary = summary.rename(columns={
            'client_code': 'Код клиента',
            'name': 'Имя',
            'status': 'Статус',
            'age': 'Возраст',
            'city': 'Город',
            'avg_monthly_balance': 'Средний баланс',
            'total_spend': 'Общие траты'
        })
        
        return summary
    
    def export_analysis_report(self, features: pd.DataFrame,
                              benefits: pd.DataFrame,
                              ranked_products: pd.DataFrame,
                              recommendations: pd.DataFrame,
                              filename: str = "analysis_report.txt") -> str:
        """
        Экспорт текстового отчета с анализом
        
        Args:
            features: DataFrame с признаками клиентов
            benefits: DataFrame с benefit scores
            ranked_products: DataFrame с ранжированными продуктами
            recommendations: DataFrame с рекомендациями
            filename: имя файла для сохранения
            
        Returns:
            Путь к сохраненному файлу
        """
        logger.info(f"Создание аналитического отчета {filename}")
        
        # Определяем правильный путь для сохранения
        if Path(filename).is_absolute() or '/' in filename or '\\' in filename:
            output_path = Path(filename)
        else:
            output_path = self.output_dir / filename
        
        # Создаем директорию если не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ ПО ПЕРСОНАЛИЗИРОВАННЫМ РЕКОМЕНДАЦИЯМ\n")
            f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Общая статистика
            f.write("ОБЩАЯ СТАТИСТИКА\n")
            f.write("-" * 40 + "\n")
            f.write(f"Обработано клиентов: {len(features)}\n")
            f.write(f"Сгенерировано рекомендаций: {len(recommendations)}\n")
            f.write(f"Всего продуктов в топ-4: {len(ranked_products)}\n\n")
            
            # Статистика по продуктам
            f.write("СТАТИСТИКА ПО ПРОДУКТАМ\n")
            f.write("-" * 40 + "\n")
            
            product_stats = ranked_products['product'].value_counts()
            for product, count in product_stats.items():
                percentage = count / len(features) * 100
                f.write(f"{product}: {count} клиентов ({percentage:.1f}%)\n")
            
            f.write("\n")
            
            # Топ-3 клиента по benefit
            f.write("ТОП-3 КЛИЕНТА ПО СУММАРНОМУ BENEFIT\n")
            f.write("-" * 40 + "\n")
            
            client_benefits = ranked_products.groupby('client_code')['benefit'].sum().reset_index()
            client_benefits = client_benefits.merge(
                features[['client_code', 'name']], 
                on='client_code'
            )
            top_clients = client_benefits.nlargest(3, 'benefit')
            
            for idx, row in top_clients.iterrows():
                f.write(f"{row['name']} (ID: {row['client_code']}): {row['benefit']:,.0f} ₸\n")
            
            f.write("\n")
            
            # Распределение по статусам клиентов
            f.write("РАСПРЕДЕЛЕНИЕ ПО СТАТУСАМ КЛИЕНТОВ\n")
            f.write("-" * 40 + "\n")
            
            status_dist = features['status'].value_counts()
            for status, count in status_dist.items():
                percentage = count / len(features) * 100
                f.write(f"{status}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n")
            
            # Средние показатели
            f.write("СРЕДНИЕ ПОКАЗАТЕЛИ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Средний баланс: {features['avg_monthly_balance'].mean():,.0f} ₸\n")
            f.write(f"Средние траты: {features['total_spend'].mean():,.0f} ₸\n")
            f.write(f"Средний benefit: {ranked_products['benefit'].mean():,.0f} ₸\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("КОНЕЦ ОТЧЕТА\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Аналитический отчет сохранен в {output_path}")
        
        return str(output_path)


def main():
    """Тестовый запуск экспорта"""
    
    # Создаем тестовые данные
    test_recommendations = pd.DataFrame({
        'client_code': [1, 2, 3],
        'product': ['Карта для путешествий', 'Премиальная карта', 'Депозит'],
        'push_notification': [
            'Рамазан, в августе вы сделали 12 поездок на такси на 27 400 ₸. С картой для путешествий вернули бы ≈1 100 ₸. Откройте карту.',
            'Алия, у вас высокий остаток на счету. Премиальная карта даст до 4% кешбэка и бесплатные снятия. Подключите сейчас.',
            'Петр, свободные средства лучше разместить на вкладе под 14.5%. Открыть вклад.'
        ]
    })
    
    test_features = pd.DataFrame({
        'client_code': [1, 2, 3],
        'name': ['Рамазан', 'Алия', 'Петр'],
        'status': ['Премиальный клиент', 'Студент', 'Зарплатный клиент'],
        'age': [35, 22, 45],
        'city': ['Алматы', 'Астана', 'Шымкент'],
        'avg_monthly_balance': [500000, 50000, 200000],
        'total_spend': [362000, 38500, 195000]
    })
    
    test_benefits = pd.DataFrame({
        'client_code': [1, 2, 3],
        'benefit_Карта для путешествий': [9560, 140, 6720],
        'benefit_Премиальная карта': [18640, 1155, 7800],
        'benefit_Депозит Сберегательный': [94250, 8550, 36000],
        'best_deposit_type': ['Мультивалютный депозит', 'Сберегательный депозит', 'Мультивалютный депозит']
    })
    
    test_ranked = pd.DataFrame({
        'client_code': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        'rank': [1, 2, 3, 4, 1, 2, 3, 1, 2, 3],
        'product': ['Карта для путешествий', 'Премиальная карта', 'Депозит', 'Валютный счет',
                   'Премиальная карта', 'Депозит', 'Валютный счет',
                   'Депозит', 'Премиальная карта', 'Карта для путешествий'],
        'benefit': [9560, 18640, 94250, 1500, 1155, 8550, 312, 36000, 7800, 6720]
    })
    
    # Экспортируем результаты
    exporter = ResultExporter()
    
    # Основной экспорт
    csv_path = exporter.export_recommendations(test_recommendations)
    print(f"Рекомендации сохранены в: {csv_path}")
    
    # Полный экспорт в Excel
    excel_path = exporter.export_full_results(
        test_features, test_benefits, test_ranked, test_recommendations
    )
    print(f"Полные результаты сохранены в: {excel_path}")
    
    # Аналитический отчет
    report_path = exporter.export_analysis_report(
        test_features, test_benefits, test_ranked, test_recommendations
    )
    print(f"Аналитический отчет сохранен в: {report_path}")


if __name__ == "__main__":
    main()
