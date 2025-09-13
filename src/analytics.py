"""
Модуль для генерации аналитических отчетов
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
# Визуализации отключены (требуется matplotlib)
# import matplotlib.pyplot as plt
# import seaborn as sns

logger = logging.getLogger(__name__)


class AnalyticsReporter:
    """Класс для создания аналитических отчетов"""
    
    def __init__(self, output_dir: str = "output"):
        """
        Инициализация репортера
        
        Args:
            output_dir: Директория для сохранения отчетов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка стиля графиков (отключено)
        # plt.style.use('seaborn-v0_8-darkgrid')
        # sns.set_palette("husl")
    
    def generate_full_report(self, 
                           features_df: pd.DataFrame,
                           benefits_df: pd.DataFrame,
                           recommendations_df: pd.DataFrame) -> Dict:
        """
        Генерация полного аналитического отчета
        
        Args:
            features_df: DataFrame с признаками клиентов
            benefits_df: DataFrame с benefit scores
            recommendations_df: DataFrame с рекомендациями
            
        Returns:
            Словарь с метриками отчета
        """
        logger.info("Генерация полного аналитического отчета")
        
        report = {
            'client_analysis': self._analyze_clients(features_df),
            'product_distribution': self._analyze_products(recommendations_df),
            'benefit_analysis': self._analyze_benefits(benefits_df),
            'segment_analysis': self._analyze_segments(features_df, recommendations_df),
            'quality_metrics': self._calculate_quality_metrics(recommendations_df)
        }
        
        # Сохраняем отчет
        self._save_report(report)
        
        return report
    
    def _analyze_clients(self, features_df: pd.DataFrame) -> Dict:
        """Анализ клиентской базы"""
        analysis = {
            'total_clients': len(features_df),
            'avg_age': features_df['age'].mean(),
            'age_distribution': {
                '18-25': len(features_df[features_df['age'] <= 25]),
                '26-35': len(features_df[(features_df['age'] > 25) & (features_df['age'] <= 35)]),
                '36-45': len(features_df[(features_df['age'] > 35) & (features_df['age'] <= 45)]),
                '46-55': len(features_df[(features_df['age'] > 45) & (features_df['age'] <= 55)]),
                '55+': len(features_df[features_df['age'] > 55])
            },
            'status_distribution': features_df['status'].value_counts().to_dict(),
            'city_distribution': features_df['city'].value_counts().to_dict(),
            'avg_balance': features_df['avg_monthly_balance_KZT'].mean(),
            'total_spending': features_df['total_spend'].sum(),
            'avg_spending': features_df['total_spend'].mean()
        }
        
        # Топ категории трат
        category_cols = [col for col in features_df.columns if col.startswith('spend_')]
        if category_cols:
            category_spending = {}
            for col in category_cols:
                category = col.replace('spend_', '')
                category_spending[category] = features_df[col].sum()
            
            # Сортируем и берем топ-10
            top_categories = dict(sorted(category_spending.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:10])
            analysis['top_spending_categories'] = top_categories
        
        return analysis
    
    def _analyze_products(self, recommendations_df: pd.DataFrame) -> Dict:
        """Анализ распределения продуктов"""
        product_counts = recommendations_df['product'].value_counts()
        
        analysis = {
            'total_recommendations': len(recommendations_df),
            'unique_products': len(product_counts),
            'product_distribution': product_counts.to_dict(),
            'product_percentages': (product_counts / len(recommendations_df) * 100).round(2).to_dict(),
            'most_recommended': product_counts.index[0] if len(product_counts) > 0 else None,
            'least_recommended': product_counts.index[-1] if len(product_counts) > 0 else None
        }
        
        # Проверяем разнообразие
        if len(product_counts) > 0:
            # Индекс Херфиндаля-Хиршмана для концентрации
            market_shares = product_counts / len(recommendations_df)
            hhi = (market_shares ** 2).sum()
            analysis['concentration_index'] = round(hhi, 4)
            analysis['diversity_score'] = round(1 - hhi, 4)  # Чем выше, тем лучше разнообразие
        
        return analysis
    
    def _analyze_benefits(self, benefits_df: pd.DataFrame) -> Dict:
        """Анализ benefit scores"""
        benefit_cols = [col for col in benefits_df.columns if col.startswith('benefit_')]
        
        analysis = {
            'avg_benefits': {},
            'max_benefits': {},
            'positive_benefits': {}
        }
        
        for col in benefit_cols:
            product = col.replace('benefit_', '')
            analysis['avg_benefits'][product] = round(benefits_df[col].mean(), 2)
            analysis['max_benefits'][product] = round(benefits_df[col].max(), 2)
            analysis['positive_benefits'][product] = (benefits_df[col] > 0).sum()
        
        return analysis
    
    def _analyze_segments(self, features_df: pd.DataFrame, recommendations_df: pd.DataFrame) -> Dict:
        """Анализ по сегментам клиентов"""
        # Объединяем данные
        merged = features_df.merge(recommendations_df, on='client_code')
        
        segments = {}
        
        # По статусу
        for status in merged['status'].unique():
            segment_data = merged[merged['status'] == status]
            segments[f'status_{status}'] = {
                'count': len(segment_data),
                'products': segment_data['product'].value_counts().to_dict(),
                'avg_balance': segment_data['avg_monthly_balance_KZT'].mean(),
                'avg_spending': segment_data['total_spend'].mean()
            }
        
        # По возрастным группам
        age_groups = [
            ('Молодежь (18-30)', 18, 30),
            ('Средний возраст (31-45)', 31, 45),
            ('Старшее поколение (45+)', 45, 100)
        ]
        
        for group_name, min_age, max_age in age_groups:
            segment_data = merged[(merged['age'] >= min_age) & (merged['age'] <= max_age)]
            if len(segment_data) > 0:
                segments[f'age_{group_name}'] = {
                    'count': len(segment_data),
                    'products': segment_data['product'].value_counts().to_dict(),
                    'avg_balance': segment_data['avg_monthly_balance_KZT'].mean(),
                    'avg_spending': segment_data['total_spend'].mean()
                }
        
        return segments
    
    def _calculate_quality_metrics(self, recommendations_df: pd.DataFrame) -> Dict:
        """Расчет метрик качества"""
        metrics = {}
        
        # Проверяем длину push-уведомлений
        push_lengths = recommendations_df['push_notification'].str.len()
        metrics['avg_push_length'] = round(push_lengths.mean(), 1)
        metrics['max_push_length'] = push_lengths.max()
        metrics['pushes_over_220'] = (push_lengths > 220).sum()
        
        # Проверяем персонализацию (содержит ли имя клиента)
        has_name = recommendations_df.apply(
            lambda row: any(name in row['push_notification'] 
                          for name in row['push_notification'].split(',')[0].split()),
            axis=1
        )
        metrics['personalized_pushes'] = has_name.sum()
        metrics['personalization_rate'] = round(has_name.mean() * 100, 1)
        
        # Проверяем наличие сумм в push
        has_amount = recommendations_df['push_notification'].str.contains(r'\d+\s*₸')
        metrics['pushes_with_amounts'] = has_amount.sum()
        metrics['amount_inclusion_rate'] = round(has_amount.mean() * 100, 1)
        
        return metrics
    
    def _save_report(self, report: Dict):
        """Сохранение отчета в файлы"""
        # Конвертируем numpy типы в Python типы для JSON
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        # Сохраняем в JSON
        import json
        json_path = self.output_dir / 'analytics_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy(report), f, ensure_ascii=False, indent=2)
        
        # Сохраняем в текстовый файл
        txt_path = self.output_dir / 'analytics_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("АНАЛИТИЧЕСКИЙ ОТЧЕТ ПО РЕКОМЕНДАЦИЯМ\n")
            f.write("="*60 + "\n\n")
            
            # Клиентский анализ
            f.write("1. АНАЛИЗ КЛИЕНТСКОЙ БАЗЫ\n")
            f.write("-"*40 + "\n")
            client_analysis = report['client_analysis']
            f.write(f"Всего клиентов: {client_analysis['total_clients']}\n")
            f.write(f"Средний возраст: {client_analysis['avg_age']:.1f} лет\n")
            f.write(f"Средний баланс: {client_analysis['avg_balance']:,.0f} ₸\n")
            f.write(f"Средние траты: {client_analysis['avg_spending']:,.0f} ₸\n")
            
            f.write("\nРаспределение по статусам:\n")
            for status, count in client_analysis['status_distribution'].items():
                f.write(f"  {status}: {count}\n")
            
            f.write("\nТоп-5 категорий трат:\n")
            if 'top_spending_categories' in client_analysis:
                for i, (cat, amount) in enumerate(list(client_analysis['top_spending_categories'].items())[:5], 1):
                    f.write(f"  {i}. {cat}: {amount:,.0f} ₸\n")
            
            # Продуктовый анализ
            f.write("\n2. РАСПРЕДЕЛЕНИЕ ПРОДУКТОВ\n")
            f.write("-"*40 + "\n")
            product_analysis = report['product_distribution']
            f.write(f"Уникальных продуктов: {product_analysis['unique_products']}\n")
            f.write(f"Индекс разнообразия: {product_analysis.get('diversity_score', 0):.2%}\n")
            
            f.write("\nРекомендации по продуктам:\n")
            for product, pct in product_analysis['product_percentages'].items():
                f.write(f"  {product}: {pct}%\n")
            
            # Метрики качества
            f.write("\n3. МЕТРИКИ КАЧЕСТВА\n")
            f.write("-"*40 + "\n")
            quality = report['quality_metrics']
            f.write(f"Средняя длина push: {quality['avg_push_length']} символов\n")
            f.write(f"Push > 220 символов: {quality['pushes_over_220']}\n")
            f.write(f"Уровень персонализации: {quality['personalization_rate']}%\n")
            f.write(f"Push с суммами: {quality['amount_inclusion_rate']}%\n")
            
            f.write("\n" + "="*60 + "\n")
        
        logger.info(f"Отчет сохранен в {txt_path}")
    
    def create_visualizations_placeholder(self, 
                            features_df: pd.DataFrame,
                            recommendations_df: pd.DataFrame):
        """Создание визуализаций"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Распределение продуктов
        product_counts = recommendations_df['product'].value_counts()
        axes[0, 0].pie(product_counts.values, labels=product_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Распределение рекомендованных продуктов')
        
        # 2. Возрастное распределение
        axes[0, 1].hist(features_df['age'], bins=20, edgecolor='black')
        axes[0, 1].set_xlabel('Возраст')
        axes[0, 1].set_ylabel('Количество клиентов')
        axes[0, 1].set_title('Возрастное распределение клиентов')
        
        # 3. Топ категории трат
        category_cols = [col for col in features_df.columns if col.startswith('spend_')]
        if category_cols:
            category_sums = {}
            for col in category_cols[:10]:  # Топ-10
                category_sums[col.replace('spend_', '')] = features_df[col].sum()
            
            categories = list(category_sums.keys())
            values = list(category_sums.values())
            axes[1, 0].barh(categories, values)
            axes[1, 0].set_xlabel('Сумма трат (₸)')
            axes[1, 0].set_title('Топ категории по сумме трат')
        
        # 4. Статусы клиентов
        status_counts = features_df['status'].value_counts()
        axes[1, 1].bar(range(len(status_counts)), status_counts.values)
        axes[1, 1].set_xticks(range(len(status_counts)))
        axes[1, 1].set_xticklabels(status_counts.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Количество')
        axes[1, 1].set_title('Распределение по статусам')
        
        plt.tight_layout()
        
        # Сохраняем
        viz_path = self.output_dir / 'analytics_visualization.png'
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Визуализация сохранена в {viz_path}")


if __name__ == "__main__":
    # Тестовый запуск
    import sys
    sys.path.append('src')
    from etl import DataLoaderV2
    from features import FeatureEngineering
    from scoring import BenefitScoringV2
    
    # Загружаем данные
    loader = DataLoaderV2()
    clients_df, transactions_df = loader.load_all_data('data/clients.csv', 'data')
    
    # Создаем признаки
    fe = FeatureEngineering('conf/weights.yaml')
    features = fe.create_features(transactions_df)
    
    # Считаем benefits
    scorer = BenefitScoringV2('conf/weights_v2.yaml')
    benefits = scorer.calculate_all_benefits(features)
    
    # Загружаем рекомендации
    recommendations = pd.read_csv('output/final_recommendations.csv')
    
    # Генерируем отчет
    reporter = AnalyticsReporter()
    report = reporter.generate_full_report(features, benefits, recommendations)
    # reporter.create_visualizations(features, recommendations)  # Требует matplotlib
    
    print("Аналитический отчет создан!")
