"""
Custom visualization functions for credit risk EDA
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class CreditRiskVisualizer:
    """Banking-specific visualizations for credit risk analysis"""
    
    def __init__(self, bank_style=True):
        self.bank_style = bank_style
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup banking-appropriate visualization styles"""
        if self.bank_style:
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # Bank color scheme
            self.colors = {
                'low_risk': '#2E8B57',  # Sea green
                'medium_risk': '#FFD700',  # Gold
                'high_risk': '#DC143C',  # Crimson
                'fraud': '#8B0000',  # Dark red
                'non_fraud': '#006400',  # Dark green
                'corporate_blue': '#003366',
                'corporate_gray': '#666666'
            }
        else:
            self.colors = sns.color_palette("husl", 8).as_hex()
    
    def create_correlation_heatmap(self, df, method='pearson', figsize=(14, 12)):
        """Create enhanced correlation heatmap for banking data"""
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) < 2:
            return None
        
        corr_matrix = numerical_df.corr(method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(
            f'{method.title()} Correlation Matrix - Credit Risk Features',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def create_distribution_grid(self, df, columns=None, n_cols=3, figsize=(18, 12)):
        """Create grid of distribution plots with banking styling"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_rows = (len(columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                data = df[col].dropna()
                
                # Histogram with KDE
                axes[idx].hist(data, bins=50, alpha=0.7, density=True, 
                              edgecolor='black', color=self.colors['corporate_blue'])
                
                # Add KDE
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 1000)
                    axes[idx].plot(x_range, kde(x_range), 'r-', linewidth=2)
                except:
                    pass
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                skew_val = data.skew()
                
                axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                 alpha=0.7, label=f'Mean: {mean_val:.2f}')
                axes[idx].axvline(median_val, color='green', linestyle='--', 
                                 alpha=0.7, label=f'Median: {median_val:.2f}')
                
                axes[idx].set_title(f'{col}\nSkew: {skew_val:.2f}', fontsize=12)
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Density')
                axes[idx].legend(fontsize=9)
                axes[idx].grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Distributions - Credit Risk Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def create_fraud_analysis_plots(self, df, fraud_col='FraudResult'):
        """Create comprehensive fraud analysis plots"""
        if fraud_col not in df.columns:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fraud Distribution', 'Amount by Fraud Status',
                          'Fraud by Hour', 'Fraud by Product Category'),
            specs=[[{'type': 'pie'}, {'type': 'box'}],
                  [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 1. Pie chart - Fraud distribution
        fraud_counts = df[fraud_col].value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Non-Fraud', 'Fraud'],
                values=fraud_counts.values,
                marker=dict(colors=[self.colors['non_fraud'], self.colors['fraud']]),
                hole=0.3
            ),
            row=1, col=1
        )
        
        # 2. Box plot - Amount by fraud status
        if 'Amount' in df.columns:
            fraud_amounts = df[df[fraud_col] == 1]['Amount'].dropna()
            non_fraud_amounts = df[df[fraud_col] == 0]['Amount'].dropna()
            
            fig.add_trace(
                go.Box(
                    y=non_fraud_amounts,
                    name='Non-Fraud',
                    marker_color=self.colors['non_fraud']
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Box(
                    y=fraud_amounts,
                    name='Fraud',
                    marker_color=self.colors['fraud']
                ),
                row=1, col=2
            )
        
        # 3. Bar chart - Fraud by hour
        if 'TransactionStartTime' in df.columns:
            df_time = df.copy()
            df_time['hour'] = pd.to_datetime(df_time['TransactionStartTime']).dt.hour
            fraud_by_hour = df_time.groupby('hour')[fraud_col].mean() * 100
            
            fig.add_trace(
                go.Bar(
                    x=fraud_by_hour.index,
                    y=fraud_by_hour.values,
                    name='Fraud Rate by Hour',
                    marker_color='crimson'
                ),
                row=2, col=1
            )
        
        # 4. Bar chart - Fraud by product category
        if 'ProductCategory' in df.columns:
            fraud_by_category = df.groupby('ProductCategory')[fraud_col].mean() * 100
            fraud_by_category = fraud_by_category.sort_values(ascending=False).head(10)
            
            fig.add_trace(
                go.Bar(
                    x=fraud_by_category.index,
                    y=fraud_by_category.values,
                    name='Fraud Rate by Category',
                    marker_color='darkred'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Fraud Analysis Dashboard",
            title_font=dict(size=20)
        )
        
        return fig
    
    def create_rfm_analysis(self, rfm_data):
        """Create RFM analysis visualizations"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RFM Customer Segmentation', 'Recency Distribution',
                          'Frequency Distribution', 'Monetary Distribution'),
            specs=[[{'type': 'pie'}, {'type': 'histogram'}],
                  [{'type': 'histogram'}, {'type': 'histogram'}]]
        )
        
        # 1. Pie chart - Customer segments
        if 'Segment' in rfm_data.columns:
            segment_counts = rfm_data['Segment'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=segment_counts.index,
                    values=segment_counts.values,
                    hole=0.3
                ),
                row=1, col=1
            )
        
        # 2. Histogram - Recency
        if 'recency' in rfm_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=rfm_data['recency'],
                    nbinsx=50,
                    name='Recency',
                    marker_color='blue'
                ),
                row=1, col=2
            )
        
        # 3. Histogram - Frequency
        if 'frequency' in rfm_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=rfm_data['frequency'],
                    nbinsx=50,
                    name='Frequency',
                    marker_color='green'
                ),
                row=2, col=1
            )
        
        # 4. Histogram - Monetary
        if 'monetary_total' in rfm_data.columns:
            fig.add_trace(
                go.Histogram(
                    x=rfm_data['monetary_total'],
                    nbinsx=50,
                    name='Monetary',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="RFM Customer Analysis",
            title_font=dict(size=20)
        )
        
        return fig
    
    def create_temporal_analysis(self, df, time_col='TransactionStartTime', value_col='Amount'):
        """Create temporal analysis visualizations"""
        if time_col not in df.columns:
            return None
        
        df_time = df.copy()
        df_time[time_col] = pd.to_datetime(df_time[time_col])
        
        # Resample to daily frequency
        daily_data = df_time.set_index(time_col)[value_col].resample('D').agg(['count', 'sum', 'mean'])
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Daily Transaction Count', 'Daily Total Amount', 'Daily Average Amount'),
            shared_xaxes=True
        )
        
        # 1. Daily transaction count
        fig.add_trace(
            go.Scatter(
                x=daily_data.index,
                y=daily_data['count'],
                mode='lines',
                name='Transaction Count',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Daily total amount
        fig.add_trace(
            go.Scatter(
                x=daily_data.index,
                y=daily_data['sum'],
                mode='lines',
                name='Total Amount',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # 3. Daily average amount
        fig.add_trace(
            go.Scatter(
                x=daily_data.index,
                y=daily_data['mean'],
                mode='lines',
                name='Average Amount',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=900,
            showlegend=True,
            title_text="Temporal Transaction Patterns",
            title_font=dict(size=20)
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Total Amount", row=2, col=1)
        fig.update_yaxes(title_text="Average Amount", row=3, col=1)
        
        return fig