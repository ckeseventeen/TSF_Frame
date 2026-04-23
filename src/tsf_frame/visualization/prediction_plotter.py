import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional
from .base_visualizer import BaseVisualizer


class PredictionPlotter(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.plot_type = config.get('plot_type', 'interactive')
    
    def plot_predictions(self, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                        title: str = 'Predictions vs Actual', **kwargs) -> Any:
        if self.plot_type == 'interactive':
            return self._plot_interactive(y_true, y_pred, title, **kwargs)
        else:
            return self._plot_static(y_true, y_pred, title, **kwargs)
    
    def _plot_static(self, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                     title: str, **kwargs) -> plt.Figure:
        num_cols = len(y_true.columns)
        fig, axes = plt.subplots(num_cols, 1, figsize=self.figure_size, squeeze=False)
        
        for idx, col in enumerate(y_true.columns):
            ax = axes[idx, 0]
            ax.plot(y_true.index, y_true[col], label='Actual', alpha=0.8)
            ax.plot(y_pred.index, y_pred[col], label='Predicted', alpha=0.8, linestyle='--')
            ax.set_title(f'{col} - {title}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.get_save_path(f'predictions_{title.lower().replace(" ", "_")}.png')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def _plot_interactive(self, y_true: pd.DataFrame, y_pred: pd.DataFrame,
                          title: str, **kwargs) -> go.Figure:
        num_cols = len(y_true.columns)
        fig = make_subplots(rows=num_cols, cols=1, subplot_titles=y_true.columns.tolist())
        
        for idx, col in enumerate(y_true.columns):
            row = idx + 1
            
            fig.add_trace(
                go.Scatter(x=y_true.index, y=y_true[col], name='Actual',
                          line=dict(color='blue'), opacity=0.8),
                row=row, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=y_pred.index, y=y_pred[col], name='Predicted',
                          line=dict(color='red', dash='dash'), opacity=0.8),
                row=row, col=1
            )
        
        fig.update_layout(height=300 * num_cols, title_text=title, showlegend=True)
        
        if self.save_plots:
            save_path = self.get_save_path(f'predictions_{title.lower().replace(" ", "_")}.html')
            fig.write_html(save_path)
        
        if self.show_plots:
            fig.show()
        
        return fig
    
    def plot_metrics(self, metrics: Dict[str, float],
                     title: str = 'Performance Metrics', **kwargs) -> Any:
        if self.plot_type == 'interactive':
            return self._plot_metrics_interactive(metrics, title, **kwargs)
        else:
            return self._plot_metrics_static(metrics, title, **kwargs)
    
    def _plot_metrics_static(self, metrics: Dict[str, float], title: str, **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color='skyblue', alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.get_save_path(f'metrics_{title.lower().replace(" ", "_")}.png')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def _plot_metrics_interactive(self, metrics: Dict[str, float], title: str, **kwargs) -> go.Figure:
        fig = go.Figure(data=[
            go.Bar(x=list(metrics.keys()), y=list(metrics.values()),
                  text=[f'{v:.4f}' for v in metrics.values()],
                  textposition='auto')
        ])
        
        fig.update_layout(title_text=title, xaxis_title='Metrics', yaxis_title='Value')
        
        if self.save_plots:
            save_path = self.get_save_path(f'metrics_{title.lower().replace(" ", "_")}.html')
            fig.write_html(save_path)
        
        if self.show_plots:
            fig.show()
        
        return fig
    
    def plot_training_history(self, history: Dict[str, list],
                              title: str = 'Training History', **kwargs) -> Any:
        if self.plot_type == 'interactive':
            return self._plot_training_interactive(history, title, **kwargs)
        else:
            return self._plot_training_static(history, title, **kwargs)
    
    def _plot_training_static(self, history: Dict[str, list], title: str, **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        epochs = range(1, len(history.get('train_loss', [])) + 1)
        
        if 'train_loss' in history:
            ax.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], label='Validation Loss', marker='s')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.get_save_path(f'training_history_{title.lower().replace(" ", "_")}.png')
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if self.show_plots:
            plt.show()
        
        return fig
    
    def _plot_training_interactive(self, history: Dict[str, list], title: str, **kwargs) -> go.Figure:
        fig = go.Figure()
        
        epochs = list(range(1, len(history.get('train_loss', [])) + 1))
        
        if 'train_loss' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'],
                                    name='Train Loss', mode='lines+markers'))
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'],
                                    name='Validation Loss', mode='lines+markers'))
        
        fig.update_layout(title_text=title, xaxis_title='Epoch', yaxis_title='Loss')
        
        if self.save_plots:
            save_path = self.get_save_path(f'training_history_{title.lower().replace(" ", "_")}.html')
            fig.write_html(save_path)
        
        if self.show_plots:
            fig.show()
        
        return fig
