"""
基础配置模块 / Base configuration module

定义框架核心配置的数据类，包括数据集、训练和模型配置。
Defines dataclasses for core framework configuration, including dataset, training, and model settings.

支持从字典和YAML文件加载配置，以及序列化为字典和YAML。
Supports loading configuration from dicts and YAML files, and serialization to dicts and YAML.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class DatasetConfig:
    """
    数据集配置 / Dataset configuration

    控制数据来源（公开数据集或自定义数据）以及训练/验证/测试集划分比例。
    Controls data source (public dataset or custom data) and train/val/test split ratios.

    Attributes:
        use_public_dataset: 是否使用内置公开数据集 / Whether to use a built-in public dataset
        dataset_type: 数据集类型，如 'darts', 'financial' 等 / Dataset type, e.g. 'darts', 'financial'
        dataset_name: 数据集名称 / Dataset name
        dataset_config: 数据集额外参数 / Additional dataset parameters
        test_size: 测试集比例 / Test set ratio
        val_size: 验证集比例 / Validation set ratio
    """
    use_public_dataset: bool = True
    dataset_type: str = 'darts'
    dataset_name: str = 'air_passengers'
    
    dataset_config: Dict[str, Any] = field(default_factory=dict)
    
    test_size: float = 0.2
    val_size: float = 0.1


@dataclass
class TrainingConfig:
    """
    训练配置 / Training configuration

    定义模型训练过程的超参数，包括学习率、优化器、调度器、早停等。
    Defines hyperparameters for the training process, including learning rate, optimizer, scheduler, early stopping, etc.

    Attributes:
        train_epochs: 训练轮数 / Number of training epochs
        learning_rate: 学习率 / Learning rate
        weight_decay: 权重衰减 / Weight decay for regularization
        optimizer: 优化器名称 / Optimizer name
        scheduler: 学习率调度器名称 / Learning rate scheduler name
        scheduler_params: 调度器参数 / Scheduler parameters
        batch_size: 批次大小 / Batch size
        num_workers: 数据加载工作线程数 / Number of data loading workers
        early_stopping_patience: 早停耐心值（连续无改善的轮数）/ Early stopping patience (epochs without improvement)
        early_stopping_min_delta: 早停最小改善阈值 / Minimum improvement threshold for early stopping
        gradient_clip_val: 梯度裁剪值 / Gradient clipping value
        gradient_accumulation_steps: 梯度累积步数 / Gradient accumulation steps
    """
    train_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    optimizer: str = 'adam'
    scheduler: str = 'cosine'
    # 使用 lambda 工厂构造默认 dict,避免所有实例共享同一 dict 对象的陷阱
    # (Python 可变默认参数问题: 直接写 default={'T_max':100} 会被所有实例共享)。
    # Use a lambda factory for mutable default; raw dict default would be shared
    # across all instances due to Python's mutable-default gotcha.
    scheduler_params: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 100,
        'eta_min': 1e-6
    })
    
    batch_size: int = 32
    num_workers: int = 4
    
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    gradient_clip_val: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class ModelConfig:
    """
    模型配置 / Model configuration

    定义模型结构参数，包括模型类型、序列长度、隐藏层大小等。
    Defines model architecture parameters, including model type, sequence lengths, hidden size, etc.

    Attributes:
        model_name: 模型名称，如 'lstm', 'transformer' / Model name, e.g. 'lstm', 'transformer'
        task_type: 任务类型，如 'forecasting' / Task type, e.g. 'forecasting'
        seq_len: 输入序列长度 / Input sequence length
        label_len: 标签序列长度（用于Transformer解码器输入）/ Label sequence length (for Transformer decoder input)
        pred_len: 预测长度 / Prediction horizon length
        hidden_size: 隐藏层维度 / Hidden layer dimension
        num_layers: 网络层数 / Number of layers
        dropout: Dropout比例 / Dropout rate
        model_specific_config: 模型特有配置 / Model-specific configuration
    """
    model_name: str = 'lstm'
    task_type: str = 'forecasting'
    
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    
    model_specific_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseConfig:
    """
    框架基础配置 / Framework base configuration

    整合数据集、训练、模型配置，并管理项目目录结构和全局设置。
    Integrates dataset, training, and model configs, and manages project directory structure and global settings.

    初始化时自动创建所需目录（data/logs/results/models）。
    Automatically creates required directories (data/logs/results/models) on initialization.

    Attributes:
        project_root: 项目根目录 / Project root directory
        data_dir: 原始数据目录 / Raw data directory
        processed_data_dir: 处理后数据目录 / Processed data directory
        log_dir: 日志目录 / Log directory
        result_dir: 结果目录 / Results directory
        model_save_dir: 模型保存目录 / Model save directory
        random_seed: 随机种子 / Random seed for reproducibility
        dataset: 数据集配置 / Dataset configuration
        training: 训练配置 / Training configuration
        model: 模型配置 / Model configuration
        log_interval: 日志打印间隔（步数）/ Logging interval (steps)
        save_interval: 模型保存间隔（轮数）/ Model save interval (epochs)
        use_gpu: 是否使用GPU / Whether to use GPU
        gpu_id: GPU设备编号 / GPU device ID
        visualization_config: 可视化配置 / Visualization settings
    """
    project_root: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    data_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw'))
    processed_data_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'processed'))
    
    log_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'logs'))
    result_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'results'))
    model_save_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments', 'models'))
    
    random_seed: int = 42
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    log_interval: int = 100
    save_interval: int = 10
    
    use_gpu: bool = True
    gpu_id: int = 0
    
    visualization_config: Dict[str, Any] = field(default_factory=lambda: {
        'plot_type': 'interactive',
        'save_plots': True,
        'show_plots': False,
        'figure_size': (12, 8),
        'dpi': 100
    })
    
    def __post_init__(self):
        # 自动创建所有必要的目录，exist_ok=True 避免目录已存在时报错
        # Automatically create all required directories; exist_ok=True prevents errors if they already exist
        for dir_path in [self.data_dir, self.processed_data_dir, self.log_dir,
                         self.result_dir, self.model_save_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置序列化为字典 / Serialize configuration to a dictionary"""
        result = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if hasattr(v, 'to_dict'):
                    result[k] = v.to_dict()
                elif hasattr(v, '__dict__') and not isinstance(v, (int, float, str, bool, list, dict)):
                    result[k] = {sk: sv for sk, sv in v.__dict__.items() if not sk.startswith('_')}
                else:
                    result[k] = v
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """
        从字典创建配置实例 / Create a configuration instance from a dictionary

        自动将嵌套字典转换为对应的子配置对象（DatasetConfig/TrainingConfig/ModelConfig）。
        Automatically converts nested dicts into corresponding sub-config objects.

        Args:
            config_dict: 配置字典 / Configuration dictionary

        Returns:
            BaseConfig 实例 / BaseConfig instance
        """
        config = cls()

        # 遍历字典，将嵌套配置还原为对应的 dataclass 实例
        # Iterate dict and reconstruct nested dataclass instances for sub-configs
        for key, value in config_dict.items():
            if key == 'dataset' and isinstance(value, dict):
                config.dataset = DatasetConfig(**value)
            elif key == 'training' and isinstance(value, dict):
                config.training = TrainingConfig(**value)
            elif key == 'model' and isinstance(value, dict):
                config.model = ModelConfig(**value)
            elif hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'BaseConfig':
        """
        从YAML文件加载配置 / Load configuration from a YAML file

        Args:
            yaml_path: YAML文件路径 / Path to the YAML file

        Returns:
            BaseConfig 实例 / BaseConfig instance
        """
        import yaml
        # 读取YAML文件并解析为字典 / Read YAML file and parse into dict
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """
        将配置保存为YAML文件 / Save configuration to a YAML file

        Args:
            yaml_path: YAML文件保存路径 / Path to save the YAML file
        """
        import yaml
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
