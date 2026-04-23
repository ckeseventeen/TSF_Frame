import sys
import os
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve()
for _p in (_HERE.parent, *_HERE.parents):
    if (_p / 'configs').is_dir() and (_p / 'src').is_dir():
        for _q in (_p, _p / 'src'):
            if str(_q) not in sys.path:
                sys.path.insert(0, str(_q))
        break

from tsf_frame.data.datasets.public_datasets import get_dataset, list_available_datasets


def example_list_datasets():
    print("=" * 60)
    print("可用的公开数据集")
    print("=" * 60)
    
    datasets = list_available_datasets()
    
    for category, dataset_list in datasets.items():
        print(f"\n{category.upper()} 数据集:")
        for dataset in dataset_list:
            print(f"  - {dataset}")
    
    print("\n" + "=" * 60)


def example_load_darts_dataset():
    print("\n" + "=" * 60)
    print("加载 Darts 数据集示例")
    print("=" * 60)
    
    config = {
        'dataset_name': 'air_passengers'
    }
    
    dataset = get_dataset('darts', config)
    data = dataset.load()
    
    print(f"\n数据集名称: {dataset.get_metadata()['dataset_name']}")
    print(f"数据形状: {dataset.get_metadata()['shape']}")
    print(f"数据列: {dataset.get_metadata()['columns']}")
    print(f"\n前5行数据:")
    print(data.head())
    
    train_data, test_data = dataset.split_train_test(test_size=0.2)
    print(f"\n训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    print("\n" + "=" * 60)


def example_load_synthetic_dataset():
    print("\n" + "=" * 60)
    print("加载合成数据集示例")
    print("=" * 60)
    
    config = {
        'n_samples': 500,
        'seasonality_period': 100,
        'trend_strength': 0.05,
        'noise_level': 0.2
    }
    
    dataset = get_dataset('synthetic', config)
    data = dataset.load()
    
    print(f"\n数据集名称: {dataset.get_metadata()['dataset_name']}")
    print(f"数据形状: {dataset.get_metadata()['shape']}")
    print(f"数据列: {dataset.get_metadata()['columns']}")
    print(f"\n前5行数据:")
    print(data.head())
    
    print("\n" + "=" * 60)


def example_load_financial_dataset():
    print("\n" + "=" * 60)
    print("加载金融数据集示例")
    print("=" * 60)
    
    config = {
        'ticker': 'AAPL',
        'start_date': '2020-01-01',
        'end_date': '2023-01-01'
    }
    
    dataset = get_dataset('financial', config)
    data = dataset.load()
    
    print(f"\n数据集名称: {dataset.get_metadata()['dataset_name']}")
    print(f"数据形状: {dataset.get_metadata()['shape']}")
    print(f"数据列: {dataset.get_metadata()['columns']}")
    print(f"\n前5行数据:")
    print(data.head())
    
    print("\n" + "=" * 60)


def example_load_energy_dataset():
    print("\n" + "=" * 60)
    print("加载能源数据集示例")
    print("=" * 60)
    
    config = {
        'data_type': 'electricity',
        'location': 'US'
    }
    
    dataset = get_dataset('energy', config)
    data = dataset.load()
    
    print(f"\n数据集名称: {dataset.get_metadata()['dataset_name']}")
    print(f"数据形状: {dataset.get_metadata()['shape']}")
    print(f"数据列: {dataset.get_metadata()['columns']}")
    print(f"\n前5行数据:")
    print(data.head())
    
    print("\n" + "=" * 60)


def main():
    example_list_datasets()
    example_load_darts_dataset()
    example_load_synthetic_dataset()
    example_load_financial_dataset()
    example_load_energy_dataset()
    
    print("\n所有示例运行完成！")


if __name__ == '__main__':
    main()
