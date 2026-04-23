"""TSF_Frame install script.

`pip install -e .` 会同时安装两个顶层包:
  - tsf_frame     (src-layout, 位于 src/tsf_frame)
  - configs       (项目级配置, 保持在仓库根目录 configs/ 下)

这样业务代码可以:
    from tsf_frame.business.hpf_adapter import HPFAdapter
    from configs.hpf import HPFConfig
"""
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]


# src-layout packages
src_packages = find_packages(where='src')
# project-level config packages (保留在根目录, 不搬进 src/)
config_packages = find_packages(where='.', include=['configs', 'configs.*'])

setup(
    name='tsf-frame',
    version='0.2.0',
    description='A general-purpose time-series forecasting framework with HPF business adapter',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='TSF_Frame Team',
    license='MIT',
    # 混合 package_dir：
    #   ''        → src-layout 下的 tsf_frame 及其子包从 src/ 找
    #   'configs' → 顶层 configs 包从仓库根目录 ./configs 找
    package_dir={
        '': 'src',
        'configs': 'configs',
    },
    packages=src_packages + config_packages,
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='time-series forecasting deep-learning machine-learning hpf housing-provident-fund',
)
