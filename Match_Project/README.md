Match_Project/
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   │   ├── train/
│   │   │   ├── angry/
│   │   │   ├── happy/
│   │   │   └── ...        
│   │   ├── val/
│   │   └── test/
├── src/
│   ├── model.py            # 模型定义
│   ├── train.py            # 训练脚本
│   ├── utils.py            # 工具函数
│   ├── config.py           # 配置文件
│   └── inference.py        # 推理示例
├── models/                 # 保存的模型
├── results/                # 训练结果/可视化
└── requirements.txt


numpy>=1.21.2
pandas>=1.3.4
scikit-learn>=0.24.2
matplotlib>=3.4.3
seaborn>=0.11.1
tqdm>=4.62.3