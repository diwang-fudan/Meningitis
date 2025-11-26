# 脑脊液疾病诊断机器学习项目 (Meningitis Diagnosis ML Project)

## 项目简介 (Project Overview)

本项目专注于使用机器学习算法进行脑脊液相关疾病的诊断和分类。通过分析脑脊液和血液检测数据，构建多种分类模型来辅助医生进行疾病诊断。

This project focuses on using machine learning algorithms for the diagnosis and classification of cerebrospinal fluid (CSF) related diseases. By analyzing CSF and blood test data, we build various classification models to assist physicians in disease diagnosis.

## 项目结构 (Project Structure)

```
├── .gitignore                    # Git忽略文件配置
├── README.md                     # 项目说明文档
└── ipynb/                        # 主要代码目录
    ├── LR/                       # 逻辑回归模型
    │   └── LR.py
    ├── LR_XGBoost/               # LR与XGBoost融合模型
    │   ├── LR_XGBoost堆叠融合.py
    │   ├── LR_XGBoost权重融合.py
    │   └── LR_XGBoost特征层融合.py
    ├── LightGBM/                 # LightGBM模型
    │   └── LightGBM.py
    ├── MLP/                      # 多层感知机模型
    │   └── MLP.py
    ├── RF/                       # 随机森林模型
    │   └── RF.py
    ├── SVM/                      # 支持向量机模型
    │   └── SVM.py
    ├── XGBoost/                  # XGBoost模型
    │   └── XGBoost.py
    ├── 清洗数据.py                # 数据预处理脚本
    ├── 特征构造.py                # 特征工程脚本
    ├── 模型运行.py                # 模型训练和评估主脚本
    ├── 结果分析.py                # 结果分析和可视化脚本
    └── 绘制可视化图.py             # 绘图工具脚本
```

## 主要算法模型 (Main Algorithm Models)

### 单一模型 (Single Models)
- **LR (Logistic Regression)** - 逻辑回归分类器
- **SVM (Support Vector Machine)** - 支持向量机分类器
- **RF (Random Forest)** - 随机森林集成学习
- **LightGBM** - 梯度提升决策树框架
- **MLP (Multi-Layer Perceptron)** - 多层感知机神经网络
- **XGBoost** - 极端梯度提升算法

### 融合模型 (Ensemble Models)
- **LR_XGBoost堆叠融合** - 基于堆叠策略的LR与XGBoost融合
- **LR_XGBoost权重融合** - 基于权重分配的LR与XGBoost融合
- **LR_XGBoost特征层融合** - 基于特征层面的LR与XGBoost融合

## 技术栈 (Technology Stack)

### 核心库 (Core Libraries)
- `pandas` - 数据处理和分析
- `numpy` - 数值计算
- `scikit-learn` - 机器学习基础库
- `matplotlib` & `seaborn` - 数据可视化
- `optuna` - 超参数优化

### 机器学习框架 (ML Frameworks)
- `xgboost` - XGBoost算法实现
- `lightgbm` - LightGBM算法实现

### 特性 (Features)
- **超参数优化**: 使用Optuna进行自动调优
- **模型解释**: 集成SHAP值分析
- **中文支持**: 配置matplotlib中文字体显示
- **结果可视化**: 生成ROC曲线、混淆矩阵等评估图表

## 数据处理流程 (Data Processing Pipeline)

1. **数据清洗** (`清洗数据.py`)
   - 缺失值处理
   - 异常值检测和处理
   - 数据类型转换

2. **特征构造** (`特征构造.py`)
   - 特征工程和变换
   - 特征选择和降维

3. **模型训练** (`模型运行.py`)
   - 多种算法的训练和调优
   - 交叉验证和模型选择

4. **结果分析** (`结果分析.py`, `绘制可视化图.py`)
   - 模型性能评估
   - 可视化分析
   - 结果对比和报告

## 模型评估指标 (Model Evaluation Metrics)

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数 (F1-Score)
- ROC曲线下面积 (AUC-ROC)
- 混淆矩阵 (Confusion Matrix)

## 使用说明 (Usage)

### 环境要求 (Requirements)
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm optuna shap
```

### 运行流程 (Run Pipeline)
1. 运行 `python 清洗数据.py` 进行数据预处理
2. 运行 `python 特征构造.py` 进行特征工程
3. 运行 `python 模型运行.py` 训练和评估模型
4. 运行 `python 结果分析.py` 和 `python 绘制可视化图.py` 分析结果

## 项目特色 (Project Highlights)

- 🏥 **医疗应用导向**: 专注于脑脊液疾病的临床辅助诊断
- 🤖 **多算法集成**: 涵盖6种主流机器学习算法及多种融合策略
- 📊 **完整流程**: 从数据预处理到结果可视化的端到端解决方案
- 🎯 **性能优化**: 使用Optuna进行超参数自动优化
- 🌟 **模型解释**: 集成SHAP值提供模型决策解释
- 🇨🇳 **本地化支持**: 完整的中文数据可视化支持

## 贡献指南 (Contributing)

欢迎提交Issue和Pull Request来改进本项目。

## 许可证 (License)

请查看LICENSE文件了解项目许可信息。

## 联系方式 (Contact)

如有问题或建议，请通过以下方式联系：
- 项目维护者: diwang-fudan
- 邮箱: d_wang@fudan.edu.cn

---

**注意**: 本项目仅用于研究和教学目的，不作为临床诊断的唯一依据。实际临床应用请结合专业医生判断。