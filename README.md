# Meningitis Diagnosis Machine Learning Project

## Project Overview

This project focuses on using machine learning algorithms for the diagnosis and classification of cerebrospinal fluid (CSF) related diseases. By analyzing CSF and blood test data, we build various classification models to assist physicians in disease diagnosis.

## Project Structure

```
├── .gitignore                         # Git ignore configuration
├── README.md                          # Project documentation
└── py/                                # Main source code directory
    ├── models/                        # Machine learning models
    │   ├── LR.py                      # Logistic Regression model
    │   ├── LR_XGBoost_feature_level.py    # Feature-level fusion of LR & XGBoost
    │   ├── LR_XGBoost_stack.py            # Stacking fusion of LR & XGBoost
    │   ├── LR_XGBoost_weighted.py         # Weighted fusion of LR & XGBoost
    │   ├── LightGBM.py               # LightGBM model
    │   ├── MLP.py                    # Multi-Layer Perceptron model
    │   ├── RF.py                     # Random Forest model
    │   ├── SVM.py                    # Support Vector Machine model
    │   └── XGBoost.py                # XGBoost model
    ├── analyse_result.py              # Results analysis and evaluation script
    ├── data_cleansing.py             # Data preprocessing script
    ├── feature_construction.py       # Feature engineering script
    ├── run_models.py                 # Main model training and evaluation script
    └── visualization.py             # Data visualization and plotting script
```

## Main Algorithm Models

### Single Models
- **LR (Logistic Regression)** - Linear classification model for binary/multi-class problems
- **SVM (Support Vector Machine)** - Kernel-based classification with maximum margin
- **RF (Random Forest)** - Ensemble learning with multiple decision trees
- **LightGBM** - Gradient boosting framework with leaf-wise growth strategy
- **MLP (Multi-Layer Perceptron)** - Feedforward neural network with multiple hidden layers
- **XGBoost** - Extreme gradient boosting with regularized objectives

### Ensemble Models
- **LR_XGBoost_feature_level** - Feature-level fusion combining LR and XGBoost predictions
- **LR_XGBoost_stack** - Stacking ensemble using LR as meta-learner on XGBoost outputs
- **LR_XGBoost_weighted** - Weighted ensemble combining LR and XGBoost predictions

## Technology Stack

### Core Libraries
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing and array operations
- `scikit-learn` - Machine learning fundamentals and utilities
- `matplotlib` & `seaborn` - Data visualization and plotting
- `optuna` - Hyperparameter optimization framework
- `shap` - Model interpretation and explainability

### Machine Learning Frameworks
- `xgboost` - XGBoost algorithm implementation
- `lightgbm` - LightGBM algorithm implementation

### Key Features
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Model Interpretability**: SHAP values for model explanation
- **Multi-language Support**: Chinese font configuration for matplotlib
- **Comprehensive Visualization**: ROC curves, confusion matrices, and performance metrics

## Data Processing Pipeline

### 1. Data Cleansing (`data_cleansing.py`)
- Missing value handling and imputation
- Outlier detection and treatment
- Data type conversion and validation
- Data quality assessment

### 2. Feature Engineering (`feature_construction.py`)
- Feature transformation and generation
- Feature selection and dimensionality reduction
- Wide format data construction from clinical measurements
- Temporal feature extraction from longitudinal data

### 3. Model Training (`run_models.py`)
- Multi-algorithm training pipeline
- Cross-validation and hyperparameter optimization
- Model comparison and selection
- Performance evaluation and benchmarking

### 4. Results Analysis (`analyse_result.py`, `visualization.py`)
- Comprehensive model performance evaluation
- Statistical analysis and significance testing
- Visualization of results and model interpretations
- Report generation and comparative analysis

## Model Evaluation Metrics

- **Accuracy** - Overall classification correctness
- **Precision** - Positive predictive value
- **Recall** - Sensitivity or true positive rate
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under the receiver operating characteristic curve
- **Confusion Matrix** - Detailed classification performance breakdown

## Usage

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install xgboost lightgbm optuna shap
```

### Running the Pipeline

1. **Data Preprocessing**
   ```bash
   python py/data_cleansing.py
   ```

2. **Feature Engineering**
   ```bash
   python py/feature_construction.py
   ```

3. **Model Training and Evaluation**
   ```bash
   python py/run_models.py
   ```

4. **Results Analysis and Visualization**
   ```bash
   python py/analyse_result.py
   python py/visualization.py
   ```

## Project Highlights

- 🏥 **Medical Application Focus**: Specialized for CSF disease diagnosis and clinical decision support
- 🤖 **Multi-Algorithm Integration**: Comprehensive coverage of 6 mainstream ML algorithms and ensemble strategies
- 📊 **End-to-End Solution**: Complete pipeline from data preprocessing to result visualization
- 🎯 **Performance Optimization**: Automated hyperparameter tuning with Optuna
- 🌟 **Model Interpretability**: Integrated SHAP analysis for model decision explanation
- 🔬 **Clinical Data Handling**: Specialized processing for medical laboratory data and patient records

## Dataset Information

The project processes clinical laboratory data including:
- **Cerebrospinal Fluid (CSF)** measurements
- **Blood test results** and biochemical markers
- **Patient demographic information**
- **Temporal measurement data** for longitudinal analysis

## Model Performance

The models are evaluated using comprehensive metrics and compared across:
- Different disease categories
- Various feature subsets
- Multiple performance thresholds
- Cross-validation scenarios

## Contributing

We welcome contributions through Issues and Pull Requests. Please ensure:
- Code follows Python best practices
- Documentation is updated
- Tests are included for new features
- Clinical considerations are respected

## License

Please refer to the LICENSE file for detailed licensing information.

## Contact

For questions, suggestions, or collaborations:
- **Maintainer**: diwang-fudan
- **Email**: d_wang@fudan.edu.cn

---

**Disclaimer**: This project is intended for research and educational purposes only and should not be used as the sole basis for clinical diagnosis. Always consult qualified medical professionals for clinical decision-making.