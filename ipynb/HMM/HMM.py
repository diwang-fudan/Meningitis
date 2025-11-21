import pandas as pd
import optuna
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib import font_manager
import math
import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)

# 注册中文字体到matplotlib
font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
mpl.font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
font_name = prop.get_name()
print("matplotlib 识别的字体名:", font_name)

plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# 提取当前数据集的数字
def extract_integers(s: str):
    """
    从字符串中提取所有整数，返回一个 int 列表
    """
    return list(map(int, re.findall(r"\d+", s)))

def label_encoding(filtered_df, data_path):
    columns_to_change1 = [item for item in filtered_df if item.endswith('_定性')]
    columns_to_change2 = [item for item in filtered_df if item.endswith('_描述性')]

    # 对每个object类型的列进行Label Encoding
    object_columns = columns_to_change1+columns_to_change2
    if 'filtered_patient1' in data_path or 'filtered_patient2' in data_path or 'filtered_patient5' in data_path:
        object_columns.remove('脑脊液_透明度_描述性')
        object_columns.remove('脑脊液_蛋白定性_定性')

    features_to_change = [ '血_抗双链DNA抗体_定性', '血_抗核抗体_定性', '血_抗胞浆型中性粒细胞抗体_定性', '血_抗核周型中性粒细胞抗体_定性', '血_梅毒非特异性抗体_定性', '血_梅毒螺旋体抗体_定性',
                        '血_人类免疫缺陷病毒抗体_定性', '血_不规则抗体_定性', '血_直接抗人球蛋白试验_定性', '血_结核感染T细胞检测_定性', '血_人类免疫缺陷病毒P24抗原_定性', '血_M蛋白_描述性']

    if 'filtered_patient2' in data_path or 'filtered_patient4' in data_path:
        for i in features_to_change:
            object_columns.remove(i)

    le_dict = {}
    for col in object_columns:
        le = LabelEncoder()
        # 处理空值：先用'unknown'填充，再进行编码
        filtered_df[col] = filtered_df[col].fillna('unknown')
        filtered_df[col] = le.fit_transform(filtered_df[col].astype(str))  # 确保没有NaN，否则会报错
        le_dict[col] = le  # 保存编码器，后续可能用于推理

    return filtered_df

# 替换特征名中的 [，]
def clean_feature_names_simple(feature_names):
    cleaned_names = []
    for name in feature_names:
        cleaned_name = name.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
        cleaned_names.append(cleaned_name)
    return cleaned_names


    # filtered_df.columns = clean_feature_names_simple(filtered_df.columns)
    # print("清洗后的列名:", filtered_df.columns.tolist())

def train(X_train, y_train, X_val, y_val):
    # 定义 Optuna 目标函数
    def objective(trial):
        # 建议超参数范围
        n_components = trial.suggest_int('n_components', 2, 10)
        covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
        n_iter = trial.suggest_int('n_iter', 50, 200)

        # 数据清理：处理NaN值
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_val_imputed = imputer.transform(X_val)

        # 数据标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_val_scaled = scaler.transform(X_val_imputed)

        # PCA降维以减少特征数量
        pca = PCA(n_components=min(10, X_train_scaled.shape[1]))
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        # 创建HMM模型
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=42
        )

        try:
            # 训练模型
            model.fit(X_train_pca)

            # 预测验证集状态
            hidden_states = model.predict(X_val_pca)

            # 根据隐状态预测类别
            state_0_samples = X_train_pca[y_train == 0]
            state_1_samples = X_train_pca[y_train == 1]

            if len(state_0_samples) > 0 and len(state_1_samples) > 0:
                state_0_pred = model.predict(state_0_samples)
                state_1_pred = model.predict(state_1_samples)

                # 确定每个状态对应的类别
                from collections import Counter
                state_0_counter = Counter(state_0_pred)
                state_1_counter = Counter(state_1_pred)

                # 为每个状态分配最可能的类别
                state_to_class = {}
                for state in range(n_components):
                    state_0_prob = state_0_counter[state] / len(state_0_samples) if state in state_0_counter else 0
                    state_1_prob = state_1_counter[state] / len(state_1_samples) if state in state_1_counter else 0
                    state_to_class[state] = 0 if state_0_prob > state_1_prob else 1

                # 预测验证集
                y_val_pred = np.array([state_to_class[state] for state in hidden_states])

                # 计算AUC
                if len(np.unique(y_val_pred)) > 1:
                    # 使用状态概率作为预测概率
                    state_probs = model.predict_proba(X_val_pca)
                    y_val_pred_proba = np.array([state_probs[i, state] for i, state in enumerate(hidden_states)])

                    # 调整概率使其与预测一致
                    for i in range(len(y_val_pred)):
                        if y_val_pred[i] == 0:
                            y_val_pred_proba[i] = 1 - y_val_pred_proba[i]

                    auc_score = roc_auc_score(y_val, y_val_pred_proba)
                else:
                    auc_score = 0.5
            else:
                auc_score = 0.5

        except Exception as e:
            print(f"训练错误: {e}")
            auc_score = 0.5

        # 由于 Optuna 默认最小化目标函数，我们返回 1 - AUC 来最大化 AUC
        return 1.0 - auc_score

    # 创建 Optuna 研究
    study = optuna.create_study(
        direction='minimize',  # 最小化目标函数 (1 - AUC)
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)  # 10轮后开始剪枝
    )

    # 运行优化
    print("开始超参数优化...")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    # 输出最佳结果
    print("\n优化完成!")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Best AUC: {1 - best_trial.value:.4f}")  # 转换回 AUC
    print("  Best params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 使用最佳参数重新训练最终模型
    print("\n使用最佳参数训练最终模型...")
    best_params = best_trial.params

    # 数据清理和标准化
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)

    # PCA降维
    pca = PCA(n_components=min(10, X_train_scaled.shape[1]))
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)

    final_model = hmm.GaussianHMM(
        n_components=best_params['n_components'],
        covariance_type=best_params['covariance_type'],
        n_iter=best_params['n_iter'],
        random_state=42
    )

    final_model.fit(X_train_pca)

    # 创建模型字典包含所有预处理步骤
    model_dict = {
        'hmm_model': final_model,
        'scaler': scaler,
        'pca': pca,
        'imputer': imputer,
        'params': best_params
    }

    return model_dict

# 在各个数据集上评估最终模型
def evaluate_model(model_dict, X, y, dataset_name):
    # 获取模型和预处理步骤
    hmm_model = model_dict['hmm_model']
    scaler = model_dict['scaler']
    pca = model_dict['pca']
    imputer = model_dict['imputer']

    # 数据预处理
    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)
    X_pca = pca.transform(X_scaled)

    # 预测隐状态
    hidden_states = hmm_model.predict(X_pca)

    # 根据训练集确定状态到类别的映射
    y_pred = hidden_states % 2  # 简单映射

    # 计算概率
    state_probs = hmm_model.predict_proba(X_pca)
    y_pred_proba = np.array([state_probs[i, state] for i, state in enumerate(hidden_states)])

    # 调整概率
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            y_pred_proba[i] = 1 - y_pred_proba[i]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y_pred)) > 1 else 0.5

    print(f"\n{dataset_name} 集评估结果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def draw_result(y_val, y_test, val_results, test_results, train_results, key, current_dataset):

    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))

    # 计算各数据集的ROC曲线

    fpr_val, tpr_val, thresholds = roc_curve(y_val, val_results['y_pred_proba'])
    fpr_test, tpr_test, thresholds = roc_curve(y_test, test_results['y_pred_proba'])
    roc_data_path = f"../../data/ROC/HMM/{key}"
    os.makedirs(roc_data_path, exist_ok=True)
    
    pd.DataFrame({
        "fpr": fpr_test,
        "tpr": tpr_test,
        "threshold": thresholds
    }).to_csv(f"{roc_data_path}/Dataset{current_dataset}_roc_curve.csv", index=False)

    # 绘制ROC曲线
    plt.plot(fpr_val, tpr_val, color='blue', lw=2,
            label=f'验证集 AUC = {val_results["auc"]:.4f}')
    plt.plot(fpr_test, tpr_test, color='red', lw=2,
            label=f'测试集 AUC = {test_results["auc"]:.4f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{key} - HMM ROC曲线', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_pic_path = f"../../images/HMM/ROC_AUC/Dataset{current_dataset}"
    os.makedirs(roc_pic_path, exist_ok=True)
    plt.savefig(f"{roc_pic_path}/{key}ROC-AUC.png", dpi=300, bbox_inches='tight')  # 保存为PNG
    # plt.show()

    model = key
    data_to_append = {'val_accuracy':round(val_results['accuracy'],4),'val_precision':round(val_results['precision'],4),'val_recall':round(val_results['recall'],4),'val_f1':round(val_results['f1'],4),'roc_auc_val':round(val_results['auc'],4),
                    'test_accuracy':round(test_results['accuracy'],4),'test_precision':round(test_results['precision'],4),'test_recall':round(test_results['recall'],4),'test_f1':round(test_results['f1'],4),'roc_auc_test':round(test_results['auc'],4),
                    'train_accuracy':round(train_results['accuracy'],4),'trian_precision':round(train_results['precision'],4),'train_recall':round(train_results['recall'],4),'train_f1':round(train_results['f1'],4),'roc_auc_train':round(train_results['auc'],4),}
    # JSON 文件路径
    json_file = "../../data/model_data"
    os.makedirs(json_file, exist_ok=True)

    # 如果文件存在，先读取已有数组，否则初始化为空列表
    if os.path.exists(f"{json_file}/HMM_data.json"):
        with open(f"{json_file}/HMM_data.json", "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if f'HMM_Dataset{current_dataset}' not in existing_data:
                    existing_data[f'HMM_Dataset{current_dataset}'] = {}
            except json.JSONDecodeError:
                existing_data = {f'HMM_Dataset{current_dataset}':{}}
    else:
        existing_data = {f'HMM_Dataset{current_dataset}':{}}

    # 追加新的数据
    existing_data[f'HMM_Dataset{current_dataset}'][model] = data_to_append

    # 写回 JSON 文件
    with open(f"{json_file}/HMM_data.json", "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


def delete_same_col(df):
    same_cols = df.nunique() == 1
    d = dict(same_cols)
    none_feature = []
    for keys, items in d.items():
        if items == True:
            none_feature.append(keys)
    return df.drop(none_feature, axis=1)

def run_model(key, item, selected_df, current_dataset):
    # 筛选需要的组别
    selected_df = selected_df[selected_df['分组'].isin(item[0])]

    # 删除数值一样的列
    selected_df = delete_same_col(selected_df)

    # 选择特征 将数据正例和反例平均分配
    filter_1 = selected_df[selected_df["分组"].isin(item[1])]
    filter_2 = selected_df[selected_df["分组"].isin(item[2])]

    X_1 = filter_1[list(filter_1.columns)[:-2]]
    X_2 = filter_2[list(filter_2.columns)[:-2]]

    # 标签（分组，1 类与其他类）
    y_1 = (filter_1['分组'].isin(item[1])).astype(int) # 如果分组为 0，标签为 1，否则为 0
    y_2 = (filter_2['分组'].isin(item[1])).astype(int)

    # 拆分数据集为训练集（60%）、验证集（20%）和测试集（20%）
    X_train_1, X_temp_1, y_train_1, y_temp_1 = train_test_split(X_1, y_1, test_size=0.4, random_state=42)
    X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(X_temp_1, y_temp_1, test_size=0.5, random_state=42)

    X_train_2, X_temp_2, y_train_2, y_temp_2 = train_test_split(X_2, y_2, test_size=0.4, random_state=42)
    X_val_2, X_test_2, y_val_2, y_test_2 = train_test_split(X_temp_2, y_temp_2, test_size=0.5, random_state=42)

    # 将正例与反例拼接
    X_train = pd.concat([X_train_1,X_train_2])
    y_train = pd.concat([y_train_1,y_train_2])

    X_val = pd.concat([X_val_1,X_val_2])
    y_val = pd.concat([y_val_1,y_val_2])

    X_test = pd.concat([X_test_1,X_test_2])
    y_test = pd.concat([y_test_1,y_test_2])

    # 得到最优模型
    model_dict = train(X_train, y_train, X_val, y_val)

    # 评估验证集和测试集
    val_results = evaluate_model(model_dict, X_val, y_val, "验证")
    test_results = evaluate_model(model_dict, X_test, y_test, "测试")
    train_results = evaluate_model(model_dict, X_train, y_train, "训练")

    # 绘制ROC——AUC图 并存储数据
    draw_result(y_val, y_test, val_results, test_results, train_results, key, current_dataset)

def main(filtered_df, data_path):

    filtered_df = label_encoding(filtered_df, data_path)
    # 对 DataFrame 的列名进行清洗
    filtered_df.columns = clean_feature_names_simple(filtered_df.columns)

    # 数据预处理：将 'gender' 列转换为数值型
    filtered_df['gender'] = filtered_df['gender'].map({'女': 0, '男': 1})

    screen_data = {
        '阴性组vs其他':[[0,1,2,3,4,5,6], [0], [1,2,3,4,5,6]],
        '感染性脑膜炎vs非感染性脑膜炎': [[1,2,3,4,5,6], [1,2,3,4], [5,6]],
        '细菌性脑膜炎vs其他感染性脑膜炎': [[1,2,3,4], [1],  [2,3,4]],
        '病毒性脑膜炎vs其他感染性脑膜炎': [[1,2,3,4], [2], [1,3,4]],
        '结核性脑膜炎vs其他感染性脑膜炎': [[1,2,3,4], [3], [1,2,4]],
        '真菌性脑膜炎vs其他感染性脑膜炎': [[1,2,3,4], [4], [1,2,3]],
        '肿瘤性脑膜炎vs其他': [[0,1,2,3,4,5,6], [5], [0,1,2,3,4,6]],
        '自身免疫性脑膜炎vs其他': [[0,1,2,3,4,5,6], [6], [0,1,2,3,4,5]],
        '病毒性脑膜炎vs病毒性脑炎': [[2,8], [2], [8]]
        }
    
    if extract_integers(data_path)[-1] == 2:
        selected_features = {
            '阴性组vs其他': filtered_df[['脑脊液_有核细胞计数_定量', '脑脊液_葡萄糖_定量','血_嗜酸性粒细胞百分比_定量','血_葡萄糖空腹_定量','血_结核感染T细胞检测_定性','血_乙型肝炎病毒核心抗体.IgM_定量', '分组', '组名']],
            '感染性脑膜炎vs非感染性脑膜炎': filtered_df[['脑脊液_单个核细胞百分比_定量','脑脊液_葡萄糖_定量', '脑脊液_激活淋巴细胞_定量','脑脊液_多个核细胞百分比_定量', '分组', '组名']],
            '细菌性脑膜炎vs其他感染性脑膜炎': filtered_df[['脑脊液_多个核细胞百分比_定量','血_C反应蛋白_定量','脑脊液_有核细胞计数_定量','血_抗凝血酶III_定量','血_血小板分布宽度_定量','血_单核细胞计数_定量', '血_细胞.CD3+CD8+百分比_定量','分组', '组名']],
            '病毒性脑膜炎vs其他感染性脑膜炎': filtered_df[['脑脊液_葡萄糖_定量','脑脊液_中性粒细胞百分比_定量','血_肌酸激酶_定量', '血_肾小球滤过率_定量', '血_免疫球蛋白E_定量','血_细胞.CD3-CD16+CD56+百分比_定量','脑脊液_多个核细胞百分比_定量','脑脊液_蛋白定量_定量','分组', '组名']],
            '结核性脑膜炎vs其他感染性脑膜炎': filtered_df[['血_结核感染T细胞检测_定性','脑脊液_氯_定量','血_结核感染T细胞测试管_定量','脑脊液_葡萄糖_定量','血_淋巴细胞计数_定量','血_血小板分布宽度_定量', '血_铁蛋白_定量', '血_游离T4_定量', '分组', '组名']],
            '真菌性脑膜炎vs其他感染性脑膜炎': filtered_df[['血_细胞.CD3-CD19+百分比_定量','血_不规则抗体_定性','血_梅毒非特异性抗体_定性', '脑脊液_有核细胞计数_定量', '分组', '组名']],
            '肿瘤性脑膜炎vs其他': filtered_df[['脑脊液_蛋白定量_定量', '脑脊液_葡萄糖_定量', '脑脊液_多个核细胞百分比_定量', '脑脊液_淋巴细胞百分比_定量', 'age','分组', '组名']],
            '自身免疫性脑膜炎vs其他': filtered_df[['血_丙型肝炎病毒抗体_定量','脑脊液_葡萄糖_定量','血_结核感染T细胞测试管_定量','血_铁蛋白_定量','脑脊液_蛋白定量_定量','血_D-二聚体_定量','血_乙型肝炎病毒核心抗体.IgM_定量','age','分组', '组名']],
            '病毒性脑膜炎vs病毒性脑炎': filtered_df[['脑脊液_有核细胞计数_定量','血_降钙素原_定量','血_同型半胱氨酸_定量','血_细胞.CD3+CD4+百分比_定量','血_血小板分布宽度_定量', '血_细胞.CD3+CD8+百分比_定量', '血_前白蛋白_定量', '血_铁蛋白_定量', '分组', '组名']]
        }

    current_dataset = extract_integers(data_path)[-1]
    for key, item in screen_data.items():
        # 训练模型 - 数据集1-8

        filtered_df2 = filtered_df.copy()
        run_model(key, item, filtered_df2, current_dataset)

        # 训练模型 - 精选特征数据集
        if current_dataset == 2:
            run_model(key, item, selected_features[key], 9)

if __name__ == "__main__":
    import os
    # 从环境变量获取dataset，如果没有则使用默认值
    dataset = os.environ.get('dataset', 'filtered_patient3.csv')
    data_path = f'../../data/final_data2/{dataset}'
    # data_path = '../../data/final_data2/filtered_patient3.csv'
    filtered_df = pd.read_csv(data_path)
    main(filtered_df, data_path)