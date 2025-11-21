# 机器学习模型批量运行

# 这个脚本用于批量运行所有模型文件
import os
import subprocess
import warnings
import sys

warnings.filterwarnings('ignore')

def run_python_script(script_path, params=None):
    """运行指定的Python脚本文件"""
    print(f"正在运行: {script_path}")

    try:
        # 构建命令
        cmd = ['python', os.path.basename(script_path)]

        # 执行脚本，使用脚本所在目录作为工作目录
        script_dir = os.path.dirname(script_path)

        # 设置环境变量来传递参数
        env = os.environ.copy()
        if params:
            for key, value in params.items():
                env[key] = str(value)

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir, env=env)

        if result.returncode == 0:
            print(f"✓ 成功完成: {os.path.basename(script_path)}")
            if result.stdout:
                print("输出:", result.stdout[-500:])  # 只显示最后500个字符
            return True
        else:
            print(f"✗ 运行失败: {os.path.basename(script_path)}")
            if result.stderr:
                print("错误:", result.stderr[-500:])  # 只显示最后500个字符
            return False

    except Exception as e:
        print(f"✗ 运行失败: {os.path.basename(script_path)} - {str(e)}")
        return False

folder_path = '../data/final_data2'
all_items = os.listdir(folder_path)
datasets = [f for f in all_items if os.path.isfile(os.path.join(folder_path, f))]
datasets = [
 'filtered_patient3.csv',]
datasets

print("找到的数据集:")
for dataset in datasets:
    print(f"  - {dataset}")

def run_models(file_path):
    """运行指定模型文件的所有数据集"""

    # 直接使用Python文件路径
    py_file_path = file_path

    print(f"\n运行模型: {py_file_path}")
    print(f"找到 {len(datasets)} 个数据集")

    total_count = len(datasets)
    success_count = 0

    for dataset in datasets:
        print(f"\n处理数据集: {dataset}")
        print("-" * 50)

        if run_python_script(py_file_path, params={"dataset": dataset}):
            success_count += 1

    print(f"\n运行总结:")
    print(f"总数据集数: {total_count}")
    print(f"成功运行: {success_count}")
    print(f"失败数量: {total_count - success_count}")

    return success_count == total_count

# 运行各个模型
models = [
    ("./LR/LR.py", "LR模型"),
    ("./XGBoost/XGBoost.py", "XGBoost模型"),
    ("./LightGBM/LightGBM.py", "LightGBM模型"),
    ("./RF/RF.py", "RF模型"),
    ("./SVM/SVM.py", "SVM模型"),
    ("./HMM/HMM.py", "HMM模型"),
    ("./MLP/MLP.py", "MLP模型"),
    ("./LR_XGBoost/LR_XGBoost权重融合.py", "LR_XGBoost权重融合模型"),
    ("./LR_XGBoost/LR_XGBoost堆叠融合.py", "LR_XGBoost堆叠融合模型"),
    ("./LR_XGBoost/LR_XGBoost特征层融合.py", "LR_XGBoost特征层融合模型")
]

total_success = 0
total_models = len(models)

print("\n" + "="*60)
print("开始批量运行所有模型")
print("="*60)

for model_path, model_name in models:
    print(f"\n{'='*20} {model_name} {'='*20}")

    # 检查Python文件是否存在
    if not os.path.exists(model_path):
        print(f"⚠️  警告: {model_path} 不存在，跳过该模型")
        continue

    if run_models(model_path):
        total_success += 1
    else:
        print(f"⚠️  模型 {model_name} 运行失败")

print(f"\n{'='*60}")
print("总体运行总结:")
print(f"总模型数: {total_models}")
print(f"成功运行: {total_success}")
print(f"失败数量: {total_models - total_success}")
print(f"成功率: {total_success/total_models*100:.1f}%")
print("="*60)