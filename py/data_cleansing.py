import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

def main():
    ruzu_df = pd.read_excel('../data/入组表_20250814.xlsx')

    group_map = {
    0:"阴性组"
    ,1:"细菌性脑膜炎"
    ,2:"病毒性脑膜炎"
    ,3:"结核性脑膜炎"
    ,4:"真菌性脑膜炎"
    ,5:"肿瘤性脑膜炎"
    ,6:"自身免疫性脑膜炎"
    # ,7:"神经梅毒性脑膜炎" # 这一期不含有神经梅毒性脑膜炎
    ,8:"病毒性脑炎"
    }

    ruzu_df['组名'] = ruzu_df['组别'].apply(lambda x:group_map[x])
    ruzu_df.head()
    ruzu_df.to_csv('../data/ruzu_20250814.csv',index=False)
    ruzu_df = pd.read_csv('../data/ruzu_20250814.csv')
    ruzu_df.head()
    ruzu_df.shape
    ruzu_df['组别'].value_counts()
    ruzu_df.head()
    merged_df = pd.read_csv('../data/入组患者时间窗内检验清洗数据_20250814.csv')
    merged_df.head()
    merged_df.columns
    # 获取清洗成功的行
    cleaned_df = merged_df[merged_df['error'] == False].copy()
    # 删除全为空的列
    empty_cols = cleaned_df.columns[cleaned_df.isna().all()].tolist()
    print(f"全为空的列有{len(empty_cols)}列：{empty_cols}")
    # 删除无意义的id列，以及与清洗状态相关的列
    cleaned_df = cleaned_df[[col for col in cleaned_df.columns if col not in empty_cols + ['_id', 'error', 'errorMsg','cleanState']]].copy()
    # 删除只有1个取值的列（nunique=1的列）
    unique_cols = cleaned_df.columns[cleaned_df.nunique() == 1].tolist()
    cleaned_df = cleaned_df[[col for col in cleaned_df.columns if col not in unique_cols]]
    # 发现两个时间列完全相同：cleaned_df["resultDateTime"].equals(cleaned_df["reportDateTime"]), 去掉其中一列
    if cleaned_df["resultDateTime"].equals(cleaned_df["reportDateTime"]):
        print("两列完全相同，删除reportDateTime")
        cleaned_df = cleaned_df[[col for col in cleaned_df.columns if col not in ['reportDateTime']]]

    # 去掉original特征列
    original_col = ['originalLaboratoryName', 'originalAbnormalIndicator',
                    'originalSpecimen', 'originalResult', 'originalReferenceInterval', 'originalUnits']

    cleaned_df = cleaned_df[[col for col in cleaned_df.columns if col not in original_col]]

    print(f"清洗后数据集有{cleaned_df.shape[0]}行，{cleaned_df.shape[1]}列")
    # 保存清洗后的数据
    cleaned_df.to_csv("../data/all_cleaned_data测试.csv", index=False)


if __name__ == "__main__":
    main()