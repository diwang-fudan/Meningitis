import pandas as pd
import numpy as np
from datetime import datetime
import math
import pandas as pd
import matplotlib.pyplot as plt

def build_wide_data(df):
    # 传入减少原始列的新csv文件
    # 只考虑组合定量、定性的特征

    # 步骤1：确保resultDateTime列的格式一致，不修改列名，只取日期部分
    df['resultDate'] = pd.to_datetime(df['resultDateTime']).dt.date  # 保留原始的resultDateTime列

    # 步骤2：定义一个函数来生成新的列名
    def create_column_name(row):
        """
        生成新列名，根据标准样本和实验室名称以及结果类型
        """
        standard_specimen = row['standardSpecimen']
        laboratory_name = row['laboratoryName']
        result_type = row['standardResultType']

        # 拼接出新列名
        if pd.isna(result_type):
            return None
        if result_type == 'QUALITATIVE':
            return f"{standard_specimen}_{laboratory_name}_定性"
        if result_type == 'QUANTIFY':
            return f"{standard_specimen}_{laboratory_name}_定量"
        if result_type == 'DESCRIPTIVE':
            return f"{standard_specimen}_{laboratory_name}_描述性"
        return None

    # 步骤3：为每一行创建新的列名
    df['new_column_name'] = df.apply(create_column_name, axis=1)

    # 步骤4：根据patientId和resultDate进行分组，聚合数据
    # 用一个字典来保存每个患者每个日期的所有结果
    aggregated_data = {}

    for _, row in df.iterrows():
        patient_id = row['patientId']
        result_date = row['resultDate']  # 保持一致的日期格式
        age = row['age']  # 获取患者的年龄
        gender = row['gender']  # 获取患者的性别
        new_col_name = row['new_column_name']
        result_type = row['standardResultType']
        value_1 = row['standardQuantitativeResult']
        value_2 = row['standardResultNorm']
        # 如果有新列名，才添加
        if new_col_name:
            # 如果还没有添加该患者该日期的数据，初始化
            if (patient_id, result_date) not in aggregated_data:
                aggregated_data[(patient_id, result_date)] = {
                    'patientId': patient_id,
                    'resultDateTime': result_date,
                    'age': age,
                    'gender': gender
                }

            # 为该患者该日期添加新的列
            if result_type  == 'QUANTIFY':
                aggregated_data[(patient_id, result_date)][new_col_name] = value_2
            elif result_type in ['DESCRIPTIVE','QUALITATIVE']:
                aggregated_data[(patient_id, result_date)][new_col_name] = value_2

    # 步骤5：将聚合的数据转换为DataFrame
    # 创建最终的DataFrame
    final_data = []
    for (patient_id, result_date), values in aggregated_data.items():
        final_data.append(values)

    # 转换成DataFrame
    final_df = pd.DataFrame(final_data)

    # 步骤6：保存为CSV文件
    # final_df.to_csv("/home/ruiqitang/naojiye/data/wide_data.csv", index=False)
    final_df.to_csv("../data/wide_data.csv", index=False)
    return final_df

def data_counts(final_df,ruzu_df):
    
    # ruzu_df['分组'].value_counts()
    ruzu_df = ruzu_df.rename(columns={'PID':'patientId'})
    ruzu_df = ruzu_df.rename(columns={'组别':'分组'})
    #统计日期数

    group_data1 = dict(final_df.groupby(['patientId'])['resultDateTime'].nunique())
    group_data2 = {}
    for key in group_data1:
        group_data2[str(key)] = group_data1[key]


    def add_date_count(row):
        a = row['patientId']

        if a not in group_data2.keys():
            print(a)
            return None

        return int(group_data2[a])


    ruzu_df['日期数'] = ruzu_df.apply(add_date_count, axis=1)

    return ruzu_df

def max_min_date(final_df,ruzu_df):
    # 患者来就诊的最小和最大日期
    all_dates_dict = {}

    def all_dates(row):

        if str(row['patientId']) not in all_dates_dict:
            all_dates_dict[str(row['patientId'])] = [row['resultDateTime']]

        else:
            all_dates_dict[str(row['patientId'])].append(row['resultDateTime'])

    final_df.apply(all_dates, axis=1)
    # 患者来就诊的最小日期
    def mini_date(row):
        a = row['patientId']

        if a not in all_dates_dict.keys():
            print(a)
            return None

        return sorted(all_dates_dict[a])[0]

    # 患者来就诊的最大日期
    def max_date(row):
        a = row['patientId']

        if a not in all_dates_dict.keys():
            print(a)
            return None

        return sorted(all_dates_dict[a])[-1]


    ruzu_df['最小日期'] = ruzu_df.apply(mini_date, axis=1)
    ruzu_df['最大日期'] = ruzu_df.apply(max_date, axis=1)
    return final_df,ruzu_df

def between_days(ruzu_df):
# 患者入组日期距离最小日期的天数
    def days_count(row):
        if row['最小日期'] is None :
            return None
        d1 = datetime.strptime(row['入组时间'], "%Y-%m-%d")
        d2 = datetime.strptime(str(row['最小日期']), "%Y-%m-%d")
        return (d1 - d2).days

    ruzu_df['日期差'] = ruzu_df.apply(days_count, axis=1)
    # ruzu_df['日期差'] = ruzu_df['日期差'].astype(int)

    return ruzu_df

def get_all_features(final_df):
    # 大特征下面的小特征类
    feature = list(final_df.columns)[4:]

    feature_dict = {}

    for i in feature:
        item = i.split('_')
        if item[0] not in feature_dict.keys():
            feature_dict[item[0]] = [item[1]]
        else:
            feature_dict[item[0]].append(item[1])

    feature_df = pd.DataFrame.from_dict(feature_dict, orient='index')

    feature_df.to_excel("../data/features.xlsx")
    return feature_df, feature_dict

def feature_missing_rate(final_df,feature_dict):
    #特征缺失率
    none_count = final_df.isna().sum().to_dict()

    data_length = final_df.shape[0]
    data_length
    for key in none_count.keys():
        none_count[key] = none_count[key]/data_length
        # if none_count[key] < 0.5:
        #     print(1)

    # 删除没有必要的信息项
    none_count.pop('patientId')
    none_count.pop('resultDateTime')
    none_count.pop('age')
    none_count.pop('gender')
    none_count
    none_count_df = pd.DataFrame.from_dict(none_count, columns=['缺失率'],orient='index')
    none_count_df['缺失率'] = none_count_df['缺失率'].apply(lambda x: '{:.2%}'.format(x))

    none_count_df.to_excel("../data/none_rate.xlsx")

    # 关键特征缺失率
    feature_dict.keys()

    blood_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "血":
            blood_dict[key] = "{:.2%}".format(none_count[key])

    naojiye_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "脑脊液":
            naojiye_dict[key] = "{:.2%}".format(none_count[key])

    niao_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "尿":
            niao_dict[key] = "{:.2%}".format(none_count[key])

    tan_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "痰":
            tan_dict[key] = "{:.2%}".format(none_count[key])

    fenbian_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "粪便":
            fenbian_dict[key] = "{:.2%}".format(none_count[key])

    bi_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "鼻/咽拭子":
            bi_dict[key] = "{:.2%}".format(none_count[key])


    gusui_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "骨髓":
            gusui_dict[key] = "{:.2%}".format(none_count[key])

    dongmaixue_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "动脉血":
            dongmaixue_dict[key] = "{:.2%}".format(none_count[key])


    nan_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "nan":
            nan_dict[key] = "{:.2%}".format(none_count[key])

    weiye_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "胃液":
            weiye_dict[key] = "{:.2%}".format(none_count[key])

    gangshizi_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "肛拭子":
            gangshizi_dict[key] = "{:.2%}".format(none_count[key])


    nongzhongye_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "脓肿液":
            nongzhongye_dict[key] = "{:.2%}".format(none_count[key])

    chuanciye_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "穿刺液":
            chuanciye_dict[key] = "{:.2%}".format(none_count[key])

    zhishua_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "支刷":
            zhishua_dict[key] = "{:.2%}".format(none_count[key])

    daoguan_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "导管":
            daoguan_dict[key] = "{:.2%}".format(none_count[key])

    zhiqiguan_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "支气管肺泡灌洗液":
            zhiqiguan_dict[key] = "{:.2%}".format(none_count[key])

    tiye_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "体液":
            tiye_dict[key] = "{:.2%}".format(none_count[key])

    outuwu_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "呕吐物":
            outuwu_dict[key] = "{:.2%}".format(none_count[key])

    pixie_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "皮屑":
            pixie_dict[key] = "{:.2%}".format(none_count[key])

    pifu_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "皮肤":
            pifu_dict[key] = "{:.2%}".format(none_count[key])

    danzhi_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "胆汁":
            danzhi_dict[key] = "{:.2%}".format(none_count[key])

    erdao_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "耳道分泌物":
            erdao_dict[key] = "{:.2%}".format(none_count[key])

    xiongqiangye_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "胸腔液":
            xiongqiangye_dict[key] = "{:.2%}".format(none_count[key])

    yinliuye_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "引流液":
            yinliuye_dict[key] = "{:.2%}".format(none_count[key])

    yanbu_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "眼部分泌物":
            yanbu_dict[key] = "{:.2%}".format(none_count[key])

    fenmiwu_dict = {}
    for key in none_count.keys():
        if key.split("_")[0] == "分泌物":
            fenmiwu_dict[key] = "{:.2%}".format(none_count[key])

    all_outer_features = [blood_dict,naojiye_dict,fenbian_dict,niao_dict,bi_dict,gusui_dict,dongmaixue_dict,nan_dict,tan_dict,weiye_dict,gangshizi_dict,nongzhongye_dict,chuanciye_dict,zhishua_dict,
                        daoguan_dict,zhiqiguan_dict,tiye_dict,outuwu_dict,pixie_dict,pifu_dict,danzhi_dict,erdao_dict,xiongqiangye_dict,yinliuye_dict,yanbu_dict,fenmiwu_dict]

    excel_name = list(feature_dict.keys())
    excel_name[4] = "鼻-咽拭子"
    excel_count = 0

    for item in all_outer_features:

        pd.DataFrame.from_dict(item, columns=['缺失率'],orient='index').to_excel("../data/"+excel_name[excel_count]+".xlsx")
        excel_count += 1

def check_key_features(final_df,feature_dict,ruzu_df):
    # 患者是否做了关键检测

    all_features = list(final_df.columns)[4:]

    inter_features = {}
    outer_features = list(feature_dict.keys())

    for i in outer_features:
        inter_features[i] = []


    for i in all_features:
        for a in outer_features:
            if i.split("_")[0] == a:
                inter_features[a].append(i)
    def check_features(row):

        for i in inter_features[feature_name]:
            if (isinstance(row[i],str) and row[i] is not None) or (isinstance(row[i],float) and not math.isnan(row[i])):
                return 1
        return 0


    for i in outer_features:
        feature_name = i
        final_df[feature_name] = final_df.apply(check_features, axis=1)

    guanjianjiance_final_df = final_df[['patientId','resultDateTime','age','gender']+outer_features]
    guanjianjiance_final_df = guanjianjiance_final_df.rename(columns={'resultDateTime': '报告时间'})
    guanjianjiance_final_df.head()
    guanjianjiance_final_df.to_excel("../data/是否做过关键检测.xlsx")

    selected_ruzu_df = ruzu_df[['patientId','入组时间','分组','组名']]
    guanjianjiance_final_df['报告时间'] = guanjianjiance_final_df['报告时间'].astype(str)
    guanjianjiance_final_df['patientId'] = guanjianjiance_final_df['patientId'].astype(str)
    result_df = pd.merge(guanjianjiance_final_df, selected_ruzu_df, on='patientId')

    sorted_result_df = result_df.sort_values(by='报告时间')
    sorted_result_df = sorted_result_df.groupby('patientId')
    sorted_result_df = sorted_result_df.apply(lambda x: x).reset_index(drop=True)
    sorted_result_df = sorted_result_df.reindex(columns=['patientId', '报告时间', '入组时间', 'age', 'gender', '分组', '组名']+ outer_features)
    # 加入所有需要的数据
    sorted_result_df.to_excel("../data/是否做过关键检测.xlsx")

def inspection_count():
    # 增加每位患者每日检查项目数和检查项目

    final_df = pd.read_csv("../data/wide_data.csv")
    check_project = final_df.columns[4:]

    def exam_project(row):
        all_exam = []
        for i in check_project:
            if not(isinstance(row[i],float) and np.isnan(row[i])):
                all_exam.append(i)
        return all_exam

    final_df['检查项目'] = final_df.apply(exam_project,axis=1)

    final_df['检查项目数量'] = final_df.apply(lambda x:len(x['检查项目']),axis=1)
    return final_df

def check_data(final_df):
    sorted_result_df = pd.read_excel("../data/是否做过关键检测.xlsx")
    sorted_result_df['报告时间'] = sorted_result_df['报告时间'].astype(str)
    sorted_result_df
    check_df = pd.merge(sorted_result_df,final_df[['patientId','resultDateTime','检查项目数量','检查项目']],left_on=['patientId','报告时间'], right_on=['patientId','resultDateTime'])
    check_df.drop(['resultDateTime','Unnamed: 0'], axis=1, inplace=True)
    check_df = check_df.reindex(columns=['patientId', '报告时间', '入组时间', 'age', 'gender', '分组', '组名', '检查项目数量', '检查项目', '血', '脑脊液',
        '粪便', '尿', '鼻/咽拭子', '骨髓', '动脉血', 'nan', '痰', '胃液', '肛拭子', '脓肿液', '穿刺液',
        '支刷', '导管', '支气管肺泡灌洗液', '体液', '呕吐物', '皮屑', '皮肤', '胆汁', '耳道分泌物', '胸腔液',
        '引流液', '眼部分泌物', '分泌物'])
    def check_range(row):
        if row['检查项目数量'] == 1:
            return 1
        elif row['检查项目数量'] == 2:
            return 2
        elif row['检查项目数量'] == 3:
            return 3
        elif row['检查项目数量'] > 3 and row['检查项目数量'] <=10:
            return 4
        elif row['检查项目数量'] > 10 and row['检查项目数量'] <=50:
            return 10
        elif row['检查项目数量'] > 50 and row['检查项目数量'] <= 100:
            return 50
        elif row['检查项目数量'] > 100:
            return 100

    check_df['检查项目数量区域'] = check_df.apply(check_range,axis=1)
    check_df = check_df.reindex(columns=['patientId', '报告时间', '入组时间', 'age', 'gender', '分组', '组名', '检查项目数量','检查项目数量区域', '检查项目', '血', '脑脊液',
        '粪便', '尿', '鼻/咽拭子', '骨髓', '动脉血', 'nan', '痰', '胃液', '肛拭子', '脓肿液', '穿刺液',
        '支刷', '导管', '支气管肺泡灌洗液', '体液', '呕吐物', '皮屑', '皮肤', '胆汁', '耳道分泌物', '胸腔液',
        '引流液', '眼部分泌物', '分泌物'])
    new_df = check_df.groupby(['检查项目数量区域'])['patientId'].count().to_frame()
    new_df = check_df.groupby(['检查项目数量区域'])['patientId'].count().to_frame()
    new_df = new_df.reset_index()

    categories = [1,2,3,'4-10','11-50','50-100','>100']
    values = new_df['patientId']
    x = range(len(categories))
    plt.bar(x,values, width=0.4, label='count')
    # plt.bar_label(plt.bar(categories, new_df['patientId']))
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom')

    plt.xticks(x, categories)
    plt.xlabel('check_number')
    plt.ylabel('count')
    plt.title('patient check_project count')
    plt.legend()
    plt.show()

def modify_qualitative(df2):
    # 脑脊液_红细胞计数_定量中无法检测的内容为脑脊液_红细胞计数_描述
    # 性中的满视野，将两项合并

    def change1(row):
        if row['脑脊液_红细胞计数_描述性'] == '满视野' and math.isnan(row['脑脊液_红细胞计数_定量']):
            return 100000
        else:
            return row['脑脊液_红细胞计数_定量']

    df2['脑脊液_红细胞计数_定量'] = df2.apply(change1,axis=1)
    # 将脑脊液_透明度_描述性特征转换为数字特征
    def change2(row):
        if row['脑脊液_透明度_描述性'] == '清':
            return 0
        elif row['脑脊液_透明度_描述性'] == '微浊':
            return 1
        elif row['脑脊液_透明度_描述性'] == '混浊':
            return 2
        return row['脑脊液_透明度_描述性']

    df2['脑脊液_透明度_描述性'] = df2.apply(change2,axis=1)


    #将脑脊液_肿瘤细胞_描述性特征转换为数字特征
    def change3(row):
        if row['脑脊液_肿瘤细胞_描述性'] == '[肿瘤细胞]少量' or row['脑脊液_肿瘤细胞_描述性'] == '【嗜碱性粒细胞计数】偶见' or row['脑脊液_肿瘤细胞_描述性'] == '【幼稚淋巴细胞】中量' or row['脑脊液_肿瘤细胞_描述性'] == '【幼稚淋巴细胞】大量':
            return math.nan
        elif row['脑脊液_肿瘤细胞_描述性'] == '阳性':
            return 1
        elif row['脑脊液_肿瘤细胞_描述性'] == '阴性':
            return 0
        return row['脑脊液_肿瘤细胞_描述性']

    df2['脑脊液_肿瘤细胞_描述性'] = df2.apply(change2,axis=1)

    # 将脑脊液_蛋白定性_定性特征转换为数字特征

    def change4(row):
        if row['脑脊液_蛋白定性_定性'] == '阴性':
            return 0
        elif row['脑脊液_蛋白定性_定性'] == '阳性':
            return 1
        return row['脑脊液_蛋白定性_定性']

    df2['脑脊液_蛋白定性_定性'] = df2.apply(change3,axis=1)

    # df2.groupby(['脑脊液_透明度_描述性'])['patientId'].count()
    change_features = ['血_隐球菌荚膜抗原_定性', '血_抗双链DNA抗体_定性', '血_抗核抗体_定性', '血_抗胞浆型中性粒细胞抗体_定性', '血_抗核周型中性粒细胞抗体_定性', '血_梅毒非特异性抗体_定性', '血_梅毒螺旋体抗体_定性',
                        '血_人类免疫缺陷病毒抗体_定性', '血_不规则抗体_定性', '血_直接抗人球蛋白试验_定性', '血_结核感染T细胞检测_定性', '血_人类免疫缺陷病毒P24抗原_定性',"血_M蛋白_描述性"]

    # 将change_features中的定性特征转换为数字特征

    for i in change_features:
        def change5(row):
            if row[i] == '阴性':
                return 0
            elif row[i] == '阳性':
                return 1
            return row[i]

        df2[i] = df2.apply(change4,axis=1)
    df2.groupby(['脑脊液_颜色_描述性'])['patientId'].count()

# 将每位病人每天的检查数据 合并为 一位患者只保留一条数据，如果有重复检测项目 保留为最早日期的检测结果
def main():
    df = pd.read_csv("../data/all_cleaned_data.csv")  # 根据实际情况读取你的数据
    final_df = build_wide_data(df)

    ruzu_df = pd.read_csv('../data/ruzu_20250814.csv')
    ruzu_df = data_counts(final_df,ruzu_df)
    final_df, ruzu_df = max_min_date(final_df,ruzu_df)
    ruzu_df = between_days(ruzu_df)

    ruzu_df.to_excel('../data/ruzu_20250825.xlsx',index=False)
    feature_df, feature_dict = get_all_features(final_df)
    feature_missing_rate(final_df,feature_dict)
    check_key_features(final_df,feature_dict,ruzu_df)
    final_df = inspection_count()
    check_data(final_df)

    # 根据医生的依据进一步整理数据
    df2 = pd.read_csv("../../data/wide_data.csv")

    # 将脑脊液中重要数据的定性结果按照医生的依据和定量结果合并

    list1 = ['脑脊液_中性粒细胞百分比_定性', '脑脊液_激活淋巴细胞_定性', '脑脊液_幼稚淋巴细胞_定性', '脑脊液_异常细胞百分比_定性','脑脊液_浆细胞_定性','脑脊液_嗜酸性粒细胞百分比_定性','脑脊液_激活巨噬细胞_定性']
    list2 = ['脑脊液_中性粒细胞百分比_定量', '脑脊液_激活淋巴细胞_定量', '脑脊液_幼稚淋巴细胞_定量', '脑脊液_异常细胞百分比_定量','脑脊液_浆细胞_定量','脑脊液_嗜酸性粒细胞百分比_定量','脑脊液_激活巨噬细胞_定量']

    for i in range(len(list1)):
        df2[list1[i]] = df2[list1[i]].map({'阴性': 0})

        def change(row):
            if row[list1[i]] == 0 and math.isnan(row[list2[i]]):
                return  0
            if not math.isnan(row[list2[i]]):
                return row[list2[i]]
            return math.nan

        df2[list2[i]] = df2.apply(change,axis=1)
    
    modify_qualitative(df2)

    def merge_group(group):

        group = group.sort_values('resultDateTime')
        result = {}
        for col in group.columns:
            if col == 'patientId':
                result[col] = group[col].iloc[0]
            elif col == 'resultDateTime':
                result[col] = group[col].min()
            if col == 'age':
                result[col] = group[col].iloc[0]
            elif col == 'gender':
                result[col] = group[col].min()
            else:
                non_null = group[col].dropna()
                if len(non_null) == 0:
                    result[col] = np.nan
                elif len(non_null) == 1:
                    result[col] = non_null.iloc[0]
                else:
                    min_date_idx = group.loc[group[col].notna(), 'resultDateTime'].idxmin()
                    result[col] = group.loc[min_date_idx, col]

        return pd.Series(result)

    result = df2.groupby(['patientId']).apply(merge_group).reset_index(drop=True)
    missing_rate_df = pd.DataFrame.from_dict(dict(result.isna().sum()/887), orient='index')
    missing_rate_df.to_excel("../data/缺失值.xlsx")
    # 根据筛选后 需要保留的特征

    blood_features = ['血_白细胞计数_定量','血_红细胞比容_定量','血_红细胞分布宽度_定量','血_红细胞计数_定量','血_平均红细胞体积_定量','血_平均红细胞血红蛋白含量_定量','血_平均红细胞血红蛋白浓度_定量','血_血小板计数_定量',
                    '血_总血红蛋白浓度_定量','血_嗜碱性粒细胞百分比_定量','血_嗜碱性粒细胞计数_定量','血_嗜酸性粒细胞百分比_定量','血_嗜酸性粒细胞计数_定量','血_中性粒细胞百分比_定量','血_中性粒细胞计数_定量',
                    '血_大血小板细胞比率_定量','血_平均血小板体积_定量','血_血小板分布宽度_定量','血_C反应蛋白_定量','血_单核细胞计数_定量','血_淋巴细胞计数_定量','血_单核细胞百分比_定量','血_淋巴细胞百分比_定量',
                    '血_血小板/淋巴细胞比值_定量','血_中性粒细胞/淋巴细胞比值_定量','血_淀粉样蛋白A_定量','血_降钙素原_定量','血_白细胞介素-6_定量','血_白细胞介素-10_定量','血_白细胞介素-12p70_定量','血_白细胞介素-17_定量',
                    '血_白细胞介素-2_定量','血_白细胞介素-4_定量','血_白细胞介素-1β_定量','血_白细胞介素-8_定量','血_白细胞介素-5_定量','血_有核红细胞百分比_定量','血_幼稚粒细胞百分比_定量',
                    '血_红细胞分布宽度标准差_定量','血_高荧光强度网织红细胞_定量']
    naojiye_features = ['脑脊液_红细胞计数_定量','脑脊液_有核细胞计数_定量','脑脊液_蛋白定量_定量','脑脊液_氯_定量','脑脊液_葡萄糖_定量','脑脊液_蛋白定性_定性','脑脊液_透明度_描述性','脑脊液_颜色_描述性','脑脊液_激活巨噬细胞_定量',
                        '脑脊液_激活淋巴细胞_定量','脑脊液_嗜酸性粒细胞百分比_定量','脑脊液_幼稚淋巴细胞_定量','脑脊液_浆细胞_定量','脑脊液_中性粒细胞百分比_定量','脑脊液_淋巴细胞百分比_定量',
                        '脑脊液_巨噬细胞百分比_定量','脑脊液_单个核细胞百分比_定量','脑脊液_多个核细胞百分比_定量']

    naojiye_selected_features = ['脑脊液_红细胞计数_定量','脑脊液_有核细胞计数_定量','脑脊液_蛋白定量_定量','脑脊液_氯_定量','脑脊液_葡萄糖_定量','脑脊液_蛋白定性_定性','脑脊液_透明度_描述性','脑脊液_颜色_描述性']
    final_features_df = result[['patientId', 'resultDateTime', 'age', 'gender']+blood_features+naojiye_features]
    final_features_df2 = pd.DataFrame.from_dict(dict(final_features_df.isna().sum()/887), orient='index')
    final_features_df2.to_excel("../data/最终缺失值.xlsx")
    final_features_df['patientId'] = final_features_df['patientId'].astype(str)
    ruzu_df = pd.read_excel('../data/ruzu_20250825.xlsx')
    final_data_df = pd.merge(final_features_df,ruzu_df[['patientId','分组','组名']])
    final_data_df = final_data_df.drop(['patientId','resultDateTime'], axis=1)
    # 生成最终特征数据
    final_data_df.to_csv("../../data/final_data2/filtered_patient1.csv", index=False)
    needed_feature = dict(result.isna().mean()[result.isna().mean() < 0.66])

    columns_to_keep1 = [item for item in needed_feature if item.startswith('血_')]
    # temp_df = pd.DataFrame.from_dict(dict(result[columns_to_keep1+naojiye_features].isna().sum()/887),orient='index')
    # 血常规特征缺失率<0.65 和 重要的脑脊液特征 (去除血_隐球菌荚膜抗原_定性，血_维生素B12_定量，血_轻链.λ型_定量 )

    needed_feature = dict(result.isna().mean()[result.isna().mean() < 0.66])

    columns_to_keep1 = [item for item in needed_feature if item.startswith('血_')]

    columns_to_keep1.remove('血_隐球菌荚膜抗原_定性')
    columns_to_keep1.remove('血_维生素B12_定量')
    columns_to_keep1.remove('血_轻链.λ型_定量')


    result['patientId'] = result['patientId'].astype(str)
    temp_df = pd.merge(result[['patientId']+columns_to_keep1+naojiye_features],ruzu_df[['patientId','性别','年龄','分组','组名']])
    temp_df = temp_df.rename(columns={'年龄':'age','性别':'gender'})
    temp_df = temp_df.drop(['patientId'], axis=1)
    temp_df.to_csv("../../data/final_data2/filtered_patient2.csv", index=False)
    # 只保留重要的血常规特征
    # temp_df2 = pd.DataFrame.from_dict(dict(result[blood_features].isna().sum()/887),orient='index')

    result['patientId'] = result['patientId'].astype(str)
    temp_df2 = pd.merge(result[['patientId']+blood_features],ruzu_df[['patientId','性别','年龄','分组','组名']])
    temp_df2 = temp_df2.rename(columns={'年龄':'age','性别':'gender'})
    temp_df2 = temp_df2.drop(['patientId'], axis=1)
    temp_df2.to_csv("../../data/final_data2/filtered_patient3.csv", index=False)
    # 只保留缺失率小于65%的血常规特征
    # temp_df3 = pd.DataFrame.from_dict(dict(result[columns_to_keep1].isna().sum()/887),orient='index')

    result['patientId'] = result['patientId'].astype(str)
    temp_df3 = pd.merge(result[['patientId']+columns_to_keep1],ruzu_df[['patientId','性别','年龄','分组','组名']])
    temp_df3 = temp_df3.rename(columns={'年龄':'age','性别':'gender'})
    temp_df3 = temp_df3.drop(['patientId'], axis=1)
    temp_df3.to_csv("../../data/final_data2/filtered_patient4.csv", index=False)
    # 只保留重要的脑脊液特征
    # temp_df4 = pd.DataFrame.from_dict(dict(result[naojiye_features].isna().sum()/887),orient='index')

    result['patientId'] = result['patientId'].astype(str)
    temp_df4 = pd.merge(result[['patientId']+naojiye_features],ruzu_df[['patientId','性别','年龄','分组','组名']])
    temp_df4 = temp_df4.rename(columns={'年龄':'age','性别':'gender'})
    temp_df4 = temp_df4.drop(['patientId'], axis=1)
    temp_df4.to_csv("../../data/final_data2/filtered_patient5.csv", index=False)
    # 只保留去除脑脊液细胞形态学分类的脑脊液特征
    # temp_df4 = pd.DataFrame.from_dict(dict(result[naojiye_features].isna().sum()/887),orient='index')

    result['patientId'] = result['patientId'].astype(str)
    temp_df5 = pd.merge(result[['patientId']+naojiye_selected_features],ruzu_df[['patientId','性别','年龄','分组','组名']])
    temp_df5 = temp_df5.rename(columns={'年龄':'age','性别':'gender'})
    temp_df5 = temp_df5.drop(['patientId'], axis=1)
    temp_df5.to_csv("../../data/final_data2/filtered_patient6.csv", index=False)
    # 只保留8项脑脊液特征+40项血液特征
    # temp_df4 = pd.DataFrame.from_dict(dict(result[naojiye_features].isna().sum()/887),orient='index')

    result['patientId'] = result['patientId'].astype(str)
    temp_df6 = pd.merge(result[['patientId']+blood_features+naojiye_selected_features],ruzu_df[['patientId','性别','年龄','分组','组名']])
    temp_df6 = temp_df6.rename(columns={'年龄':'age','性别':'gender'})
    temp_df6 = temp_df6.drop(['patientId'], axis=1)
    temp_df6.to_csv("../../data/final_data2/filtered_patient7.csv", index=False)
    # 只保留8项脑脊液特征+206项血液特征
    # temp_df4 = pd.DataFrame.from_dict(dict(result[naojiye_features].isna().sum()/887),orient='index')

    result['patientId'] = result['patientId'].astype(str)
    temp_df7 = pd.merge(result[['patientId']+columns_to_keep1+naojiye_selected_features],ruzu_df[['patientId','性别','年龄','分组','组名']])
    temp_df7 = temp_df7.rename(columns={'年龄':'age','性别':'gender'})
    temp_df7 = temp_df7.drop(['patientId'], axis=1)
    temp_df7.to_csv("../../data/final_data2/filtered_patient8.csv", index=False)



if __name__ == "__main__":
    main()