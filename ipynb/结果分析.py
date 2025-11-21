import pandas as pd
import os


def save_to_excel():
    # 将json数据分模型存储至Excel
    folder = "../data/model_data4"
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            all_files.append(os.path.join(root, file))

    # 数据输出
    output_file = "../模型训练结果.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for file in all_files:
            # if file == '../../data/model_data/SVM_data1.json':
            #     continue
            # if file == '../../data/model_data/RF_data1.json':
            #     continue
            print(file)
            data = pd.read_json(file)
            result = []
            count = 0
            for dataset_name, comparisons in data.items():
                records = []

                for comparison_name, metrics in comparisons.items():
                    # print(metrics)
                    auc = round(metrics['roc_auc_test'],4)
                    # CI = (round(metrics['test_CI'][0],4),round(metrics['test_CI'][1],4))
                    row = { "任务组": comparison_name, f"{dataset_name}": f'{auc}'}
                    # row = {"数据集":dataset_name, "任务组": comparison_name}
                    # print(metrics['roc_auc_test'])
                    # row.update(metrics)
                    records.append(row)
                    # print(records)
                    # # break
                if count == 0:
                    results = records
                    count += 1
                else:
                    for record in records:
                        for result in results:
                            if result['任务组'] == record['任务组']:
                                result[list(record.keys())[1]] = record[list(record.keys())[1]]

            df = pd.DataFrame(results)
            model = (file.split('/')[-1]).split('_')[0]
            if (file.split('/')[-1]).split('_')[1] != 'data.json':
                model = model + '_' + (file.split('/')[-1]).split('_')[1]
            df = df.reindex(columns=['任务组',f'{model}_Dataset1',f'{model}_Dataset2',f'{model}_Dataset3',f'{model}_Dataset4',f'{model}_Dataset5',f'{model}_Dataset6',f'{model}_Dataset7',f'{model}_Dataset8',f'{model}_Dataset9'])
            df.rename(columns={f'{model}_Dataset1': '血液37项+脑脊液19项', f'{model}_Dataset2': '血液203项+脑脊液19项', f'{model}_Dataset3': '血液37项',
                            f'{model}_Dataset4': '血液203项', f'{model}_Dataset5': '脑脊液19项' ,f'{model}_Dataset6': '脑脊液8项',f'{model}_Dataset7': '血液37项+脑脊液8项',
                            f'{model}_Dataset8': '血液203+脑脊液8',f'{model}_Dataset9': '精选特征集'}, inplace=True)

            sheet = ((file.split("/")[-1]).split(".")[0]).split("_")[0]
            if ((file.split("/")[-1]).split(".")[0]).split("_")[1] != 'data':
                sheet = sheet + '+' + ((file.split("/")[-1]).split(".")[0]).split("_")[1]
            # else:
            #     sheet = sheet + ((file.split("/")[-1]).split(".")[0]).split("_")[1]
            sheet_name = f"{sheet}模型结果"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"已保存到 {output_file} 的 sheet: {sheet_name}")

def save_result():
    # 生成单模型的最优模型结果

    dfs = pd.read_excel("../模型训练结果.xlsx", sheet_name=None)

    df = dfs['XGboost模型结果']

    result1 = {}
    row = df['任务组']

    for i in (df.columns)[1:]:
        for j in range(len(row)):
            if i not in result1.keys():
                # result1[i] = {row[j]: f'{df[i][j]*100:.1f}%(XGboost)'}
                result1[i] = {row[j]: f'{df[i][j]*100:.1f}'}
            else:
                # result1[i][row[j]] = f'{df[i][j]*100:.1f}%(XGboost)'
                result1[i][row[j]] = f'{df[i][j]*100:.1f}'

    print(result1)
    for sheet, df in dfs.items():
        if sheet == '所有模型最优结果对比' or sheet == 'HMM模型结果' or sheet == 'LR+XGBoost1模型结果' or sheet == 'LR+XGBoost2模型结果' or sheet == 'LR+XGBoost3模型结果' :
            continue
        model = sheet[:-4]
        row = df['任务组']

        for i in (df.columns)[1:]:
            for j in range(len(row)):
                # compare1 = float(result1[i][row[j]].split("%")[0])
                compare1 = float(result1[i][row[j]])
                compare2 = float(f'{df[i][j]*100:.1f}')
                # print(compare1,compare2)
                if compare2 > compare1:
                    # result1[i][row[j]] = f'{compare2}%({model})'
                    result1[i][row[j]] = f'{compare2}'


    df3 = pd.DataFrame(result1)
    df3.index.name = '任务组'
    df3 = df3.reset_index()
    df3.to_excel("../单模型最优结果对比.xlsx", index=False)

def save_result_with_model_name():
    # 生成 单模型最优结果对比(带模型名称)
    dfs = pd.read_excel("../模型训练结果.xlsx", sheet_name=None)

    df = dfs['XGboost模型结果']

    result1 = {}
    row = df['任务组']

    for i in (df.columns)[1:]:
        for j in range(len(row)):
            if i not in result1.keys():
                result1[i] = {row[j]: f'{df[i][j]*100:.1f}%(XGboost)'}
                # result1[i] = {row[j]: f'{df[i][j]*100:.1f}'}
            else:
                result1[i][row[j]] = f'{df[i][j]*100:.1f}%(XGboost)'
                # result1[i][row[j]] = f'{df[i][j]*100:.1f}'


    for sheet, df in dfs.items():
        if sheet == '所有模型最优结果对比' or sheet == 'HMM模型结果' or sheet == 'LR+XGBoost1模型结果' or sheet == 'LR+XGBoost2模型结果' or sheet == 'LR+XGBoost3模型结果' :
            continue
        model = sheet[:-4]
        row = df['任务组']

        for i in (df.columns)[1:]:
            for j in range(len(row)):
                compare1 = float(result1[i][row[j]].split("%")[0])
                # compare1 = float(result1[i][row[j]])
                compare2 = float(f'{df[i][j]*100:.1f}')
                # print(compare1,compare2)
                if compare2 > compare1:
                    result1[i][row[j]] = f'{compare2}%({model})'
                    # result1[i][row[j]] = f'{compare2}'

    df3 = pd.DataFrame(result1)
    df3.index.name = '任务组'
    df3 = df3.reset_index()
    df3.to_excel("../单模型最优结果对比(带模型).xlsx", index=False)

def save_result_all():
    # 生成 全部模型最优结果对比

    dfs = pd.read_excel("../模型训练结果.xlsx", sheet_name=None)

    df = dfs['XGboost模型结果']

    result1 = {}
    row = df['任务组']

    for i in (df.columns)[1:]:
        for j in range(len(row)):
            if i not in result1.keys():
                # result1[i] = {row[j]: f'{df[i][j]*100:.1f}%(XGboost)'}
                result1[i] = {row[j]: f'{df[i][j]*100:.1f}'}
            else:
                # result1[i][row[j]] = f'{df[i][j]*100:.1f}%(XGboost)'
                result1[i][row[j]] = f'{df[i][j]*100:.1f}'


    for sheet, df in dfs.items():
        if sheet == '所有模型最优结果对比' or sheet == 'HMM模型结果':
            continue
        model = sheet[:-4]
        row = df['任务组']

        for i in (df.columns)[1:]:
            for j in range(len(row)):
                # compare1 = float(result1[i][row[j]].split("%")[0])
                compare1 = float(result1[i][row[j]])
                compare2 = float(f'{df[i][j]*100:.1f}')
                # print(compare1,compare2)
                if compare2 > compare1:
                    # result1[i][row[j]] = f'{compare2}%({model})'
                    result1[i][row[j]] = f'{compare2}'

    # sheet_name = f"最优结果对比"
    # df.to_excel(writer, sheet_name=sheet_name, index=False)

    # print(f"已保存到 {output_file} 的 sheet: {sheet_name}")

    df3 = pd.DataFrame(result1)
    df3.index.name = '任务组'
    df3 = df3.reset_index()
    df3.to_excel("../全部模型最优结果对比.xlsx", index=False)


def save_result_all_with_model_name():
    # 生成 全部模型最优结果对比（带模型名称）

    dfs = pd.read_excel("../模型训练结果7.xlsx", sheet_name=None)

    df = dfs['XGboost模型结果']

    result1 = {}
    row = df['任务组']

    for i in (df.columns)[1:]:
        for j in range(len(row)):
            if i not in result1.keys():
                result1[i] = {row[j]: f'{df[i][j]*100:.1f}%(XGboost)'}
                # result1[i] = {row[j]: f'{df[i][j]*100:.1f}'}
            else:
                result1[i][row[j]] = f'{df[i][j]*100:.1f}%(XGboost)'
                # result1[i][row[j]] = f'{df[i][j]*100:.1f}'


    for sheet, df in dfs.items():
        if sheet == '所有模型最优结果对比' or sheet == 'HMM模型结果':
            continue
        model = sheet[:-4]
        row = df['任务组']

        for i in (df.columns)[1:]:
            for j in range(len(row)):
                compare1 = float(result1[i][row[j]].split("%")[0])
                # compare1 = float(result1[i][row[j]])
                compare2 = float(f'{df[i][j]*100:.1f}')
                # print(compare1,compare2)
                if compare2 > compare1:
                    result1[i][row[j]] = f'{compare2}%({model})'
                    # result1[i][row[j]] = f'{compare2}'

    # sheet_name = f"最优结果对比"
    # df.to_excel(writer, sheet_name=sheet_name, index=False)

    # print(f"已保存到 {output_file} 的 sheet: {sheet_name}")

    df3 = pd.DataFrame(result1)
    df3.index.name = '任务组'
    df3 = df3.reset_index()
    df3.to_excel("../全部模型最优结果对比（带模型）.xlsx", index=False)


def main():
    save_to_excel()
    save_result()
    save_result_with_model_name()
    save_result_all()
    save_result_all_with_model_name()

    
if __name__ == "__main__":
    main()