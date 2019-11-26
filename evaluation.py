import pandas as pd

class Evaluation(object):
    def __init__(self, trueResultFileDir, preResultFileDir):

        self.dfTrueResult = pd.DataFrame()
        self.dfPreResult = pd.DataFrame()
        self.acc = 0

        # preprocess scholar_final_truth.txt, 共有5367个scholars
        with open(trueResultFileDir, 'r', encoding = 'utf-8') as f:
            trueResultList = list(f.read().split('\n')[:-1])
        # 提取最终true数据中task2的部分
        flag = 0
        trueData = []
        for item in trueResultList:
            if item == '<task2>':
                flag = 1
            if item == '</task2>':
                break
            if flag == 1:
                trueData.append(item)
        trueDataList = trueData[2:]
        # 将true数据中task2的部分存为dataframe
        self.dfTrueResult = pd.DataFrame(columns=['author', 'Interest1', 'Interest2', 'Interest3'])
        for idx, item in enumerate(trueDataList):
            self.dfTrueResult.loc[idx] = item.split('\t')
        self.dfTrueResult.set_index(['author'], drop=False, inplace=True)

        #read the predict result
        self.dfPreResult = pd.read_csv(preResultFileDir)
        self.dfPreResult = self.dfPreResult.drop([0, 5368])  # 删除第一行和最后一行
        self.dfPreResult.rename(columns={'<task2>': 'author'}, inplace=True)
        self.dfPreResult = pd.concat([self.dfPreResult['author'].str.split('\t', expand=True)], axis=1)
        self.dfPreResult.columns = ['author', 'Interest1', 'Interest2', 'Interest3', 'Interest4', 'Interest5']
        self.dfPreResult.set_index(['author'], drop=True, inplace=True)
        # ？？？？？
        # 不保留'author'这一列的准确度为0.322
        # 保留dfTrueResult在'author'这一列的准确度为0.242
        # 这是为什么？？？？因为len(set(self.dfTrueResult.iloc[i].tolist()))要减一

        #evaluation function
        for i in range(self.dfPreResult.shape[0]):
            # print(set(self.dfPreResult.iloc[i].tolist()))
            # print(set(self.dfTrueResult.iloc[i].tolist()))
            # break
            acci = len(set(self.dfPreResult.iloc[i].tolist()) & set(self.dfTrueResult.iloc[i].tolist()))
            acciRate = acci / (len(set(self.dfTrueResult.iloc[i].tolist()))-1)
            self.acc = self.acc + acciRate
        self.acc = self.acc / self.dfPreResult.shape[0]

if __name__=='__main__':

    trueResultFileDir = '/Users/yuansha/Desktop/portrait/data of task2/scholar_final_truth.txt'
    preResultFileDir = '/Users/yuansha/task2_team9/validation2.csv'

    evalInstance = Evaluation(trueResultFileDir, preResultFileDir)
    print(f'The accuracy is: {evalInstance.acc}') #validation2文件的正确值是0.32246
