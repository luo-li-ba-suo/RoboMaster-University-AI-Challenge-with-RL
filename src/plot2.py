import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_data():
    '''获取数据
    '''
    basecond = np.array([[18, 20, 19, 18, 13, 4, 1],[20, 17, 12, 9, 3, 0, 0],[20, 20, 20, 12, 5, 3, 0]])
    basecond = np.array([[18, 20, 19, 18, 13, 4, 1]])
    cond1 = np.array([[18, 19, 18, 19, 20, 15, 14],[19, 20, 18, 16, 20, 15, 9],[19, 20, 20, 20, 17, 10, 0]])
    cond2 = np.array([[20, 20, 20, 20, 19, 17, 4],[20, 20, 20, 20, 20, 19, 7],[19, 20, 20, 19, 19, 15, 2]])
    cond3 = np.array([[20, 20, 20, 20, 19, 17, 12],[18, 20, 19, 18, 13, 4, 1], [20, 19, 18, 17, 13, 2, 0]])
    return basecond, cond1, cond2, cond3


if __name__ == '__main__':
    data = get_data()
    label = ['algo1', 'algo2', 'algo3', 'algo4']
    df=[]
    for i in range(len(data)):
        df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='loss'))
        df[i]['algo']= label[i]

    df=df[0] # 合并
    sns.lineplot(x="episode", y="loss", hue="algo", style="algo",data=df)
    plt.title("some loss")
    plt.show()
    
