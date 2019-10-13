
# -*- coding: utf-8 -*-

# 引入模块
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

# 读取数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv("sample_submit.csv")



#预览一下
print(train.head())
print(test.head())
print(submit.head())



# 取出训练集的y
y_train = train.pop('y')

#building
clf = DecisionTreeRegressor()
clf.fit(train, y_train)
y_pred = clf.predict(test)

#决策树模型会依照数据的标签为每个条件进行决策分类

# 输出预测结果至my_DT_prediction.csv
submit['y'] = y_pred
submit.to_csv('my_DT_prediction.csv', index=False)


print("success!")
