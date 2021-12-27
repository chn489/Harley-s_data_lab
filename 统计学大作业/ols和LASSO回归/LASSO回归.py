import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
import data_handler
import matplotlib.pyplot as plt

df = pd.read_excel('创业精神.xlsx')
sch_list = data_handler.build_sch_list(100)
school = df['(1)1、您就读的大学是:___']
school_label = data_handler.relabel_school(school, sch_list)
score = df['总分']
major = df['2、您的专业类型']
sex = df['3、您的性别是:']
grade = df['4、您的年级']
imagination = df['1、我对许多不同的事物都感兴趣，富有想象力']
confidence = df['3、我性格坚定自信，敢于表达自己的观点']
method = df['2、做同一件事的时候，我乐意想出与别人不同的方法。']
thinking = df['4、解决问题过程中，我会有意识地对自己的思路是否正确、合理进行监督控制']
risk = df['6、为了达到预期目标，我乐于在完成过程中去承担各种风险']
sch_edu = df['2、我所在的高校在创新创业教育课程方面']
support = df['6、我所在学校在创业基金方面对学生创业项目支持力度']
guidance = df['7、我所在学校通过多种渠道为我们提供创业指导']

feature = np.vstack((school_label, major, sex, grade, imagination, confidence, method, thinking, risk, sch_edu, support,
                     guidance)).transpose()
target = score
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2,
                                                                          random_state=0)

Lamdas = np.logspace(-5, 2, 200)
lasso_cv = LassoCV(alphas=Lamdas, normalize=True, max_iter=2000)
lasso_cv.fit(feature_train, target_train)
# print(lasso_cv.alpha_)
lasso = Lasso(alpha=lasso_cv.alpha_, normalize=True, max_iter=2000)
lasso.fit(feature_train, target_train)
print(lasso.coef_, lasso.intercept_)
# 评价模型
pred = lasso.predict(feature_test)
MSE = mean_squared_error(target_test, pred)
print('mse=', MSE)
r2 = r2_score(target_test, pred)
print('r2=', r2)
plt.plot(pred, label='pred')
plt.plot(target_test, label='target_test')
plt.legend()
plt.show()
