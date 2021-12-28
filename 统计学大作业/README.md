这是一份关于大学生创业精神的调查结果分析。选取了一些因子对总分进行了建模。  
  
因子：  
具体12个因子分别是：school_label,major,sex,grade,imagination,confidence,method,thinking,risk,school_education,financial_support,guidance;  
特别说明的是，在school_label的构造上，我的做法是爬取2021年校友会中国大学星级排名形成一个excel,然后选取前100所学校作为sch_list（即‘重点大学’名单），并且将一些学校的缩写也加入到list中。然后将被调查者的学校与sch_list比对，在list里面则school_label=1,不在则等于0.
前4个因子是基本信息，从imagination到risk是个人特质，最后三个是外部因素，主要是学校在创新创业上的支持。  
  
模型：  
我使用了5种方法对总分进行了建模，分别是ols,决策树，LASSO回归，XGBoost回归和XGBoost多分类。（或许以后还有DNN和CNN的加入）具体结果请查看对应子目录。  
  
数据清洗：  
数据清洗主要依靠自己写的data_handler库，里面的功能有按照性别、学校和专业进行数据分组、relabel（给学校和总分重新打上标签，对总分relabel时又有relabel_score和relabel_score_xgb两种，分别用于普通决策树和xgboost）和描述统计（最大最小值、平均值和方差）  
  
爬虫：  
见get_sch_list.py
