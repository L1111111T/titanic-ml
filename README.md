# Titanic - 机器学习生存预测项目

> 二分类任务 · [Kaggle竞赛页面](https://www.kaggle.com/competitions/titanic)
> 
> 公榜得分: 0.77511 · 比赛排名: / 12,867

---

# 项目概述

泰坦尼克号（Titanic）于1912年沉没，共造成2224名乘客与船员中的1502人遇难。本项目基于乘客的人口统计信息与船票相关信息，构建机器学习二分类模型，对乘客是否生还进行预测。

项目完整实现了数据探索分析（EDA）、特征工程（Feature Engineering）、数据预处理（Preprocessing）、模型训练与比较（Model Comparison）、超参数调优（Hyperparameter Tuning）的一整套机器学习流程。

* **训练集**：891条数据 · 11个特征
* **测试集**：418条数据
* **目标变量**：`Survived`

  * `0` = 未生还
  * `1` = 生还
* 整体生还率约为38%

---

# 项目流程

## 1. 数据读取与初步检查
使用 `pandas` 读取 `train.csv` 与 `test.csv`，并对数据结构进行检查：
* 使用 `df.info()` 查看数据类型与缺失值
* 使用 `df.describe()` 查看统计信息
* 识别主要缺失字段： `Age`, `Cabin`, `Embarked`

---

## 2. Exploratory Data Analysis（EDA）
在模型训练之前，使用多种可视化方式探索数据规律。
本项目共使用了六种图表：
| 图表类型            | 用途                 |
| --------------- | ------------------ |
| 饼图（Pie Chart）   | 查看整体生还分布           |
| 柱状图（Bar Chart）  | 比较性别、舱位等级、登船港口的生还率 |
| 直方图 + KDE       | 分析年龄分布与生还关系        |
| 箱线图（Box Plot）   | 比较不同舱位与票价分布        |
| 折线图（Line Chart） | 分析家庭人数与生还率关系       |
| 热力图（Heatmap）    | 查看数值变量相关性          |

### 关键发现
* 女性乘客生还率显著高于男性；一等舱乘客生还率远高于三等舱；独自出行乘客生还率较低；较高票价通常对应更高生还概率

---

# 3. 特征工程（Feature Engineering）
基于原始字段构建了三个新的机器学习特征：
```python
def feature_engineering(df):

    # 从姓名中提取称谓（Mr、Mrs、Miss等）
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

    # 家庭人数
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    # 是否独自出行
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    return df
```

新增特征：
| 特征           | 含义     |
| ------------ | ------ |
| `Title`      | 乘客称谓   |
| `FamilySize` | 家庭人数   |
| `IsAlone`    | 是否独自出行 |

同时删除了 `Name` `Ticket` `Cabin` `PassengerId`等对模型帮助较小或缺失严重的字段。

---

# 4. 数据预处理 Pipeline
使用 `Pipeline` `ColumnTransformer`构建完整的数据预处理流程，防止测试集信息泄露（Data Leakage）。

## 数值变量处理
```python
SimpleImputer(strategy='median')
→ StandardScaler()
```

## 类别变量处理
```python
SimpleImputer(strategy='most_frequent')
→ OneHotEncoder()
```

---

# 5. 模型训练与比较
本项目系统性训练并比较了五种机器学习模型：
| 模型                  | 作用             | 调参方式         |
| ------------------- | -------------- | ------------ |
| DummyClassifier     | 基线模型（Baseline） | 无            |
| Logistic Regression | 线性分类模型         | 5-fold CV    |
| Decision Tree       | 可解释非线性模型       | GridSearchCV |
| Random Forest       | Bagging集成学习    | GridSearchCV |
| Gradient Boosting   | Boosting集成学习   | GridSearchCV |

---

# GridSearchCV 超参数搜索
## Decision Tree

```python
{
 'dt_clf__criterion': ['gini', 'entropy'],
 'dt_clf__max_depth': [2, 3, 4, 5, 6],
 'num_imputer__strategy': ['mean', 'median']
}
```

## Random Forest

```python
{
 'rf_clf__n_estimators': [50, 100, 200],
 'rf_clf__max_depth': [3, 5, 7, None],
 'rf_clf__criterion': ['gini', 'entropy']
}
```

## Gradient Boosting

```python
{
 'gb_clf__n_estimators': [100, 200],
 'gb_clf__learning_rate': [0.05, 0.1],
 'gb_clf__max_depth': [3, 4]
}
```

---

# 模型结果

| 模型                         | Accuracy |
| -------------------------- | -------- |
| Baseline (DummyClassifier) | ~0.616   |
| Logistic Regression        | [填入]     |
| Decision Tree              | [填入]     |
| Random Forest              | [填入]     |
| Gradient Boosting          | [填入]     |

> 最终提交模型：Logistic Regression
> Kaggle Public Score：0.77511

---

# Feature Importance（随机森林）

| 排名 | 特征   | 含义   |
| -- | ---- | ---- |
| 1  | [填入] | [填入] |
| 2  | [填入] | [填入] |
| 3  | [填入] | [填入] |
| 4  | [填入] | [填入] |
| 5  | [填入] | [填入] |

---

# 数据来源

本项目数据来自 Kaggle 竞赛官网：
1. 访问 [比赛页面](https://www.kaggle.com/competitions/titanic)
2. 在 **Data** 中下载并解压 `train.csv` 和 `test.csv` 放在与 Notebook 相同的目录下

---

# 仓库结构

| 文件                   | 说明       |
| -------------------- | -------- |
| `Titanic_code.ipynb` | 完整机器学习流程 |
| `requirements.txt`   | Python 依赖包版本 |
| `.gitignore`         | 排除数据文件和临时文件  |

---

# 项目涉及核心知识点

| 方向             | 内容                             |
| -------------- | ------------------------------ |
| 数据结构           | List、DataFrame                 |
| 函数与流程控制        | 特征工程函数                         |
| NumPy / pandas | 缺失值处理、groupby、类型转换             |
| 数据可视化          | 六类统计图                          |
| 机器学习分类         | LR、DT、RF、GBM                   |
| 面向对象设计         | Pipeline封装                     |
| 超参数调优          | GridSearchCV                   |
| 模型评估           | Accuracy、CV、Feature Importance |

---

# 技术栈

```text
Python 3.12
pandas
NumPy
scikit-learn
Matplotlib
seaborn
```
