这步之后，你将会理解过拟合和欠拟合的概念，你可以使用这些ideas来是你的Models更加准确

# Experimenting with Different Models
现在你可以使用sklearn.metric中的mae来评估你的模型了，也可以使用其他的模型来实验，看哪种模型可以获得最好的预测结果。但是你都有哪些可以替换的模型呢？


在sklearn的文档中，决策树的模型有很多选项。最重要的一个选项是决策树的深度。树的深度是在预测之前对其进行多少次分裂的度量。

在实际中，在树的顶层（所有房屋）和一个叶子之间有10次分裂并不少见。随着树越深，数据集被分为具有更少房屋的叶子。如果一棵树只有一次分裂，那么它将数据分为2组。如果再次分裂，将会有4组房屋。再次分裂，则有8组。若分裂次数为10次，那么我们将会获得1024组房屋，也就是1024个叶子。

当我们划分越多的叶子时，每个叶子分到的房屋数也就越少。非常少房屋的叶子将会使得预测只非常解决房屋的实际价格，但是也会导致对新数据的预测不可靠（因为每个预测只基于非常少的房屋）。

这种现象叫做**过拟合**，就是对训练数据拟合特别好，但是对验证集或者其他新数据的预测就不好。另一方面，如果我们让树非常浅，那么它就不会将房屋们划分为非常不同的组别。

如果只将房屋分为2或者4组，每一组将会有各种各样的房屋。大多数房屋的预测结果可能都不理想，即使是在训练数据中，同样的原因也会导致验证数据预测不理想。模型不能捕获到数据中重要的差别和模式，所以在训练数据中表现很差，这叫做**欠拟合**。

既然我们比较关注新数据中的准确率，因此要找到过拟合与欠拟合的sweet spot。


## 例子

通过控制max_leaf_nodes来调节模型复杂度，通过比较MAE score来比较不同模型的效果。

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_y)
    mae = mean_absolute_error(val_y, preds_val)
    return mae
```

可以使用for循环来比较有不同max_leaf_nodes值的模型的准确率

```python
#compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" %(max_leaf_nodes, my_mae))

```

通过返回的结果可以发现，500是叶子最佳数量。


## 结论

模型可能受到以下任何一种的影响：
    过拟合：捕获了在未来不会再次出现的虚假模型，导致预测结果不理想
    欠拟合：没有捕获到重要的模式，导致预测结果不理想

我们使用验证数据来评测每个候选模型的准确率。这可以让我们查收更多的模型并且保持最好的那一个。

## 练习

先将训练数据分为train和val, 选出最合适的深度best_tree_size，然后用整个的训练数据X,y去训练深度为best_tree_size的model

```python
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def get_mae(max_leaf_nodes, trian_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_y)
    return mae


# path of the file to read
iowa_file_paht = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
# create target object and call it y
y = home_data.SalePrice
# create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_ndoes}
best_tree_size = min(scores, key=scores.get)

# specific model
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model
final_model.fit(X, y)
```
