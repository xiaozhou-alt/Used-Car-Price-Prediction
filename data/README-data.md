# **【AI入门系列】车市先知：二手车价格预测学习赛**

# 赛制

<table><tr><td>赛题与数据</td><td>文档</td><td>大小</td><td>操作</td><td>ossutil命令</td></tr><tr><td>排行榜</td><td>used_car_sample_submit.csv</td><td>.csv(439KB)</td><td>下载</td><td>复制命令</td></tr><tr><td>代码规范</td><td>used_car_test_20200421.zip</td><td>zip(7MB)</td><td>下载</td><td>复制命令</td></tr><tr><td>学习建议</td><td>used_car_train_20200313.zip</td><td>zip(2MB)</td><td>下载</td><td>复制命令</td></tr><tr><td>获奖名单</td><td>used_car_testA_20200313.csv.zip</td><td>zip(7MB)</td><td>下载</td><td>复制命令</td></tr><tr><td>论坛</td><td>used_car_testA_20200313.csv.zip</td><td></td><td></td><td></td></tr></table>



==项目源于：[【AI入门系列】车市先知：二手车价格预测学习赛_学习赛_赛题与数据_天池大赛-阿里云天池的赛题与数据 (aliyun.com)](https://tianchi.aliyun.com/competition/entrance/231784/information)==

# 一、赛题数据

赛题以预测二手车的交易价格为任务，数据集报名后可见并可下载，该数据来自某交易平台的二手车交易记录，总数据量超过 40w，包含 3 息，其中 15 列为匿名变量。为了保证比赛的公平性，将会从中抽取 15 万条作为训练集，5 万条作为测试集A，5 万条作为测试集B，同时会对model、brand 和 regionCode 等信息进行脱敏。

字段表  

<table><tr><td>Field</td><td>Description</td></tr><tr><td>SaleID</td><td>交易ID，唯一编码</td></tr><tr><td>name</td><td>汽车交易名称，已脱敏</td></tr><tr><td>regDate</td><td>汽车注册日期，例如20160101，2016年01月01日</td></tr><tr><td>model</td><td>车型编码，已脱敏</td></tr><tr><td>brand</td><td>汽车品牌，已脱敏</td></tr><tr><td>bodyType</td><td>车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7</td></tr><tr><td>fuelType</td><td>燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6</td></tr><tr><td>gearbox</td><td>变速箱：手动：0，自动：1</td></tr><tr><td>power</td><td>发动机功率：范围[0,600]</td></tr><tr><td>kilometer</td><td>汽车已行驶公里，单位万km</td></tr><tr><td>notRepairedData</td><td>汽车有尚未修复的损坏：是：0，否：1</td></tr><tr><td>regionCode</td><td>地区编码，已脱敏</td></tr><tr><td>seller</td><td>销售方：个体：0，非个体：1</td></tr><tr><td>offerType</td><td>报价类型：提供：0，请求：1</td></tr><tr><td>creatDate</td><td>汽车上线时间，即开始售卖时间</td></tr><tr><td>price</td><td>二手车交易价格（预测目标）</td></tr><tr><td>v系列特征</td><td>匿名特征，包含v0-14在内15个匿名特征</td></tr></table>

# 二、评测标准

评价标准为 MAE(MeanAbsoluteError)

若真实值为  $= (y_{1},y_{2},\dots ,y_{n})$  ，模型的预测值为  $\hat{y} = (\hat{y}_1,\hat{y}_2,\dots ,\hat{y}_n)$  ，那么该模型的MAE计算公式为

$$
MAE = \frac{\sum_{i = 1}^{n}\left|y_{i} - \hat{y}_{i}\right|}{n}.
$$

例如，真实值  $y = (15,20,12)$  ，预测值  $\hat{y} = (17,24,9)$  ，那么这个预测结果的MAE为

$$
MAE = \frac{|15 - 17| + |20 - 24| + |12 - 9|}{\mathfrak{J}} = 3.
$$

MAE 越小，结果越准确。

# 三、结果提交

提交前请确保预测结果的格式与 sample_submit.csv 中的格式一致，以及提交文件后缀名为 csv。

形式如下：

```makefile
SaleID,price
150000,687
150001,1250
150002,2580
150003,1178
```

