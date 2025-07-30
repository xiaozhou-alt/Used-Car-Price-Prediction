import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import seaborn as sns
import joblib
import os
import warnings
from datetime import datetime

# 忽略警告
warnings.filterwarnings('ignore')

# 创建输出目录
os.makedirs('./output', exist_ok=True)

# 全局LabelEncoder字典
global_label_encoders = {}

# 1. 数据加载与预处理
def load_and_preprocess(file_path, is_train=True):
    df = pd.read_csv(file_path, sep=' ')
    
    # 日期特征处理 - 处理无效日期
    for col in ['regDate', 'creatDate']:
        # 先将日期转换为字符串，然后处理无效值
        df[col] = df[col].astype(str)
        # 过滤无效日期（长度不为8或包含非数字字符）
        invalid_mask = (df[col].str.len() != 8) | (df[col].str.contains(r'\D'))
        
        # 对于无效日期，使用该列的中位数日期（转换为字符串）替换
        if is_train:
            median_date = df.loc[~invalid_mask, col].astype(int).median()
            median_date = str(int(median_date))
        else:
            # 对于测试集，使用训练集中计算的中位数日期
            median_date = "20120101"  # 默认值，实际应该从训练集计算
        
        df.loc[invalid_mask, col] = median_date
        
        # 转换为日期类型
        df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')
        
        # 提取日期特征
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
    
    # 计算车龄（年）
    df['vehicle_age'] = (df['creatDate'] - df['regDate']).dt.days / 365.25
    
    # 处理负车龄（注册日期晚于创建日期的情况）
    df.loc[df['vehicle_age'] < 0, 'vehicle_age'] = 0
    
    # 处理特殊值
    df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', np.nan)
    
    # 分类特征编码
    cat_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 
                   'notRepairedDamage', 'regionCode', 'seller', 'offerType']
    
    if is_train:
        # 训练集：拟合编码器
        for col in cat_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            global_label_encoders[col] = le
    else:
        # 测试集：使用训练集的编码器
        for col in cat_features:
            if col not in global_label_encoders:
                # 如果还没有训练编码器，创建一个临时的
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                global_label_encoders[col] = le
            else:
                le = global_label_encoders[col]
                # 处理测试集中可能出现的新类别
                known_categories = set(le.classes_)
                test_categories = set(df[col].astype(str).unique())
                
                # 将新类别标记为"unknown"
                new_categories = test_categories - known_categories
                if new_categories:
                    df.loc[df[col].astype(str).isin(new_categories), col] = 'unknown'
                
                # 只转换已知类别
                valid_mask = df[col].astype(str).isin(known_categories)
                df.loc[valid_mask, col] = le.transform(df.loc[valid_mask, col].astype(str))
                
                # 对于未知类别，使用最常见的类别编码
                if not valid_mask.all():
                    most_common = le.transform([le.classes_[0]])[0]
                    df.loc[~valid_mask, col] = most_common
    
    # 删除原始日期列
    df.drop(columns=['regDate', 'creatDate', 'name'], inplace=True)
    
    # 处理缺失值
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    return df

# 2. 特征工程
def feature_engineering(df):
    # 功率分桶 - 修复边界并安全转换
    df['power_bin'] = pd.cut(
        df['power'],
        bins=[-np.inf, 0, 100, 200, 300, 600, np.inf],
        labels=[0, 1, 2, 3, 4, 5],
        include_lowest=True
    ).astype('int64')
    
    # 处理可能的NaN值
    df['power_bin'].fillna(0, inplace=True)
    
    # 里程对数变换
    df['kilometer_log'] = np.log1p(df['kilometer'])
    
    # 匿名特征组合
    for i in range(5):
        df[f'v_{i}_squared'] = df[f'v_{i}'] ** 2
    
    # 创建日期相关特征
    df['creatDate_year_month'] = df['creatDate_year'] * 100 + df['creatDate_month']
    
    return df

# 3. 训练模型
def train_model(X_train, y_train, X_val, y_val):
    # 基本参数
    params = {
        'objective': 'regression_l1',  # MAE损失
        'boosting_type': 'gbdt',
        'metric': 'mae',
        'num_leaves': 63,  # 减少叶子数量防止过拟合
        'learning_rate': 0.01,  # 降低学习率
        'feature_fraction': 0.8,  # 增加正则化
        'bagging_fraction': 0.8,
        'bagging_freq': 10,
        'min_data_in_leaf': 50,  # 增加叶子最小样本
        'min_child_samples': 30,
        'reg_alpha': 0.8,  # 增强L1正则
        'reg_lambda': 0.8,  # 增强L2正则
        'max_depth': 7,  # 限制树深
        'n_estimators': 150000,  # 增加迭代次数
        'early_stopping_round': 1000,
        'verbosity': -1,
        'seed': 42
    }
    
    # 尝试使用GPU，如果失败则回退到CPU
    try:
        # 添加GPU参数
        gpu_params = params.copy()
        gpu_params.update({
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        })
        
        print("尝试使用GPU进行训练...")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            gpu_params,
            train_data,
            num_boost_round=5000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=1000, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        print("GPU训练成功完成！")
        return model
    except Exception as e:
        print(f"GPU训练失败: {e}")
        print("回退到CPU训练...")
        
        # 使用CPU参数
        cpu_params = params.copy()
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            cpu_params,
            train_data,
            num_boost_round=5000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        print("CPU训练成功完成！")
        return model

# 4. 特征重要性分析
def plot_feature_importance(model, features):
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 16))
    sns.barplot(x='Importance', y='Feature', data=importance.head(30))
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig('./output/feature_importance.png')
    plt.close()  # 避免在非交互式环境中显示
    
    return importance

# 5. 绘制训练历史
def plot_training_history(evals_result):
    plt.figure(figsize=(10, 6))
    
    # 提取训练和验证的MAE
    train_mae = evals_result['train']['l1']
    valid_mae = evals_result['valid']['l1']
    
    # 绘制曲线
    plt.plot(train_mae, label='Training MAE')
    plt.plot(valid_mae, label='Validation MAE')
    
    # 找到最小验证误差点
    if valid_mae:
        min_val_mae = min(valid_mae)
        min_idx = valid_mae.index(min_val_mae)
        plt.axvline(x=min_idx, color='r', linestyle='--', alpha=0.3)
    
    plt.title('Training History')
    plt.xlabel('Iterations')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/output/training_history.png')
    plt.close()

# 6. 主流程
def main():
    # 加载训练数据
    print("正在加载和预处理训练数据...")
    train_df = load_and_preprocess('/kaggle/input/used-car-price-prediction/data/used_car_train_20200313.csv', is_train=True)
    print("训练数据形状:", train_df.shape)
    
    print("正在进行特征工程...")
    train_df = feature_engineering(train_df)
    
    # 确保所有特征都是数值类型
    non_numeric_cols = train_df.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        print(f"发现非数值特征: {list(non_numeric_cols)}，正在进行转换...")
        for col in non_numeric_cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
    
    # 处理任何剩余的缺失值
    train_df.fillna(train_df.median(numeric_only=True), inplace=True)
    
    # 准备训练数据
    X = train_df.drop(columns=['price', 'SaleID'])
    y = train_df['price']
    
    print("特征矩阵形状:", X.shape)
    print("目标变量形状:", y.shape)
    
    # 划分数据集
    print("划分训练集和验证集...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    # 训练模型
    print("开始训练模型...")
    model = train_model(X_train, y_train, X_val, y_val)
    
    # 保存模型
    model_path = '/kaggle/working/output/car_price_model.pkl'
    joblib.dump(model, model_path)
    print(f"模型已保存到 {model_path}")
    
    # 评估模型
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    print(f'验证集 MAE: {mae:.2f}')
    
    # 特征重要性分析
    print("分析特征重要性...")
    feature_importance = plot_feature_importance(model, X.columns.tolist())
    feature_importance.to_csv('/kaggle/working/feature_importance.csv', index=False)
    
    # 随机样本验证
    print("随机选取10个样本进行验证...")
    sample_indices = np.random.choice(len(X_val), 10, replace=False)
    sample_data = X_val.iloc[sample_indices].copy()
    
    # 添加原始列名和值
    sample_data['true_price'] = y_val.iloc[sample_indices].values
    sample_data['pred_price'] = val_pred[sample_indices]
    
    # 添加SaleID（从原始数据获取）
    sample_data['SaleID'] = train_df.iloc[X_val.index[sample_indices]]['SaleID'].values
    
    # 保存样本数据
    sample_path = '/kaggle/working/sample_predictions.csv'
    sample_data.to_csv(sample_path, index=False)
    print(f"样本验证结果已保存到 {sample_path}")
    
    # 打印样本结果
    print("\n样本验证结果:")
    print(sample_data[['SaleID', 'true_price', 'pred_price']])
    
    # 处理测试数据
    print("\n处理测试数据...")
    test_df = load_and_preprocess('/kaggle/input/used-car-price-prediction/data/used_car_testB_20200421.csv', is_train=False)
    test_df = feature_engineering(test_df)
    
    # 确保所有特征都是数值类型
    non_numeric_cols = test_df.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        print(f"发现非数值特征: {list(non_numeric_cols)}，正在进行转换...")
        for col in non_numeric_cols:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    
    # 处理任何剩余的缺失值
    test_df.fillna(test_df.median(numeric_only=True), inplace=True)
    
    test_ids = test_df['SaleID']
    X_test = test_df.drop(columns=['SaleID'])
    
    print("测试数据形状:", X_test.shape)
    
    # 预测测试集
    print("预测测试集...")
    test_pred = model.predict(X_test)
    
    # 保存预测结果
    submission_path = '/kaggle/working/submission.csv'
    submission = pd.DataFrame({'SaleID': test_ids, 'price': test_pred})
    submission.to_csv(submission_path, index=False)
    print(f"预测结果已保存到 {submission_path}")
    
    # 绘制训练历史
    if hasattr(model, 'evals_result_'):
        plot_training_history(model.evals_result_)
        print("训练历史图表已保存")
    
    print("\n训练完成！所有结果已保存到 ./output 目录")

if __name__ == '__main__':
    main()