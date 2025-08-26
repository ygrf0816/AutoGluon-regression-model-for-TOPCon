# AutoGluon Regression Model for TOPCon

## Project Overview

This is paper "Machine Learning Guided Device-Level Design for High-Efficiency Tunnel Oxide Passivating Contact Solar Cells" related code files. Related models and dependence plot can be seen in https://data.mendeley.com/drafts/s2vnyfgc4s

You can cite us with:

C. Zhang, Z. Yang, Y. Zhang, et al. “ Machine Learning Guided Device-Level Design for High-Efficiency Tunnel Oxide Passivating Contact Solar Cells.” Small (2025): e06958. https://doi.org/10.1002/smll.202506958

<img src=".\figure\TOC.png"  >

This project builds a framework for forward prediction and reverse design of n-type TOPCon (Tunnel Oxide Passivated Contact) solar cells based on machine learning methods. The framework combines tools such as AutoGluon, LightGBM, and genetic algorithms to achieve:

1. **Forward Prediction**: Predicting output performance (such as Voc, Jsc, FF, Eff, etc.) based on material and structural parameters of the cell
2. **Feature Importance Analysis**: Analyzing the influence of key parameters on cell performance using SHAP (SHapley Additive exPlanations) method
3. **Reverse Design**: Using genetic algorithms to optimize and find the best parameter combinations that can achieve target performance

This framework can help researchers quickly evaluate the performance of different design schemes and guide the optimization design process of TOPCon cells, reducing the number of experiments and costs. 


## Environment Configuration

### Basic Environment

- Python 3.10.16
- CUDA (if GPU acceleration is needed)

### Installation of Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

- autogluon.tabular: For automated machine learning modeling

- geatpy: Genetic algorithm toolkit

- lightgbm: Gradient boosting decision tree framework

- shap: For model interpretability analysis

- pytorch: Deep learning framework

- pandas, numpy, matplotlib: For data processing and visualization

  

### Notes on SHAP Library Modifications

We modified the `_scatter.py` file in the SHAP library to enable convenient modification of chart labels and add shadow effects. The modified `_scatter.py` file is included in the project folder. You can:

1. Directly replace the corresponding file in the original SHAP library

2. Or delete the relevant code according to your needs

   

## Code File Description

### Core Files

- **auto_final.py**: Trains regression models for forward prediction of various performance parameters of TOPCon cells (Vm, Im, Voc, Jsc, FF, Eff)

- **Note**: auto_final.py was designed to use GPU acceleration; it may report errors when called in CPU mode

- **ga_auto.py**: Uses genetic algorithms for reverse design, finding optimal parameter combinations

- **importance_LGBM.py**: Trains a LightGBM classifier for SHAP analysis and parameter importance evaluation

- **func.py**: A collection of auxiliary functions used in the project, including custom neural network models, plotting functions, etc.

  

### Data Folders

- **dataset/**: Stores training and testing datasets

- **Models/**: Stores trained model files 

- **output/**: Stores output results

- **figure/**: Stores generated charts

  

## Usage Instructions

    Please modify hyperparameters directly in the code

### 1. Forward Prediction Model Training

```bash
python auto_final.py
```

This script will train multiple regression models to predict various performance parameters of the cell. After training, the models will be saved in the `Models/final/` directory.

### 2. Feature Importance Analysis

```bash
python importance_LGBM.py
```

This script will train a LightGBM classifier and use SHAP to analyze the influence of each parameter on cell performance. The generated visualization results will be saved in the `figure/` directory.

### 3. Reverse Design Optimization

```bash
python ga_auto.py
```

This script uses genetic algorithms to find the optimal parameter combinations that can achieve specified performance goals. The optimization results will be saved in the `ga_result/` directory.



## Implementation Principles

### Forward Prediction

Uses AutoGluon automated machine learning framework to train regression models for predicting cell output performance based on TOPCon cell structure and material parameters (such as thickness, doping concentration, interface state density, etc.).

### Feature Importance Analysis

1. Uses LightGBM to train a classifier that categorizes cells by performance (e.g., high efficiency, medium efficiency, low efficiency, etc.)
2. Uses SHAP method to analyze the contribution of each parameter to the classification results
3. Generates visualization results such as Beeswarm plots, Summary plots, and Dependence plots to intuitively show parameter importance

### Reverse Design

1. Defines optimization objectives (such as maximizing FF, Eff, etc.)
2. Uses genetic algorithms to find the optimal parameter combinations
3. During the optimization process, uses trained forward prediction models to evaluate the performance of parameter combinations
4. Continuously optimizes parameter combinations through genetic operations such as crossover and mutation



## Result Interpretation

### Forward Prediction Results

After model training, prediction models for various performance parameters will be generated. Evaluation metrics include R², MAE, MSE, etc., used to measure the prediction accuracy of the models.

### Feature Importance Analysis Results

- **Beeswarm Plots**: Show the distribution of SHAP values for each parameter for specific categories
- **Summary Plots**: Show the average absolute SHAP values of each parameter, reflecting overall importance
- **Dependence Plots**: Show the interaction effects between parameters on prediction results

### Reverse Design Results

After optimization, the best parameter combinations and their predicted performance will be output and saved as CSV files. At the same time, convergence curve plots of the optimization process will be generated, showing the change of objective function values with iteration numbers.



------



# AutoGluon Regression Model for TOPCon

## 项目概述

这个项目是论文“Machine Learning Guided Device-Level Design for High-Efficiency Tunnel Oxide Passivating Contact Solar Cells”的具体实现。相关的模型以及论文中提到的所有参数的dependence plot可在此下载： https://data.mendeley.com/drafts/s2vnyfgc4s

我们的论文引用为：

C. Zhang, Z. Yang, Y. Zhang, et al. “ Machine Learning Guided Device-Level Design for High-Efficiency Tunnel Oxide Passivating Contact Solar Cells.” Small (2025): e06958. https://doi.org/10.1002/smll.202506958

<img src=".\figure\TOC.png"  >

本项目基于机器学习方法，构建了n型TOPCon（Tunnel Oxide Passivated Contact）太阳能电池的前向预测和反向设计框架。该框架结合了AutoGluon、LightGBM和遗传算法等工具，实现了：

1. **前向预测**：根据电池的材料和结构参数，预测其输出性能（如Voc、Jsc、FF、Eff等）
2. **特征重要性分析**：利用SHAP（SHapley Additive exPlanations）方法分析关键参数对电池性能的影响
3. **反向设计**：基于遗传算法，反向优化设计寻找能实现目标性能的最佳参数组合

该框架可以帮助研究人员快速评估不同设计方案的性能，并指导TOPCon电池的优化设计过程，减少实验次数和成本。

## 环境配置

### 基础环境

- Python 3.10.16
- CUDA（如需GPU加速）

### 依赖库安装

```bash
pip install -r requirements.txt
```

主要依赖库包括：

- autogluon.tabular：用于自动化机器学习建模
- geatpy：遗传算法工具包
- lightgbm：梯度提升决策树框架
- shap：用于模型可解释性分析
- pytorch：深度学习框架
- pandas, numpy, matplotlib：数据处理与可视化

### SHAP库修改说明

我们修改了SHAP库中的`_scatter.py`文件，以实现便捷地修改图的标签并添加阴影效果。修改后的`_scatter.py`文件已包含在项目文件夹中，您可以：

1. 直接替换原SHAP库中的对应文件
2. 或者根据自己的需求删除相关代码

## 代码文件说明

### 核心文件

- **auto_final.py**：训练回归模型，用于前向预测TOPCon电池的各种性能参数（Vm, Im, Voc, Jsc, FF, Eff）
- **注**： auto_final.py进行设计时使用了GPU加速，CPU模式进行调用时可能会报错
- **ga_auto.py**：使用遗传算法进行反向设计，寻找电池参数的最优组合
- **importance_LGBM.py**：训练LightGBM分类器，进行SHAP分析，评估参数重要性
- **func.py**：项目中使用的辅助函数集合，包括自定义神经网络模型、绘图函数等

### 数据文件夹

- **dataset/**：存放训练和测试数据集
- **Models/**：存放训练好的模型文件
- **output/**：存放输出结果
- **figure/**：存放生成的图表



## 使用方法

    超参数设置请直接在代码中进行修改

### 1. 前向预测模型训练

```bash
python auto_final.py
```

此脚本将训练多个回归模型，用于预测电池的各种性能参数。训练完成后，模型将保存在`Models/final/`目录下。

### 2. 特征重要性分析

```bash
python importance_LGBM.py
```

此脚本将训练LightGBM分类器，并使用SHAP分析各参数对电池性能的影响。生成的可视化结果将保存在`figure/`目录下。

### 3. 反向设计优化

```bash
python ga_auto.py
```

此脚本使用遗传算法寻找能实现指定性能目标的最佳参数组合。优化结果将保存在`ga_result/`目录下。

## 实现原理

### 前向预测

使用AutoGluon自动机器学习框架，基于TOPCon电池的结构和材料参数（如厚度、掺杂浓度、界面态密度等），训练回归模型预测电池的输出性能。

### 特征重要性分析

1. 使用LightGBM训练分类器，将电池按性能分为多个类别（如高效率、中等效率、低效率等）
2. 利用SHAP方法分析各参数对分类结果的贡献
3. 生成Beeswarm图、Summary图和Dependence图等可视化结果，直观展示参数重要性

### 反向设计

1. 定义优化目标（如最大化FF、Eff等）
2. 使用遗传算法寻找最佳参数组合
3. 在优化过程中，使用训练好的前向预测模型评估参数组合的性能
4. 通过交叉、变异等遗传操作不断优化参数组合



## 结果解读

### 前向预测结果

模型训练完成后，将生成各性能参数的预测模型，评估指标包括R²、MAE、MSE等，用于衡量模型预测精度。

### 特征重要性分析结果

- **Beeswarm图**：展示各参数对特定类别的SHAP值分布
- **Summary图**：展示各参数的平均绝对SHAP值，反映整体重要性
- **Dependence图**：展示参数之间的交互作用对预测结果的影响

### 反向设计结果

优化完成后，将输出最佳参数组合及其预测性能，结果保存为CSV文件。同时，还会生成优化过程的收敛曲线图，展示目标函数值随迭代次数的变化。
