原理图如下：
```mermaid
graph TD
    subgraph 数据预处理模块 (Data Preprocessing)
        A1[sample_remove.py<br>清洗/剔除异常样本] --&gt; A2[sample_rename.py<br>样本规范化重命名]
        A2 --&gt; A3[sample_split.py<br>划分训练集/验证集/测试集]
    end

    subgraph 全局配置与辅助 (Config &amp; Utils)
        B1((config.py<br>超参数与路径配置))
        B2((utils.py<br>通用辅助工具函数))
    end

    subgraph 数据加载模块 (Data Loading)
        C1[dataset.py<br>单模态/普通数据加载]
        C2[muti_dataset.py<br>多分支/多模态数据加载]
    end

    subgraph 模型构建模块 (Model Architecture)
        D1[model.py<br>基础网络结构]
        D2[resnet50_pca.py<br>ResNet50特征提取 + PCA降维]
    end

    subgraph 核心执行模块 (Execution Pipeline)
        E1{train.py<br>模型训练脚本}
        E2{predict.py<br>模型推理预测}
    end

    A3 --&gt; C1
    A3 --&gt; C2
    C1 --&gt; E1
    C2 --&gt; E1
    D1 --&gt; E1
    D2 --&gt; E1
    D1 --&gt; E2
    D2 --&gt; E2

    B1 -.参数注入.-&gt; C1
    B1 -.参数注入.-&gt; C2
    B1 -.参数注入.-&gt; D1
    B1 -.参数注入.-&gt; D2
    B1 -.参数注入.-&gt; E1
    B1 -.参数注入.-&gt; E2

    B2 -.函数调用.-&gt; E1
    B2 -.函数调用.-&gt; E2
```
