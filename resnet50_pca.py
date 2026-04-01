import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, f1_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], 'GPU')
print("Available devices:", tf.config.list_physical_devices())

# 数据路径设置
folders = {
    "good": r"/Users/risetto/Documents/MATLAB/1550/小鼠卵泡数据集/健康组/001_res/amp",
    "bad": r"/Users/risetto/Documents/MATLAB/1550/小鼠卵泡数据集/健康组/002_res/amp"
}
labels_mapping = {"good": 0, "bad": 1}

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 全局字体设置
rcParams['axes.titlesize'] = 30
rcParams['axes.labelsize'] = 24
rcParams['xtick.labelsize'] = 24
rcParams['ytick.labelsize'] = 24
rcParams['legend.fontsize'] = 16
rcParams['axes.titleweight'] = 'bold'
rcParams['axes.labelweight'] = 'bold'

# 新参数设置
IMAGE_SIZE = (224, 224)  # ResNet标准输入尺寸
PCA_COMPONENTS = 2       # PCA降维维度

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = tf.keras.applications.resnet50.preprocess_input(img)  # ResNet专用预处理
            images.append(img)
            labels.append(label)
    print(f"从 {folder} 加载了 {len(images)} 张图像")
    return np.array(images), np.array(labels)

def build_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)  # 添加全局池化
    return Model(inputs=base_model.input, outputs=x)

def extract_resnet_features(model, images):
    features = model.predict(images, batch_size=32)
    return features  # 输出形状为(n_samples, 2048)

def evaluate_clustering(original_data, labels):
    silhouette = silhouette_score(original_data, labels) if len(np.unique(labels)) > 1 else -1
    calinski_harabasz = calinski_harabasz_score(original_data, labels) if len(np.unique(labels)) > 1 else -1
    davies_bouldin = davies_bouldin_score(original_data, labels) if len(np.unique(labels)) > 1 else -1
    return silhouette, davies_bouldin, calinski_harabasz

def calculate_acc(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    D = max(true_labels.max(), predicted_labels.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=np.int64)
    for i in range(len(true_labels)):
        cost_matrix[true_labels[i], predicted_labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / len(true_labels)

# 加载数据
all_images = []
true_labels = []
for folder_name, folder_path in folders.items():
    images, labels = load_images(folder_path, labels_mapping[folder_name])
    all_images.extend(images)
    true_labels.extend(labels)

# all_images = np.array(all_images)
all_images = np.array(all_images, dtype=np.float32)
true_labels = np.array(true_labels)
print("\n总加载图像数量:", len(all_images))

# 特征处理流程
resnet_model = build_resnet_model()
features = extract_resnet_features(resnet_model, all_images)

# 标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 定义一个函数来评估不同PCA维度下的聚类性能
def evaluate_pca_dimensions(features_scaled, true_labels, n_clusters, pca_dimensions_range):
    acc_results = {'K-means': [], 'Hierarchical': [], 'GMM': []}

    for n_components in pca_dimensions_range:
        pca_reducer = PCA(n_components=n_components)
        features_pca = pca_reducer.fit_transform(features_scaled)

        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(features_pca)
        labels_kmeans = kmeans.labels_
        acc_kmeans = calculate_acc(true_labels, labels_kmeans)
        acc_results['K-means'].append(acc_kmeans)

        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters).fit(features_pca)
        labels_hierarchical = hierarchical.labels_
        acc_hierarchical = calculate_acc(true_labels, labels_hierarchical)
        acc_results['Hierarchical'].append(acc_hierarchical)

        # GMM
        gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(features_pca)
        labels_gmm = gmm.predict(features_pca)
        acc_gmm = calculate_acc(true_labels, labels_gmm)
        acc_results['GMM'].append(acc_gmm)

    return acc_results

# 设置PCA维度范围
pca_dimensions_range = list(range(2, 11, 1))  # 从2到10以1为步长
n_clusters = 2  # 聚类数

# 评估不同PCA维度下的ACC
acc_results = evaluate_pca_dimensions(features_scaled, true_labels, n_clusters, pca_dimensions_range)

# 将折线图数据导出到Excel
def export_to_excel(acc_results, dimensions, file_path, sheet_name):
    df = pd.DataFrame(acc_results, index=dimensions)
    df.index.name = 'PCA Dimensions'
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name)

# 绘制ACC随PCA维度变化的折线图
def plot_acc_vs_pca_dimensions(acc_results, dimensions):
    plt.figure(figsize=(10, 8))

    for method, accs in acc_results.items():
        plt.plot(dimensions, accs, marker='o', label=method)

    plt.xlabel('PCA Dimensions', fontsize=24, fontweight='bold')
    plt.ylabel('ACC', fontsize=24, fontweight='bold')
    plt.grid(False)
    plt.ylim(0, 1)
    plt.xlim(dimensions[0] - 0.5, dimensions[-1] + 0.5)
    plt.xticks(dimensions, fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')

    ax = plt.gca()
    ax.spines['left'].set_position(('data', dimensions[0] - 0.5))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_bounds(0, 1)
    ax.spines['bottom'].set_bounds(dimensions[0] - 0.5, dimensions[-1] + 0.5)

    legend = plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

# 绘制折线图
plot_acc_vs_pca_dimensions(acc_results, pca_dimensions_range)

# 导出数据到Excel
export_to_excel(acc_results, pca_dimensions_range, r"./result/Resnet-pca.xlsx", 'Results')

# 使用PCA进行降维
pca = PCA(n_components=PCA_COMPONENTS)
features_pca = pca.fit_transform(features_scaled)

# 聚类分析
n_clusters = 2

# 使用ResNet特征进行聚类
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42).fit(features_pca)
labels_kmeans_pca = kmeans_pca.labels_
hierarchical_pca = AgglomerativeClustering(n_clusters=n_clusters).fit(features_pca)
labels_hierarchical_pca = hierarchical_pca.labels_
gmm_pca = GaussianMixture(n_components=n_clusters, random_state=42).fit(features_pca)
labels_gmm_pca = gmm_pca.predict(features_pca)

# 计算评价指标
def evaluate_and_print_results(title, features, labels_kmeans, labels_hierarchical, labels_gmm, true_labels):
    print(f"{title}:")

    # K-means
    kmeans_results = evaluate_clustering(features, labels_kmeans)
    acc_kmeans = calculate_acc(true_labels, labels_kmeans)
    f1_kmeans = f1_score(true_labels, labels_kmeans, average='macro')
    print("K-means聚类:")
    print(f"轮廓系数: {kmeans_results[0]:.4f}, DB指数: {kmeans_results[1]:.4f}, CH指数: {kmeans_results[2]:.4f}")
    print(f"ACC: {acc_kmeans:.4f}, F1分数: {f1_kmeans:.4f}")

    # Hierarchical
    hierarchical_results = evaluate_clustering(features, labels_hierarchical)
    acc_hierarchical = calculate_acc(true_labels, labels_hierarchical)
    f1_hierarchical = f1_score(true_labels, labels_hierarchical, average='macro')
    print("Hierarchical聚类:")
    print(f"轮廓系数: {hierarchical_results[0]:.4f}, DB指数: {hierarchical_results[1]:.4f}, CH指数: {hierarchical_results[2]:.4f}")
    print(f"ACC: {acc_hierarchical:.4f}, F1分数: {f1_hierarchical:.4f}")

    # GMM
    gmm_results = evaluate_clustering(features, labels_gmm)
    acc_gmm = calculate_acc(true_labels, labels_gmm)
    f1_gmm = f1_score(true_labels, labels_gmm, average='macro')
    print("GMM聚类:")
    print(f"轮廓系数: {gmm_results[0]:.4f}, DB指数: {gmm_results[1]:.4f}, CH指数: {gmm_results[2]:.4f}")
    print(f"ACC: {acc_gmm:.4f}, F1分数: {f1_gmm:.4f}")

    return {
        'K-means': {'silhouette': kmeans_results[0], 'db': kmeans_results[1], 'ch': kmeans_results[2], 'acc': acc_kmeans, 'f1': f1_kmeans},
        'Hierarchical': {'silhouette': hierarchical_results[0], 'db': hierarchical_results[1], 'ch': hierarchical_results[2], 'acc': acc_hierarchical, 'f1': f1_hierarchical},
        'GMM': {'silhouette': gmm_results[0], 'db': gmm_results[1], 'ch': gmm_results[2], 'acc': acc_gmm, 'f1': f1_gmm}
    }

# 使用ResNet特征聚类的结果
results = evaluate_and_print_results("PCA", features_pca,
                                     labels_kmeans_pca, labels_hierarchical_pca, labels_gmm_pca, true_labels)

# 绘制散点图
def plot_clusters(features, labels, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(title, fontsize=30, fontweight='bold')
    plt.xticks(fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close()

# 绘制散点图
plot_clusters(features_pca, labels_kmeans_pca, "K-means")
plot_clusters(features_pca, labels_hierarchical_pca, "Hierarchical")
plot_clusters(features_pca, labels_gmm_pca, "GMM")

