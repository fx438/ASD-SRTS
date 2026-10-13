# 正文核心效率证据

三种分类器采用同一标准五折划分。AUC 统一报告为五折均值±样本标准差（ddof=1）；最大五折均值差为 **0.0039 < 0.004**。该比较说明三者判别性能在描述性层面接近，不构成统计等效性检验。

| 方法 | AUC（五折均值±样本标准差） | 参数量 | 同 CPU 推理时延（batch=1，中位数） | 原生部署时延（batch=1，中位数） | 峰值显存 | 原生推理设备 |
|---|---:|---:|---:|---:|---:|---|
| SRTS + LR | 0.7726 ± 0.0404 | 77,029 | 0.6319 ms | 0.6319 ms | N/A（无需 GPU） | Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz |
| SRTS + SVM | 0.7699 ± 0.0403 | 77,031 | 39.4690 ms | 39.4690 ms | N/A（无需 GPU） | Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz |
| SRTS + BrainNetMLP | 0.7687 ± 0.0465 | 2,477,634 | 2.3266 ms | 0.3393 ms | 20.853 MiB allocated / 24.000 MiB reserved | NVIDIA GeForce RTX 2080 Ti |

“同 CPU”列在同一 CPU、单核心和单线程环境下比较三种分类器：LR 与 SVM 的时延包含 `StandardScaler.transform + predict_proba`，BrainNetMLP 为包含 rFFT 的完整前向传播。“原生部署”列保留原始实现：LR/SVM 使用 scikit-learn CPU 推理，BrainNetMLP 使用 CUDA device-resident 前向传播（输入已在 GPU，不含主机到设备传输）。两列均不包含三种方法共享的 SRTS 表示预处理。由于原生部署列使用异构设备，其毫秒数仅用于呈现各自部署开销，不作硬件归一化速度倍数比较。LR 与 SVM 无需 GPU，因此峰值显存记为 `N/A（无需 GPU）`，而不是未经测量的 0 MB。

LR 参数量按 77,028 个特征系数加 1 个截距计算。线性 SVM 的 77,031 个概率推理有效参数包括 77,028 个 primal 系数、1 个截距和 2 个 Platt calibration 标量；本次锁定第 5 折的实际 `SVC.predict_proba` 部署仍保留 599 个支持向量，核心数值状态共 46,140,374 个标量。

## SRTS 表示预处理成本

以下成本在正文中单独报告，避免把共享表示构建成本隐藏在分类器时延中。

| 阶段 | 时间（均值±样本标准差） | 测量数 | 设备 |
|---|---:|---:|---|
| 860 人相关矩阵计算 | 0.0178 ± 0.0001 ms/人 | 10 | NVIDIA GeForce RTX 2080 Ti |
| 训练折参考矩阵拟合 | 18.2883 ± 0.0515 ms/折 | 5 | NVIDIA GeForce RTX 2080 Ti |
| 固定参考矩阵下单目标切线映射 | 13.3038 ± 0.0289 ms/人 | 5 | NVIDIA GeForce RTX 2080 Ti |
| 参考拟合+172目标切线映射（batch=32，摊销） | 13.1115 ± 0.3938 ms/人 | 5 | NVIDIA GeForce RTX 2080 Ti |

端到端 SRTS 行按每折训练集拟合参考矩阵，并对该折 172 个测试目标执行 batch=32 切线映射，再摊销至每个目标；参考矩阵未使用测试折。
