---
tags:
  - paper
arxiv_id: 2506.08718v1
published: 2025-06-10
authors: Juan Plazuelo Pascual, Carlos Tardon Rubio, Juan Toro Cebada, Angel Hernando Veciana
category: 
---

# Price Discovery in Cryptocurrency Markets

## 基本信息
- **arXiv ID:** [2506.08718v1](http://arxiv.org/abs/2506.08718v1)
- **作者:** [[Juan Plazuelo Pascual]], [[Carlos Tardon Rubio]], [[Juan Toro Cebada]], [[Angel Hernando Veciana]]
- **发布时间:** 2025-06-10
- **分类:** 

<details>
<summary>详细摘要</summary>

## 论文信息
- **标题：** Price Discovery in Cryptocurrency Markets
- **作者：** Juan Plazuelo Pascual, Carlos Tardón Rubio, Juan Toro Cebada, Ángel Hernando Veciana
- **发布时间：** 2025年6月10日

## 研究领域
- **其他**（金融计量经济学 / 量化金融 / 区块链与DeFi分析）
    *注：该论文属于金融学与数据科学的交叉领域，主要应用统计学方法分析加密货币市场价格发现机制，不属于计算机视觉、NLP等典型的AI子领域。*

## 核心问题
本论文旨在解决**加密货币市场中价格发现机制的动态过程**这一核心问题。具体而言，它探讨了以下两个主要议题：
1.  **中心化交易所（CEX，如Binance）与去中心化交易所（DEX，如Uniswap）之间的信息效率与领先-滞后关系**：即在价格形成过程中，是链上市场还是链下市场主导了信息的融入。
2.  **现货市场与期货市场（CME BTC期货）之间的价格发现贡献**：分析在不同市场压力和波动性条件下，信息如何在这些市场间流动。

## 主要创新点
1.  **链上订单簿重构与流动性快照构建**：
    提出了一套详细的方法论，将基于自动做市商（AMM）的流动性池数据（特别是Uniswap v2和v3）转化为可与传统订单簿（CEX）对比的形式。论文深入解释了如何处理链上数据，包括Tick数据处理和流动性快照的构建，以实现DEX与CEX数据的同质化比较。
2.  **多方法融合的实证分析框架**：
    在加密货币领域，综合运用了**Hasbrouck信息份额（IS）**、**Gonzalo-Granger永久-短暂分解**以及**Hayashi-Yoshida相关性度量**等多种高级计量经济学方法，从长周期和短窗口（特别是高波动性事件）两个维度，全面评估了市场的价格发现效率。
3.  **DeFi作为预言机的绩效评估与套利分析**：
    不仅分析了价格发现，还结合Gas费等交易成本，评估了AMM作为去中心化预言机的有效性，并量化了中心化与去中心化市场之间的套利机会，为理解DeFi市场的实际运作效率提供了实证依据。

## 方法概述
论文采用**比较实证分析**的研究方法，主要包含以下步骤：
1.  **数据获取与处理**：
    *   **链下数据**：获取Binance等中心化交易所的高频交易数据和订单簿数据。
    *   **链上数据**：通过节点直接访问以太坊区块链，提取Uniswap v2/v3的交易日志和流动性池状态。针对Uniswap v3，论文详细解释了基于Tick的流动性集中机制，并构建了流动性快照来模拟订单簿深度。
2.  **理论模型应用**：
    *   **协整检验与VECM模型**：验证不同市场（如Binance vs Uniswap）间的价格序列是否存在长期均衡关系，并建立向量误差修正模型（VECM）。
    *   **价格发现度量**：应用**Hasbrouck信息份额**分析各市场对共同有效价格变动的贡献度；利用**Gonzalo-Granger分解**区分价格变动中的永久成分（Permanent Component，反映长期信息）和短暂成分（Transitory Component，反映市场摩擦）。
    *   **相关性分析**：使用**Hayashi-Yoshida**方法处理非同步交易数据，精确计算市场间的领先-滞后关系。

## 实验结果
基于对ETH（Binance vs Uniswap v2）和BTC（Spot vs CME Futures）的实证分析，得出以下关键结论：
1.  **CEX主导价格发现**：在ETH的分析中，中心化交易所（Binance）普遍在价格发现过程中占据主导地位，即Binance的价格变动往往领先于Uniswap。
2.  **期货市场领先但波动期趋同**：在BTC的分析中，期货市场（CME）通常领先于现货市场，但在高波动性期间（如2024年5月和8月的特定事件），两者的领先-滞后关系变得混合，差异不如CEX与DEX之间显著。
3.  **市场效率与套利**：研究发现存在明显的

</details>