# Hermes 数据契约

> **Document type:** data and artifact contract
> **Audience:** developers, data-pipeline owners, operators
> **Authoritative sources:** ETL/strategy code and current producer configuration
> **Reviewed:** 2026-11-17

本文档说明 Hermes 消费和生成的主要产物。字段细节以实际生产脚本和读取代码为准；新增字段或改名时必须同步更新生产者、消费者和测试。

## 1. 生命周期分类

| 类别 | 示例 | 默认处理 |
|---|---|---|
| 原始输入 | QMT 行情、股票池、外部预测 | 不覆盖；按日期保留 |
| 中间产物 | 清洗数据、因子评分、缓存 | 由对应生产脚本管理，先确认引用再清理 |
| 交易状态 | SQLite 订单账本、持仓、Top10 批次 | 审计数据，禁止当缓存清理 |
| 日志 | 服务日志、策略日志、评分导出日志 | 按保留策略轮转，保留故障现场 |
| 模型/预测 | Hep checkpoint、预测 CSV | 外部项目产物，按版本和日期关联 |

## 2. 主要数据路径

### 行情和股票池

```text
D:/tool/dataset/quant_data
D:/tool/dataset/index_data
D:/tool/dataset/stock_pools.json
D:/tool/dataset/index_pools.json
```

`etl/2_*` 负责下载或更新行情，`etl/1_*` 负责股票池。文件日期和股票池版本必须覆盖策略所需交易日。

### 策略输入和状态

```text
D:/tool/dataset/strategy_data
D:/tool/dataset/strategy_positions
D:/tool/pycharm_workspace/Hermes/logs
```

策略评分、持仓、成交记录和批次状态可能被多个进程读取。清理前必须检查 Windows 计划任务、Hermes 进程和审计需要。

### Hephaestus-Forge 产物

Hermes 只消费 Hep 生成的结果，例如：

```text
D:/tool/pycharm_workspace/Hephaestus-Forge/inference_results/all_predictions.csv
D:/tool/pycharm_workspace/Hephaestus-Forge/top10_ranker/output/top10_<YYYYMMDD>_v8.csv
```

Hermes 不应修改 Hep 的 checkpoint、训练缓存或训练脚本。跨项目格式变更需要双方同步。

## 3. 订单和账户状态

### SQLite 订单账本

```text
D:/tool/dataset/strategy_positions/orders_<account_key>.sqlite3
```

订单意图、客户端订单 ID、券商订单、成交、状态转换和对账结果由 `execution/order_protocol.py` 管理。订单 ID 必须保持幂等；重复提交和重复成交事件不能创建第二个逻辑订单。

### 人工账户快照

人工快照的 schema、校验和停止条件见 [`MANUAL_ACCOUNT_SNAPSHOT_RUNBOOK.md`](MANUAL_ACCOUNT_SNAPSHOT_RUNBOOK.md)。快照必须带账户标识、带时区的 `as_of`、现金、权益和持仓字段。真实账户数据不得提交到仓库。

## 4. 文件命名约定

常见命名：

- `hs_quant_base_<YYYYMMDD>.json`：量价输入。
- `multi_factor_scores_<date>.json`：多因子评分。
- `dragon_v6_scores_<date>.json`：DragonV6 评分。
- `theme_momentum_scores_<date>.json`：题材动量评分。
- `top10_<YYYYMMDD>_v8.csv`：Hep Top10 输出。
- `orders_<account_key>.sqlite3`：账户订单账本。

日期格式、时区和交易日语义必须由生产者和消费者共同确认；不能仅凭文件修改时间判断数据有效性。

## 5. 兼容和变更规则

数据契约变更应说明：

1. 生产者和消费者。
2. 字段、类型、单位、时区和缺失值策略。
3. 是否兼容旧文件。
4. 迁移和回滚方法。
5. 保留和清理策略。
6. 验证样本和安全测试范围。

没有完成消费者迁移前，不要删除旧字段或改变文件名。不要把超大训练缓存复制进 Hermes 作为“临时输入”。
