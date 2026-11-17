# Hermes

> **Document type:** project entry point
> **Audience:** agents, developers, operators, reviewers
> **Reviewed:** 2026-11-17

> **线上量化交易执行服务**。Hermes 连接 QMT/xtquant、加载已配置策略、消费共享行情与模型产物，并通过统一订单协议执行交易。

| 项目属性 | 当前约定 |
|---|---|
| 运行形态 | Windows + QMT/xtquant |
| 正式入口 | `run.py` |
| 主配置 | `conf/config.ini` |
| 订单边界 | `execution/OrderManager`、`ManagedTrader`、网关与风险闸门 |
| 离线训练 | 不在 Hermes 内进行；归属 `D:/tool/pycharm_workspace/Hephaestus-Forge` |
| 生产数据 | `D:/tool/dataset` 及券商/QMT 运行环境 |

## 先读什么

代理、开发者和运维人员按以下顺序阅读：

1. [`AGENTS.md`](AGENTS.md)：项目约束、危险操作和验证边界。
2. [`docs/INDEX.md`](docs/INDEX.md)：文档地图和单一事实源。
3. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：运行时架构和组件职责。
4. [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)：配置层、策略启用和路径约定。
5. [`docs/OPERATIONS.md`](docs/OPERATIONS.md)：启动、监控、故障和变更流程。
6. [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)：安全开发和测试方式。

人工账户快照场景另读 [`docs/MANUAL_ACCOUNT_SNAPSHOT_RUNBOOK.md`](docs/MANUAL_ACCOUNT_SNAPSHOT_RUNBOOK.md)。

## 当前运行边界

当前生效的策略列表来自 `conf/config.ini` 的 `[strategy] name`，不要根据目录中存在的脚本推断策略已启用。配置中保留的历史策略段不会自动加载。

Hermes 可以消费以下外部产物：

- `D:/tool/dataset` 中的行情、股票池、因子和策略评分。
- `D:/tool/pycharm_workspace/Hephaestus-Forge` 生成的预测和 Top10 结果。
- `D:/software/atropos` 生成的部分评分文件。

Hermes 不承担模型训练、模型评估或大规模数据预处理。需要训练时进入 Hephaestus-Forge，并按其项目规则操作。

## 安装与运行边界

依赖清单位于 [`requirements.txt`](requirements.txt)。环境安装本身不会连接券商，但执行 `run.py`、计划任务、ETL 脚本或策略脚本可能读写生产数据并提交订单。

因此：

- 不要把 `python run.py` 当作冒烟测试。
- 不要在未核对账户、策略、`is_test`、数据日期和订单闸门前启动线上流程。
- 不要为了验证导入而加载会初始化 QMT、网络或账户的模块。
- 开发验证优先使用无外部副作用的单元测试、静态检查和临时 SQLite/假网关。

## 目录速览

```text
Hermes/
├─ run.py                 # 当前完整线上服务入口
├─ conf/                  # 主配置和策略配置
├─ execution/             # 订单协议、网关、风险、恢复、健康状态
├─ strategy/              # 策略实现
├─ etl/                   # 行情、股票池、因子和策略数据准备
├─ test/                  # 单元、集成和行情接口测试
├─ docs/                  # 项目文档和决策记录
├─ base_strategy*.py      # 策略基类
└─ scheduler.md           # 旧入口兼容说明，正文以 docs/SCHEDULER.md 为准
```

旧版 `incubate` 目录已整体迁移到：

```text
D:/tool/pycharm_workspace/Hephaestus-Forge/incubate
```

不要直接把它恢复到 Hermes，或直接取消注释旧 MLP 策略。恢复前必须重新设计导入路径、模型边界和线上依赖。

## 贡献与交接

提交前完成：

- 说明变更涉及的运行边界和数据路径。
- 更新受影响的架构、配置、运维或数据契约文档。
- 使用安全的测试范围验证订单幂等、风险拒单、恢复和闸门行为。
- 保留并说明工作区中与本次任务无关的用户改动。

涉及策略启用、账户、网关、风险、计划任务或共享数据的变更，需要在交接说明中单独标注，并由操作人员按运维文档复核。
