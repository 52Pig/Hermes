# Hermes 文档中心

> **Document type:** documentation index
> **Audience:** agents, developers, operators, reviewers
> **Reviewed:** 2026-11-17

本文档是 Hermes 的入口。每份文档只维护一个主题；如果多个文档描述同一行为，应以“单一事实源”表为准，并在变更时同步更新引用。

## 快速导航

| 任务 | 首选文档 | 继续阅读 |
|---|---|---|
| Agent 接手项目 | [`../AGENTS.md`](../AGENTS.md) | 架构、开发指南 |
| 理解运行时 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | 配置、数据契约 |
| 修改配置或启用策略 | [`CONFIGURATION.md`](CONFIGURATION.md) | 运维、架构 |
| 启动、停机、故障处理 | [`OPERATIONS.md`](OPERATIONS.md) | 调度、账户快照 |
| 开发、测试、提交 | [`DEVELOPMENT.md`](DEVELOPMENT.md) | ADR、架构 |
| 判断文件和数据能否删除 | [`DATA_CONTRACTS.md`](DATA_CONTRACTS.md) | 运维、架构 |
| Windows 计划任务 | [`SCHEDULER.md`](SCHEDULER.md) | 配置、运维 |
| 人工账户快照 | [`MANUAL_ACCOUNT_SNAPSHOT_RUNBOOK.md`](MANUAL_ACCOUNT_SNAPSHOT_RUNBOOK.md) | 运维 |
| 重大边界决策 | [`decisions/`](decisions/) | 架构 |
| 文档变更 | [`CHANGELOG.md`](CHANGELOG.md) | 评审和交接 |

## 单一事实源

| 主题 | 权威来源 | 说明 |
|---|---|---|
| 当前启用策略 | `conf/config.ini` `[strategy] name` | 目录中存在的脚本不代表已启用 |
| 账户和客户端参数 | `conf/config.ini` `[client]` | 文档只能描述键和流程，不复制真实账户值 |
| 策略专属参数 | 对应 `conf/*.ini` | 以策略实际读取方式为准 |
| 订单状态和闸门 | `execution/order_protocol.py`、`execution/recovery.py` | 文档描述不应替代代码状态机 |
| 运行入口 | `run.py` | `main.py` 是旧的简化示例 |
| 生产数据路径 | 当前配置、ETL 和策略代码 | 绝对路径迁移前必须核对所有引用 |
| 离线训练 | Hephaestus-Forge | Hermes 只消费产物 |

## 文档规范

每份长期维护文档应包含：

- 文档类型、目标读者、适用范围和审阅日期。
- 明确的权威来源或依赖。
- 可执行的前置条件、停止条件和验证步骤。
- 生产数据、账户和外部系统的安全边界。
- 变更后需要同步更新的文件。

文档中的路径、策略名、配置键和命令必须能在当前工作区找到；无法验证的内容应标记为假设或待确认。

## 文档生命周期

- 代码、配置和文档发生行为变化时，同一变更内更新文档。
- 重大设计边界写入 `docs/decisions/`，不要只写在聊天或提交说明里。
- 过时内容保留兼容入口时，应明确标记为 legacy，并指向当前文档。
- 删除文档前先搜索仓库引用，避免破坏 Agent 或运维入口。
