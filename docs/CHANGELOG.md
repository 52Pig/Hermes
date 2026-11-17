# Hermes 文档变更记录

> **Document type:** documentation changelog
> **Audience:** maintainers, reviewers, agents
> **Reviewed:** 2026-11-17

文档变更记录采用 Keep a Changelog 风格。这里只记录文档和运行契约变化；产品行为变化应在代码变更和对应 ADR 中记录。

## [Unreleased]

### Added

- 增加项目级 `AGENTS.md`，定义 Agent 接手、开发、验证和生产安全边界。
- 增加 `docs/INDEX.md`，建立文档地图和单一事实源。
- 增加配置、运维、开发、数据契约和调度文档。
- 增加 Hermes 与 Hephaestus-Forge 边界、fail-closed 订单闸门两条 ADR。

### Changed

- 重写 README，改为线上执行服务的真实定位和安全入口。
- 重写架构文档、人工账户快照手册和调度说明。
- 将根目录 `scheduler.md` 保留为旧链接兼容入口。
- 将历史反转策略数据说明标记为 legacy，指向当前数据契约。

### Unchanged

- 本次只整理文档，没有修改交易代码、配置值、账户状态或生产数据。
- 没有启动 QMT、计划任务、ETL 或交易策略。
