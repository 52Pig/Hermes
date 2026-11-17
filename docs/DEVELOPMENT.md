# Hermes 开发指南

> **Document type:** development guide
> **Audience:** developers and coding agents
> **Depends on:** [`../AGENTS.md`](../AGENTS.md), [`ARCHITECTURE.md`](ARCHITECTURE.md)
> **Reviewed:** 2026-11-17

## 1. 开发前检查

1. 在仓库根目录运行 `git status --short`。
2. 检查现有 diff，区分用户改动、未跟踪实验文件、生成物和本次任务。
3. 判断任务属于策略、执行、ETL、配置、文档还是外部项目集成。
4. 查找现有测试和调用方，再决定修改位置。
5. 对账户、订单、共享数据和计划任务相关任务，先阅读 [`OPERATIONS.md`](OPERATIONS.md)。

不要为了“整理干净”而重置、删除或覆盖工作区中的现有改动。

## 2. 代码边界

| 目录 | 责任 | 不应放入 |
|---|---|---|
| `execution/` | 订单状态、网关、风险、恢复、健康和审计 | 策略选股规则、训练脚本 |
| `strategy/` | 策略决策、持仓策略状态、策略级输入消费 | 原始券商连接管理 |
| `etl/` | 股票池、行情、因子和策略输入准备 | 统一订单状态机 |
| `conf/` | 机器和运行配置 | 账户凭据、长篇业务逻辑 |
| `test/` | 单元、集成和接口测试 | 真实订单验证 |
| `docs/` | 约束、契约、运行手册和架构决策 | 临时日志、生产快照 |

策略应保留动态加载约定：配置中的 `name` 对应脚本中的类名，类提供 `do(accounts)`。策略下单必须走 `OrderManager` 或 `ManagedTrader`。

## 3. 推荐变更流程

### 3.1 设计

- 明确输入、输出、失败行为和回滚方式。
- 对订单或账户状态变化，先画出状态转移或补充 ADR。
- 对跨 Hermes/Hep/Atropos 的改动，列出每个外部边界、路径和产物格式。
- 不把“训练”“推理”“执行”混为一个模块；离线训练继续留在 Hephaestus-Forge。

### 3.2 实现

- 保持幂等：重复事件、重复成交、重复调度不能产生重复订单或破坏状态。
- 保持 fail-closed：连接、对账、校验或风险检查失败时拒绝提交。
- 避免模块导入时连接 QMT、读取账户、调用网络或创建订单。
- 用配置传入机器路径；不要把新的绝对路径散落在策略代码中。
- 生产日志不得打印账户敏感信息或完整原始数据。

### 3.3 文档同步

| 变更 | 必须同步 |
|---|---|
| 启动顺序、模块职责、外部边界 | `docs/ARCHITECTURE.md` |
| 配置键、策略启用、路径 | `docs/CONFIGURATION.md` |
| 启停、排障、计划任务 | `docs/OPERATIONS.md`、`docs/SCHEDULER.md` |
| 文件格式、产物和保留策略 | `docs/DATA_CONTRACTS.md` |
| 长期架构/安全边界 | `docs/decisions/` ADR |
| Agent 约束 | `AGENTS.md` |

## 4. 验证矩阵

| 验证 | 适用场景 | 备注 |
|---|---|---|
| Markdown 链接/路径检查 | 文档变更 | 必须运行 |
| `git diff --check` | 所有文本改动 | 必须运行 |
| Python 语法编译 | Python 代码变更 | 不会连接 QMT，但仍需检查生成副作用 |
| 临时 SQLite + 假网关 | 订单、恢复、风险变更 | 首选 |
| 定向 pytest | execution 纯单元测试 | 先阅读测试内容 |
| QMT/行情/网络集成测试 | 需要真实环境 | 不得在普通开发验证中盲目运行 |
| 全量测试 | 发布前专门窗口 | 需明确外部依赖和账户隔离 |

优先选择：

- `test/test_order_protocol.py`
- `test/test_managed_trader.py`
- `test/test_multi_gateway.py`
- `test/test_recovery.py`
- `test/test_risk.py`
- `test/test_portfolio_pipeline.py`
- `test/test_portfolio_shadow.py`

不要把需要 QMT、行情服务、网络或真实账户的测试当作默认单元测试。

## 5. 提交和交接清单

- [ ] 工作区原有改动未被覆盖。
- [ ] 变更范围与用户请求一致。
- [ ] 没有绕过订单协议、风险检查或恢复闸门。
- [ ] 没有把离线训练移入 Hermes。
- [ ] 文档和配置引用已更新。
- [ ] 已运行安全的验证，并记录跳过的 live/integration 检查。
- [ ] 说明了数据迁移、配置变更和回滚步骤。
- [ ] 没有提交账户 ID、凭据、生产快照或生成的大文件。
