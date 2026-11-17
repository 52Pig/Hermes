# Hermes 配置参考

> **Document type:** configuration reference
> **Audience:** developers, operators, reviewers
> **Authoritative sources:** `conf/config.ini`, `conf/top10_v8_v1.ini`, strategy code
> **Reviewed:** 2026-11-17

## 1. 配置原则

- 配置文件决定运行行为；文档不替代配置文件。
- 真实账户值、QMT 私有路径和生产凭据不写入文档或示例。
- 修改账户、策略集合、网关、风险或测试模式前，必须走 [`OPERATIONS.md`](OPERATIONS.md) 的变更流程。
- 只修改已授权的配置段；未出现在 `[strategy] name` 中的策略不会被 `run.py` 加载。
- `run.py` 的配置观察只更新已存在策略实例的配置对象，不能安全地完成策略集合、账户或网关的热切换。此类变更应重启并重新对账。

## 2. 主配置文件

文件：`conf/config.ini`

### `[project]`

保存历史工程路径。它不是当前服务入口的运行目录来源；迁移机器时不要只修改这一项而假设所有路径已迁移。

### `[client]`

| 键 | 作用 | 约束 |
|---|---|---|
| `mini_path` | 兼容/默认 QMT 客户端路径 | 真实路径只在本机配置 |
| `acc_dict` | 账户键、账户标识和 `userdata_mini` 路径映射 | 修改后必须重启并重新对账 |
| `is_test` | 全局客户端测试标记 | 不等于每个策略的测试开关 |

`run.py` 使用 `ast.literal_eval` 解析 `acc_dict`，不要改成任意代码执行方式。

### `[logging]`

| 键 | 作用 |
|---|---|
| `log_level` | 日志级别 |
| `log_format` | 日志格式 |
| `log_file` | `TimedRotatingFileHandler` 的主日志路径 |
| `log_backup_count` | 保留的轮转文件数量 |

### `[strategy]`

| 键 | 作用 |
|---|---|
| `base_dir` | 策略持仓和成交记录的根目录 |
| `name` | 逗号分隔的已启用策略 section 名称 |

当前工作区的生效值应以配置文件为准；不要将本段的历史注释行当成活动配置。

### 策略 section

每个被 `[strategy] name` 引用的 section 至少需要：

```ini
[ExampleStrategy]
name = ExampleStrategy
path = strategy/example_strategy.py
config_file = conf/example.ini
interval = 60
```

- `name` 必须匹配脚本中的类名。
- `path` 由动态导入加载，通常相对于 Hermes 工作目录。
- `config_file` 会被 `StrategyManagementService` 读取为策略配置。
- `interval` 单位是秒。

## 3. Top10 独立测试开关

文件：`conf/top10_v8_v1.ini`

`Top10V8V1` 有自己的 `is_test`。它与 `[client] is_test` 独立：

- `is_test=1`：策略不使用 managed order manager 提交真实订单，适合观察和验证。
- `is_test=0`：进入真实订单路径前，必须完成账户连接、启动对账、风险和闸门检查。

文件中还配置 Python 解释器、Hephaestus-Forge 根目录、Top-K、持有天数、现金缓冲、再平衡时间和管线超时。变更这些项时要同时检查 `strategy/top10_v8_v1.py`。

## 4. 执行配置的代码默认值

当前 `config.ini` 没有显式 `[execution]` 段。`run.py` 因此使用代码默认值，例如：

- `enabled=True`
- `default_channel=mini_qmt`
- `legacy_stock_channel=mini_qmt`
- `risk_enabled=True`
- `max_order_equity_fraction=0.25`
- `max_gross_equity_fraction=1.0`
- `max_margin_equity_fraction=0.40`
- `max_order_quantity=1000000`

如果这些值需要成为可审计的部署契约，应补充显式配置段并同时更新测试、运维文档和变更记录；不要只在文档中宣称配置已存在。

## 5. 路径和产物

常见路径如下：

| 路径 | 用途 | 生命周期 |
|---|---|---|
| `D:/tool/dataset/quant_data` | QMT 量价输入 | 由 ETL/任务维护 |
| `D:/tool/dataset/strategy_data` | 因子和策略状态/评分 | 被策略消费或更新 |
| `D:/tool/dataset/strategy_positions` | 订单账本、持仓和成交记录 | 交易审计数据 |
| `D:/tool/dataset/server_log` | 服务日志 | 轮转和审计 |
| `Hermes/logs` | 部分策略评分产物 | 由策略/外部任务维护 |
| `Hephaestus-Forge/inference_results` | Hep 推理结果 | 外部项目产物 |

删除、移动或清空这些目录前，必须先确认没有活动任务、订单恢复或审计依赖。
