# Hermes Windows 计划任务

> **Document type:** scheduler operations runbook
> **Audience:** operators and deployment maintainers
> **Source of truth:** Windows Task Scheduler plus task XML/configuration
> **Reviewed:** 2026-11-17

## 1. 调度边界

Windows 计划任务负责按时间启动数据准备、评分导出或外部项目任务；Hermes `run.py` 负责线上服务生命周期。两者可能共享 `D:/tool/dataset`，因此必须避免并发写同一个产物。

任务动作必须明确：

- Python 解释器路径。
- 脚本或模块路径。
- 工作目录。
- 输出目录和日志位置。
- 失败重试、超时和运行账户。

## 2. 当前任务类别

当前机器上的任务主要分为：

| 类别 | 典型任务 | 作用 |
|---|---|---|
| 股票池 | `1_run_index_pools`、`1_run_stock_pools` | 生成交易股票池 |
| 行情下载 | `2_rt_download_hs_*` | 更新实时/日行情输入 |
| 策略数据 | `3_*`、`prepare_multi_factor_data` | 生成策略评分或特征 |
| Hep 外部管线 | `DragonLeader_*`、Hep 日常预测任务 | 生成 Hermes 消费的外部产物 |

具体任务列表、状态和最后运行结果以 Task Scheduler 为准。仓库中的 XML 只是创建任务的输入，不能证明任务已注册或启用。

## 3. 安全创建/修改流程

1. 在 Task Scheduler 中导出当前任务 XML 和记录任务状态。
2. 比较解释器、参数、工作目录、运行账户、权限级别和触发时间。
3. 确认输出文件没有被其他任务或 Hermes 同时写入。
4. 先在隔离目录或 `is_test`/观察模式验证产物。
5. 注册或修改任务后，检查“上次运行结果”和实际输出日期。
6. 更新本文件、[`OPERATIONS.md`](OPERATIONS.md) 和相关数据契约。

不要通过“右键运行”把连接 QMT、写共享数据或生成订单的任务当作普通测试。任何手动触发都要先确认任务的副作用和当前交易时段。

## 4. 任务 XML 约定

仓库中的任务 XML 应包含：

- 清晰的 `<Description>`，说明输入、输出和副作用。
- 明确的 `<Command>`、`<Arguments>` 和 `<WorkingDirectory>`。
- 合理的超时、失败重试和“错过后尽快启动”策略。
- 最小必要权限，不使用无关的最高权限。
- 不把账户凭据写进 XML。

`etl/prepare_multi_factor_data_task.xml` 是多因子日常准备任务的示例配置；修改前先确认当前任务注册信息和输入日期。

## 5. 排障

按以下顺序核对：

1. 任务是否启用，运行账户是否仍有效。
2. 解释器、脚本和工作目录是否存在。
3. 输入目录是否有正确日期的文件。
4. 输出文件是否生成、大小是否合理、修改时间是否来自本次运行。
5. 日志中是否有异常或部分写入。
6. 是否与 Hermes、Hep 或其他 ETL 任务发生并发写入。

失败重试前先判断脚本是否已经写出部分产物，避免重复追加或覆盖。

## 6. 维护规则

- 任务改名、停用、改时区、改输出路径或改权限时，必须更新文档和交接记录。
- 任务失败不通过删除输出文件来“恢复正常”。
- 任务产生的生产数据不纳入仓库提交。
- 任务输入输出格式变化时，更新 [`DATA_CONTRACTS.md`](DATA_CONTRACTS.md)。
