# ADR-0001: Hermes 与 Hephaestus-Forge 的职责边界

- **Status:** Accepted
- **Date:** 2026-11-17
- **Scope:** 训练、推理产物和线上交易执行的项目边界

## Context

Hermes 曾保留离线训练实验、模型定义和大规模预处理缓存。线上执行项目和训练项目混在一起会造成：部署体积膨胀、Agent 误判运行边界、线上启动误触训练、模型依赖难以审计。

## Decision

- Hermes 负责线上策略调度、账户连接、订单协议、风险、恢复和产物消费。
- Hephaestus-Forge 负责离线训练、模型评估和训练数据处理。
- Hermes 可以调用 Hep 的明确脚本或读取其版本化输出，但不复制训练缓存，也不在启动流程中训练。
- 旧 Hermes `incubate` 目录迁移到 `D:/tool/pycharm_workspace/Hephaestus-Forge/incubate`，恢复前必须重新设计导入和部署边界。

## Alternatives considered

1. 保留两个项目中的训练代码：部署和维护边界不清，拒绝。
2. 把训练代码复制进 Hermes 的隐藏目录：仍会造成误加载和空间浪费，拒绝。
3. 只删除旧代码不留归档：不利于追溯，拒绝。

## Consequences

- Hermes 更适合线上部署和审计。
- 跨项目需要维护文件格式、路径和版本兼容。
- 旧 MLP 策略不能直接取消注释启用。

## Validation and rollback

验证 Hermes 当前活动策略、启动前外部调用和 Hep 输出路径；如需恢复旧实验，先在 Hep 中建立隔离环境并完成离线验证，不直接恢复到线上仓库。
