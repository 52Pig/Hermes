# Architecture Decision Records

> **Document type:** decision-record index
> **Audience:** maintainers, reviewers, agents
> **Reviewed:** 2026-11-17

本目录记录会影响长期架构、运行安全或跨项目边界的决定。每条 ADR 只描述一个决定，并保留被取代的历史记录。

## 状态

- **Proposed**：讨论中，尚未作为约束执行。
- **Accepted**：当前采用。
- **Superseded**：已被新的 ADR 取代。
- **Rejected**：明确不采用。

## 新增 ADR 模板

```markdown
# ADR-NNNN: 标题

- Status: Proposed
- Date: YYYY-MM-DD
- Owners: 维护人或团队

## Context

## Decision

## Alternatives considered

## Consequences

## Validation and rollback
```

涉及订单放行、账户连接、数据迁移、训练/推理/执行边界或外部项目接口时，应新增 ADR，并链接到受影响的架构/运维文档。
