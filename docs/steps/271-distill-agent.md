# 第11章 蒸馏与 Agent RL — 知识压缩与工具使用

## 学习目标

掌握知识蒸馏（模型压缩）和 Agent RL（工具调用强化学习）。

## 知识蒸馏

### 原理

用教师模型的**软标签**（soft logits）指导学生模型学习：

```
硬标签: [0, 0, 1, 0, 0]  ← 只知道正确答案
软标签: [0.05, 0.15, 0.60, 0.12, 0.08]  ← 还包含类间相似度（暗知识）
```

### 温度参数

Temperature T > 1 使分布更平滑，暴露更多暗知识：

```
T=1: [0.01, 0.02, 0.94, 0.02, 0.01]  ← 几乎和硬标签一样
T=3: [0.08, 0.14, 0.45, 0.18, 0.15]  ← 相似度信息更明显
```

### 蒸馏损失

```
Loss = α * CE(student, hard_labels) + (1-α) * T² * KL(teacher_soft || student_soft)
```

- α=0: 纯蒸馏（只用软标签）
- α=1: 纯 CE（和普通训练一样）
- 推荐 α=0.5, T=1.5

### 应用场景

- MoE → Dense 蒸馏（保留性能，降低推理成本）
- 大模型 → 小模型蒸馏（部署到边缘设备）

## Agent RL

### 工具调用流程

```
用户: "北京今天天气怎么样？"
  ↓
模型生成: ```json {"name": "get_current_weather", "arguments": {"location": "北京"}} ```
  ↓
execute_tool → {"city": "北京", "temperature": "28°C", "condition": "晴"}
  ↓
模型继续: "北京今天天气晴朗，气温28°C。"
```

### 多维度奖励

| 维度 | 条件 | 奖励 |
|------|------|------|
| 长度 | 20-800 字 | +0.5 / -0.5 |
| 工具调用 | 有效工具 | +1.0 |
| GT 匹配 | 答案正确 | +1.0 |
| 重复惩罚 | n-gram 重复 | -0~0.5 |

### 核心函数

| 函数 | 作用 |
|------|------|
| `parse_tool_calls(text)` | 从 ```json``` 块解析工具调用 |
| `execute_tool(name, args)` | 执行模拟工具 |
| `validate_gt_in_text(gt, resp)` | 验证 GT 是否在回复中 |
| `calculate_agent_reward(...)` | 多维度奖励 |

### 6 个模拟工具

calculate_math, unit_converter, get_current_weather, get_current_time, get_exchange_rate, translate_text

## 验证

```bash
pytest tests/m11_distill_agent/ -v   # 33 个测试全绿
```
