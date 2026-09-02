"""
CLongEval Benchmark Prompts
===========================

Chinese prompts for the CLongEval (港中文长对话记忆评测) benchmark.
Based on LCvMem sub-task from CLongEval by OpenLMLab.

Category mapping (derived from question types):
    1 = single-hop (单跳跃)
    2 = multi-hop (多跳跃)
    3 = temporal (时间推理)
    4 = conversation-understanding (对话理解)
"""

# ===============================================================================
# CATEGORY NAME MAPPING
# ===============================================================================

CATEGORY_NAMES = {
    1: "single-hop",
    2: "multi-hop",
    3: "temporal",
    4: "conversation-understanding",
}

CATEGORIES_TO_EVALUATE = [1, 2, 3, 4]


# ===============================================================================
# ANSWER GENERATION PROMPT (Chinese)
# ===============================================================================

ANSWER_GENERATION_PROMPT = """你是一个精确的问答助手。根据以下检索到的记忆内容回答问题。

## 记忆内容
{memories}

## 问题
{question}

## 参考日期
最后一次对话发生在 {reference_date}。请据此进行时间推理。

## 要求
1. 仔细阅读所有记忆，逐条检查，找到最相关的信息
2. 如果问题提到某个日期（如"4月27日"），优先查找与该日期相关的记忆
3. 如果需要多步推理，请连接不同记忆中的事实
4. 如果需要时间计算，基于参考日期进行推理
5. 记忆中可能包含答案，也可能需要从多条记忆中推断
6. 直接给出答案，不要说"无法确定"——尝试从记忆中推断最可能的答案
7. 用中文回答

答案：
"""


# ===============================================================================
# JUDGE PROMPT (Chinese)
# ===============================================================================

JUDGE_SYSTEM_PROMPT = "你是一个严格的答案评判员。判断生成的答案是否正确。只返回有效的 JSON。"

JUDGE_PROMPT = """判断以下问题的答案是否正确。

问题: {question}
标准答案: {gold_answer}
生成答案: {generated_answer}

评判规则:
1. **部分正确也算对**: 生成的答案包含标准答案中的至少一个正确内容，即算正确
2. **同义词算对**: 同一概念的不同表达方式算正确（如"巧克力蛋糕"="巧克力味的蛋糕"）
3. **额外细节不扣分**: 比标准答案更详细不算错，只要核心事实正确
4. **日期容错**: 日期相差14天以内算正确。时间跨度相差50%以内算正确
5. **语义优先**: 判断语义一致性，而非字面匹配

只有当生成的答案完全没有正确内容或完全偏题时才判错。

只返回 JSON:
{{
  "label": "CORRECT" 或 "WRONG",
  "reasoning": "简要说明判断理由"
}}
"""


def get_answer_generation_prompt(question: str, memories: list[dict], reference_date: str = "") -> str:
    """Build the answer generation prompt from search results."""
    memories_text = "\n".join(
        f"{i+1}. {m.get('memory', '')}" for i, m in enumerate(memories)
    )
    return ANSWER_GENERATION_PROMPT.format(
        memories=memories_text,
        question=question,
        reference_date=reference_date,
    )


def get_judge_prompt(question: str, gold_answer: str, generated_answer: str) -> str:
    """Build the judge prompt."""
    return JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )
