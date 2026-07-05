RAG_SYSTEM_PROMPT = """你是一个专业的物流调度算法专家，精通车辆路径问题（VRP）、自适应大邻域搜索（ALNS）、动态调度、车货匹配等物流优化领域知识。请根据以下参考资料回答用户问题。若参考资料不足，请明确告知。"""

RAG_USER_PROMPT_TEMPLATE = """参考资料：
{context}

用户问题：{question}

请给出专业、准确、结构清晰的回答："""


def build_rag_prompt(question: str, chunks: list) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("source", "未知来源")
        text = chunk.get("text", "")
        context_parts.append(f"[{i + 1}] ({source}) {text}")

    context = "\n\n".join(context_parts)
    return RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)


def build_rag_messages(question: str, chunks: list) -> list:
    user_prompt = build_rag_prompt(question, chunks)
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


JUDGE_SYSTEM_PROMPT = """你是一个严谨的物流领域专家和评测裁判。请对以下系统回答进行客观评分（1-5分，5分满分）。"""

JUDGE_USER_PROMPT_TEMPLATE = """请对以下物流领域的AI系统回答进行评分。

问题：{question}
参考答案：{reference}
系统回答：{system_answer}

请按以下维度打分：
1. 相关性（1-5分）：是否切题，有没有答非所问
2. 事实一致性（1-5分）：是否与公认物流知识冲突或存在幻觉
3. 完整性（1-5分）：是否覆盖了参考答案中的关键点

请以JSON格式输出评分结果，只输出JSON，不要有其他解释：
{{"relevance": int, "consistency": int, "completeness": int, "overall": float, "reason": "简短理由"}}"""


def build_judge_prompt(question: str, reference_answer: str, system_answer: str) -> str:
    return JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        reference=reference_answer,
        system_answer=system_answer,
    )


def build_judge_messages(question: str, reference_answer: str, system_answer: str) -> list:
    user_prompt = build_judge_prompt(question, reference_answer, system_answer)
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


DIRECT_QWEN_PROMPT_TEMPLATE = """你是一个专业的物流调度算法专家。请回答以下问题。

问题：{question}

请给出专业、准确、结构清晰的回答："""