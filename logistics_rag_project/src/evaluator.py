import json
import csv
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

from .rag_pipeline import RAGPipeline
from .qa_chain import QwenChain, parse_judge_response
from .prompt_templates import build_judge_messages


class Evaluator:
    """
    自动化评测执行器，使用 LLM-as-a-Judge 对 RAG Pipeline 回答进行评分。
    支持细粒度诊断字段：检索文本、检索质量指标、难度分级。
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        judge_chain: QwenChain,
        test_set_path: Path,
        output_csv_path: Path,
    ):
        self.rag_pipeline = rag_pipeline
        self.judge_chain = judge_chain
        self.test_set_path = Path(test_set_path)
        self.output_csv_path = Path(output_csv_path)

    def load_test_set(self) -> List[Dict]:
        with open(self.test_set_path, "r", encoding="utf-8") as f:
            test_set = json.load(f)
        print(f"已加载测试集: {len(test_set)} 条用例")
        return test_set

    def run_evaluation(self, delay: float = 0.5) -> List[Dict]:
        test_set = self.load_test_set()
        results = []

        for item in tqdm(test_set, desc="评测进度"):
            qid = item.get("id", "unknown")
            question = item["question"]
            reference = item.get("reference_answer", "")
            category = item.get("category", "未分类")
            difficulty = item.get("difficulty", "未知")

            rag_result = self.rag_pipeline.query(question, stream=False)
            system_answer = rag_result["answer"]

            judge_messages = build_judge_messages(question, reference, system_answer)
            judge_response = self.judge_chain.generate(judge_messages)
            scores = parse_judge_response(judge_response)

            retrieval_scores = [round(s, 4) for s in rag_result["scores"]]
            retrieval_stats = self._compute_retrieval_stats(retrieval_scores)

            results.append({
                "id": qid,
                "category": category,
                "difficulty": difficulty,
                "question": question,
                "reference_answer": reference,
                "system_answer": system_answer,
                "relevance": scores.get("relevance", 0),
                "consistency": scores.get("consistency", 0),
                "completeness": scores.get("completeness", 0),
                "overall": scores.get("overall", 0.0),
                "judge_reason": scores.get("reason", ""),
                "retrieved_sources": [s["source"] for s in rag_result["sources"]],
                "retrieval_scores": retrieval_scores,
                "retrieval_max_score": retrieval_stats["max_score"],
                "retrieval_mean_score": retrieval_stats["mean_score"],
                "retrieval_std_score": retrieval_stats["std_score"],
                "retrieved_texts": [s["text"][:200] for s in rag_result["sources"]],
            })

            if delay > 0:
                time.sleep(delay)

        self._save_results(results)
        return results

    def _compute_retrieval_stats(self, scores: List[float]) -> Dict:
        if not scores:
            return {"max_score": 0.0, "mean_score": 0.0, "std_score": 0.0}
        arr = np.array(scores)
        return {
            "max_score": round(float(arr.max()), 4),
            "mean_score": round(float(arr.mean()), 4),
            "std_score": round(float(arr.std()), 4),
        }

    def _save_results(self, results: List[Dict]):
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "id", "category", "difficulty", "question", "reference_answer",
            "system_answer", "relevance", "consistency", "completeness",
            "overall", "judge_reason", "retrieved_sources", "retrieval_scores",
            "retrieval_max_score", "retrieval_mean_score", "retrieval_std_score",
            "retrieved_texts",
        ]
        with open(self.output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"评测结果已保存至: {self.output_csv_path}")

    def run_ablation(self, test_set: List[Dict] = None, delay: float = 0.5) -> Dict:
        """
        消融实验：对比纯Qwen（无RAG上下文）和 RAG-Qwen 的得分差异。
        """
        if test_set is None:
            test_set = self.load_test_set()

        rag_results = []
        direct_results = []

        for item in tqdm(test_set, desc="消融实验-RAG"):
            qid = item.get("id", "unknown")
            question = item["question"]
            reference = item.get("reference_answer", "")

            rag_result = self.rag_pipeline.query(question, stream=False)
            system_answer = rag_result["answer"]

            judge_messages = build_judge_messages(question, reference, system_answer)
            judge_response = self.judge_chain.generate(judge_messages)
            scores = parse_judge_response(judge_response)
            rag_results.append({
                "id": qid,
                "overall": scores.get("overall", 0.0),
                "relevance": scores.get("relevance", 0),
                "consistency": scores.get("consistency", 0),
                "completeness": scores.get("completeness", 0),
            })
            if delay > 0:
                time.sleep(delay)

        for item in tqdm(test_set, desc="消融实验-纯Qwen"):
            qid = item.get("id", "unknown")
            question = item["question"]
            reference = item.get("reference_answer", "")

            direct_answer = self.rag_pipeline.qwen_chain.generate_direct(question)

            judge_messages = build_judge_messages(question, reference, direct_answer)
            judge_response = self.judge_chain.generate(judge_messages)
            scores = parse_judge_response(judge_response)
            direct_results.append({
                "id": qid,
                "overall": scores.get("overall", 0.0),
                "relevance": scores.get("relevance", 0),
                "consistency": scores.get("consistency", 0),
                "completeness": scores.get("completeness", 0),
            })
            if delay > 0:
                time.sleep(delay)

        return {"rag": rag_results, "direct": direct_results}