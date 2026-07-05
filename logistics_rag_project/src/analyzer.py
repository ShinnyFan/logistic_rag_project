import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats


plt.rcParams["axes.unicode_minus"] = False


def _setup_chinese_font():
    """尝试设置中文字体"""
    chinese_fonts = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei", "Noto Sans CJK SC", "STSong",
        "PingFang SC", "Heiti SC", "AR PL UMing CN",
    ]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
            return font
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return None


class Analyzer:
    """
    结果统计分析器：计算统计指标、识别低分案例、自动归因诊断、生成可视化图表。
    """

    RETRIEVAL_WEAK_THRESHOLD = 0.5
    RETRIEVAL_STRONG_THRESHOLD = 0.7

    def __init__(self, csv_path: Path, charts_dir: Path):
        self.csv_path = Path(csv_path)
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None
        _setup_chinese_font()

    def load_results(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig")
        print(f"已加载评测结果: {len(self.df)} 条记录")
        return self.df

    def compute_statistics(self) -> Dict:
        if self.df is None:
            self.load_results()

        score_cols = ["relevance", "consistency", "completeness", "overall"]
        stats_dict = {}
        for col in score_cols:
            if col in self.df.columns:
                series = self.df[col].dropna()
                stats_dict[col] = {
                    "mean": round(series.mean(), 2),
                    "median": round(series.median(), 2),
                    "std": round(series.std(), 2),
                    "min": round(series.min(), 2),
                    "max": round(series.max(), 2),
                }

        print("\n=== 评测统计 ===")
        for col, s in stats_dict.items():
            print(f"{col}: 均值={s['mean']}, 中位数={s['median']}, "
                  f"标准差={s['std']}, 范围=[{s['min']}, {s['max']}]")

        return stats_dict

    def compute_category_stats(self) -> pd.DataFrame:
        if self.df is None:
            self.load_results()

        if "category" not in self.df.columns:
            print("数据中无 category 字段，跳过分类统计")
            return pd.DataFrame()

        score_cols = ["relevance", "consistency", "completeness", "overall"]
        available_cols = [c for c in score_cols if c in self.df.columns]
        category_stats = self.df.groupby("category")[available_cols].mean().round(2)

        print("\n=== 按分类统计平均分 ===")
        print(category_stats.to_string())

        return category_stats

    def find_low_score_cases(self, threshold: float = 3.0) -> pd.DataFrame:
        if self.df is None:
            self.load_results()

        if "overall" in self.df.columns:
            low_score = self.df[self.df["overall"] < threshold]
        else:
            score_cols = ["relevance", "consistency", "completeness"]
            available = [c for c in score_cols if c in self.df.columns]
            if available:
                avg_score = self.df[available].mean(axis=1)
                low_score = self.df[avg_score < threshold]
            else:
                low_score = pd.DataFrame()

        print(f"\n=== 低分案例 (overall < {threshold}) ===")
        if len(low_score) == 0:
            print("无低分案例")
        else:
            display_cols = ["id", "category", "question", "overall"]
            display_cols = [c for c in display_cols if c in low_score.columns]
            print(low_score[display_cols].to_string())

        return low_score

    def analyze_by_difficulty(self) -> pd.DataFrame:
        if self.df is None:
            self.load_results()

        if "difficulty" not in self.df.columns:
            print("数据中无 difficulty 字段，跳过难度分析")
            return pd.DataFrame()

        score_cols = ["relevance", "consistency", "completeness", "overall"]
        available = [c for c in score_cols if c in self.df.columns]
        difficulty_order = ["简单", "中等", "困难", "对抗"]
        difficulty_stats = self.df.groupby("difficulty")[available].mean().round(2)
        difficulty_stats = difficulty_stats.reindex(
            [d for d in difficulty_order if d in difficulty_stats.index]
        )

        print("\n=== 按难度统计平均分 ===")
        print(difficulty_stats.to_string())

        if "retrieval_max_score" in self.df.columns:
            retrieval_by_diff = self.df.groupby("difficulty")["retrieval_max_score"].mean().round(4)
            retrieval_by_diff = retrieval_by_diff.reindex(
                [d for d in difficulty_order if d in retrieval_by_diff.index]
            )
            print("\n=== 按难度统计平均最大检索相似度 ===")
            print(retrieval_by_diff.to_string())

        return difficulty_stats

    def diagnose_low_score_cases(self, threshold: float = 3.0) -> List[Dict]:
        """
        低分案例自动归因诊断。
        使用决策树逻辑判断每个低分案例的根本原因：
          - 类型1: 检索失败（检索分数低，正确内容未被检索到）
          - 类型2: 模型幻觉/偏离上下文（检索到内容但事实一致性低）
          - 类型3: 缺失关键要点（检索到部分内容但完整性低）
          - 类型4: 知识库覆盖盲区（检索内容与问题不相关，所有维度都低）
        """
        if self.df is None:
            self.load_results()

        low_score = self.find_low_score_cases(threshold)
        if len(low_score) == 0:
            return []

        diagnoses = []
        for _, row in low_score.iterrows():
            retrieval_max = row.get("retrieval_max_score", 0)
            relevance = row.get("relevance", 0)
            consistency = row.get("consistency", 0)
            completeness = row.get("completeness", 0)

            if retrieval_max < self.RETRIEVAL_WEAK_THRESHOLD:
                root_cause = "类型1: 检索失败 — 正确知识点未被检索到，检索最高相似度仅{:.3f}".format(retrieval_max)
                suggestion = "建议：检查术语匹配、chunk切分策略、Embedding模型适配性，或增大top_k"
            elif consistency <= 2 and retrieval_max >= self.RETRIEVAL_STRONG_THRESHOLD:
                root_cause = "类型2: 模型幻觉/偏离上下文 — 检索到高分内容但事实一致性低({})".format(consistency)
                suggestion = "建议：增强系统prompt对检索内容的约束力，降低temperature，或检查模型是否过度依赖预训练知识"
            elif completeness <= 2 and relevance >= 3:
                root_cause = "类型3: 缺失关键要点 — 检索到部分内容但完整性低({})，可能跨chunk信息未综合".format(completeness)
                suggestion = "建议：增大top_k、减小chunk_size、增加chunk_overlap，或采用多轮检索策略"
            elif relevance <= 2 and consistency <= 2 and completeness <= 2:
                root_cause = "类型4: 知识库覆盖盲区 — 检索内容与问题不相关，所有维度均低"
                suggestion = "建议：扩充知识库覆盖范围，或优化系统prompt引导模型诚实告知知识盲区"
            else:
                root_cause = "混合原因 — 检索max={:.3f}, rel={}, con={}, comp={}".format(
                    retrieval_max, relevance, consistency, completeness)
                suggestion = "建议：综合排查检索质量和生成质量"

            diagnosis = {
                "id": row.get("id", ""),
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", ""),
                "question": row.get("question", ""),
                "overall": row.get("overall", 0),
                "relevance": relevance,
                "consistency": consistency,
                "completeness": completeness,
                "retrieval_max_score": retrieval_max,
                "root_cause": root_cause,
                "suggestion": suggestion,
                "judge_reason": row.get("judge_reason", ""),
            }
            diagnoses.append(diagnosis)

        print(f"\n=== 低分案例归因诊断 ({len(diagnoses)} 条) ===")
        for d in diagnoses:
            print(f"\n[{d['id']}] {d['difficulty']} | {d['category']}")
            print(f"  问题: {d['question'][:60]}...")
            print(f"  得分: overall={d['overall']}, rel={d['relevance']}, con={d['consistency']}, comp={d['completeness']}")
            print(f"  检索最高相似度: {d['retrieval_max_score']}")
            print(f"  根因: {d['root_cause']}")
            print(f"  {d['suggestion']}")
            print(f"  裁判评语: {d['judge_reason'][:100]}")

        return diagnoses

    def compute_retrieval_quality_report(self) -> Dict:
        """
        检索质量综合报告：计算所有案例的检索分数统计、弱检索案例比例等。
        """
        if self.df is None:
            self.load_results()

        report = {}

        if "retrieval_max_score" in self.df.columns:
            scores = self.df["retrieval_max_score"].dropna()
            report["retrieval_max_mean"] = round(scores.mean(), 4)
            report["retrieval_max_median"] = round(scores.median(), 4)
            report["weak_retrieval_ratio"] = round(
                (scores < self.RETRIEVAL_WEAK_THRESHOLD).mean(), 4
            )
            report["strong_retrieval_ratio"] = round(
                (scores >= self.RETRIEVAL_STRONG_THRESHOLD).mean(), 4
            )

        print("\n=== 检索质量报告 ===")
        for k, v in report.items():
            print(f"{k}: {v}")

        return report

    def ablation_analysis(self, ablation_data: Dict) -> Dict:
        """
        消融实验分析：对比 RAG vs 纯Qwen，使用配对T检验和箱线图。
        """
        rag_data = ablation_data.get("rag", [])
        direct_data = ablation_data.get("direct", [])

        if not rag_data or not direct_data:
            print("消融数据为空，跳过分析")
            return {}

        rag_df = pd.DataFrame(rag_data)
        direct_df = pd.DataFrame(direct_data)

        rag_scores = rag_df["overall"].values
        direct_scores = direct_df["overall"].values

        min_len = min(len(rag_scores), len(direct_scores))
        rag_scores = rag_scores[:min_len]
        direct_scores = direct_scores[:min_len]

        t_stat, p_value = stats.ttest_rel(rag_scores, direct_scores)
        rag_mean = np.mean(rag_scores)
        direct_mean = np.mean(direct_scores)
        improvement = rag_mean - direct_mean
        improvement_pct = (improvement / direct_mean * 100) if direct_mean > 0 else 0

        print("\n=== 消融实验分析 ===")
        print(f"RAG-Qwen 平均分: {rag_mean:.2f}")
        print(f"纯Qwen 平均分:   {direct_mean:.2f}")
        print(f"提升幅度:        {improvement:+.2f} ({improvement_pct:+.1f}%)")
        print(f"配对T检验:       t={t_stat:.3f}, p={p_value:.4f}")

        self._plot_ablation_boxplot(rag_scores, direct_scores)
        self._plot_ablation_comparison(rag_df, direct_df)

        return {
            "rag_mean": rag_mean,
            "direct_mean": direct_mean,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "t_statistic": t_stat,
            "p_value": p_value,
        }

    def generate_all_charts(self):
        if self.df is None:
            self.load_results()

        self._plot_score_distribution()
        self._plot_category_radar()
        self._plot_retrieval_score_analysis()
        self._plot_difficulty_analysis()
        self._plot_retrieval_vs_score()
        print(f"\n所有图表已保存至: {self.charts_dir}")

    def _plot_score_distribution(self):
        score_cols = ["relevance", "consistency", "completeness", "overall"]
        available = [c for c in score_cols if c in self.df.columns]
        if not available:
            return

        fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
        if len(available) == 1:
            axes = [axes]

        for ax, col in zip(axes, available):
            self.df[col].dropna().hist(ax=ax, bins=10, edgecolor="black", alpha=0.7)
            ax.set_title(f"{col} 分布")
            ax.set_xlabel("分数")
            ax.set_ylabel("频次")

        plt.tight_layout()
        path = self.charts_dir / "score_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {path}")

    def _plot_category_radar(self):
        if "category" not in self.df.columns:
            return

        cat_stats = self.compute_category_stats()
        if cat_stats.empty:
            return

        score_cols = ["relevance", "consistency", "completeness"]
        available = [c for c in score_cols if c in cat_stats.columns]
        if len(available) < 2:
            return

        categories = cat_stats.index.tolist()
        n_cat = len(categories)
        n_scores = len(available)
        angles = np.linspace(0, 2 * np.pi, n_scores, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for i, cat in enumerate(categories):
            values = cat_stats.loc[cat, available].values.tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=cat)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available)
        ax.set_ylim(0, 5.5)
        ax.set_title("各分类评分雷达图")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

        path = self.charts_dir / "category_radar.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {path}")

    def _plot_retrieval_score_analysis(self):
        if "retrieval_scores" not in self.df.columns:
            return

        all_scores = []
        for val in self.df["retrieval_scores"].dropna():
            try:
                if isinstance(val, str):
                    scores = json.loads(val.replace("'", '"'))
                else:
                    scores = val
                all_scores.extend(scores)
            except (json.JSONDecodeError, TypeError):
                continue

        if not all_scores:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(all_scores, bins=20, edgecolor="black", alpha=0.7)
        ax.set_title("检索相似度分数分布")
        ax.set_xlabel("余弦相似度")
        ax.set_ylabel("频次")
        ax.axvline(np.mean(all_scores), color="red", linestyle="--",
                   label=f"均值: {np.mean(all_scores):.3f}")
        ax.legend()

        path = self.charts_dir / "retrieval_score_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {path}")

    def _plot_ablation_boxplot(self, rag_scores: np.ndarray, direct_scores: np.ndarray):
        fig, ax = plt.subplots(figsize=(8, 6))
        data = [rag_scores, direct_scores]
        bp = ax.boxplot(data, labels=["RAG-Qwen", "纯Qwen"], patch_artist=True)
        bp["boxes"][0].set_facecolor("#4ECDC4")
        bp["boxes"][1].set_facecolor("#FF6B6B")

        for i, scores in enumerate(data):
            jitter = np.random.normal(i + 1, 0.04, size=len(scores))
            ax.scatter(jitter, scores, alpha=0.5, color="black", s=20)

        ax.set_title("消融实验: RAG-Qwen vs 纯Qwen")
        ax.set_ylabel("Overall 分数")
        ax.set_ylim(0, 5.5)

        path = self.charts_dir / "ablation_boxplot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {path}")

    def _plot_ablation_comparison(self, rag_df: pd.DataFrame, direct_df: pd.DataFrame):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ["relevance", "consistency", "completeness"]

        for ax, metric in zip(axes, metrics):
            if metric in rag_df.columns and metric in direct_df.columns:
                rag_vals = rag_df[metric].dropna().values
                direct_vals = direct_df[metric].dropna().values
                min_len = min(len(rag_vals), len(direct_vals))
                x = np.arange(min_len)
                width = 0.35
                ax.bar(x - width / 2, rag_vals[:min_len], width, label="RAG-Qwen", color="#4ECDC4")
                ax.bar(x + width / 2, direct_vals[:min_len], width, label="纯Qwen", color="#FF6B6B")
                ax.set_title(metric)
                ax.set_ylim(0, 5.5)
                ax.legend()

        plt.suptitle("消融实验: 各维度对比")
        plt.tight_layout()
        path = self.charts_dir / "ablation_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {path}")

    def _plot_difficulty_analysis(self):
        if "difficulty" not in self.df.columns:
            return

        score_cols = ["relevance", "consistency", "completeness", "overall"]
        available = [c for c in score_cols if c in self.df.columns]
        if not available:
            return

        difficulty_order = ["简单", "中等", "困难", "对抗"]
        difficulty_stats = self.df.groupby("difficulty")[available].mean()
        difficulty_stats = difficulty_stats.reindex(
            [d for d in difficulty_order if d in difficulty_stats.index]
        )

        if difficulty_stats.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(available))
        width = 0.2
        colors = ["#4ECDC4", "#FF6B6B", "#FFE66D", "#95E1D3"]

        for i, (diff, row) in enumerate(difficulty_stats.iterrows()):
            offset = (i - len(difficulty_stats) / 2 + 0.5) * width
            ax.bar(x + offset, row.values, width, label=diff, color=colors[i % len(colors)])

        ax.set_xticks(x)
        ax.set_xticklabels(available)
        ax.set_ylim(0, 5.5)
        ax.set_title("不同难度评分对比")
        ax.legend()
        ax.set_ylabel("平均分")

        path = self.charts_dir / "difficulty_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {path}")

    def _plot_retrieval_vs_score(self):
        if "retrieval_max_score" not in self.df.columns or "overall" not in self.df.columns:
            return

        valid = self.df[["retrieval_max_score", "overall", "difficulty"]].dropna()
        if valid.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        difficulty_colors = {"简单": "#4ECDC4", "中等": "#FFE66D", "困难": "#FF6B6B", "对抗": "#95E1D3"}
        for diff in valid["difficulty"].unique():
            subset = valid[valid["difficulty"] == diff]
            color = difficulty_colors.get(diff, "#999999")
            ax.scatter(subset["retrieval_max_score"], subset["overall"],
                       label=diff, color=color, s=80, alpha=0.7, edgecolors="black")

        ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="低分阈值 (3.0)")
        ax.axvline(x=self.RETRIEVAL_WEAK_THRESHOLD, color="orange", linestyle="--", alpha=0.5,
                   label="弱检索阈值 ({})".format(self.RETRIEVAL_WEAK_THRESHOLD))

        ax.set_xlabel("检索最高相似度")
        ax.set_ylabel("Overall 评分")
        ax.set_title("检索质量 vs 回答质量")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 5.5)
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = self.charts_dir / "retrieval_vs_score.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存: {path}")