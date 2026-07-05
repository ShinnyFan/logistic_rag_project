import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import config
from src.pdf_loader import load_documents
from src.chunker import chunk_documents, save_chunks, load_chunks
from src.embedder import create_embedder
from src.vector_store import VectorStore
from src.retriever import create_retriever
from src.qa_chain import QwenChain
from src.rag_pipeline import RAGPipeline
from src.evaluator import Evaluator
from src.analyzer import Analyzer


def cmd_build_index(args):
    """构建知识库索引：PDF → 文本 → 切块 → Embedding → FAISS索引"""
    print("=" * 60)
    print("步骤 1/4: 加载PDF文件")
    print("=" * 60)
    data_dir = config.RAW_PDFS_DIR
    documents = load_documents(data_dir)
    if not documents:
        print(f"错误: {data_dir} 目录下未找到PDF或TXT文件，请将文档放入该目录后重试")
        return

    print(f"\n步骤 2/4: 文本切块 (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    chunks = chunk_documents(
        documents,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    if not chunks:
        print("错误: 未生成有效文本块")
        return
    save_chunks(chunks, config.CHUNKS_PATH, config.CHUNKS_METADATA_PATH)

    print(f"\n步骤 3/4: 生成Embedding向量 (模式: {config.EMBEDDING_MODE})")
    embedder = create_embedder(
        mode=config.EMBEDDING_MODE,
        model_name=config.LOCAL_EMBEDDING_MODEL_NAME,
        device=config.LOCAL_EMBEDDING_DEVICE,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        api_model=config.QWEN_EMBEDDING_MODEL,
    )
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress=True, normalize=True)

    print(f"\n步骤 4/4: 构建FAISS索引")
    vector_store = VectorStore(dim=embeddings.shape[1])
    vector_store.build_index(chunks, embeddings)
    vector_store.save(config.FAISS_INDEX_PATH, config.ID_MAP_PATH)

    print("\n✓ 索引构建完成！现在可以运行问答或评测命令。")


def _load_index_and_embedder():
    """加载FAISS索引和Embedding模型"""
    vector_store = VectorStore()
    vector_store.load(config.FAISS_INDEX_PATH, config.ID_MAP_PATH)

    embedder = create_embedder(
        mode=config.EMBEDDING_MODE,
        model_name=config.LOCAL_EMBEDDING_MODEL_NAME,
        device=config.LOCAL_EMBEDDING_DEVICE,
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        api_model=config.QWEN_EMBEDDING_MODEL,
    )

    retriever = create_retriever(vector_store, embedder, top_k=config.RETRIEVAL_TOP_K)

    qwen_chain = QwenChain(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        model=config.QWEN_GENERATION_MODEL,
        temperature=config.GENERATION_TEMPERATURE,
        max_tokens=config.GENERATION_MAX_TOKENS,
    )

    rag_pipeline = RAGPipeline(retriever, qwen_chain, top_k=config.RETRIEVAL_TOP_K)
    return rag_pipeline


def cmd_ask(args):
    """交互式问答"""
    rag = _load_index_and_embedder()
    print("物流调度RAG问答系统已就绪，输入问题开始对话（输入 quit 退出）")
    print("-" * 60)
    while True:
        try:
            question = input("\n提问: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue
        print("\n回答: ", end="")
        result = rag.query(question, stream=True)
        print(f"\n[来源: {[s['source'] for s in result['sources']]}]")


def cmd_eval(args):
    """运行自动化评测"""
    print("=" * 60)
    print("自动化评测模式")
    print("=" * 60)

    rag = _load_index_and_embedder()

    judge_chain = QwenChain(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL,
        model=config.QWEN_JUDGE_MODEL,
        temperature=0.0,
        max_tokens=1024,
    )

    evaluator = Evaluator(
        rag_pipeline=rag,
        judge_chain=judge_chain,
        test_set_path=config.TEST_SET_PATH,
        output_csv_path=config.EVAL_RESULTS_PATH,
    )

    results = evaluator.run_evaluation(delay=args.delay)

    if args.ablation:
        print("\n" + "=" * 60)
        print("消融实验: RAG-Qwen vs 纯Qwen")
        print("=" * 60)
        ablation_data = evaluator.run_ablation(delay=args.delay)
        analyzer = Analyzer(config.EVAL_RESULTS_PATH, config.CHARTS_DIR)
        analyzer.ablation_analysis(ablation_data)

    if args.analyze:
        print("\n" + "=" * 60)
        print("结果分析")
        print("=" * 60)
        analyzer = Analyzer(config.EVAL_RESULTS_PATH, config.CHARTS_DIR)
        analyzer.load_results()
        analyzer.compute_statistics()
        analyzer.compute_category_stats()
        analyzer.analyze_by_difficulty()
        analyzer.compute_retrieval_quality_report()
        analyzer.find_low_score_cases(threshold=3.0)
        analyzer.diagnose_low_score_cases(threshold=3.0)
        analyzer.generate_all_charts()

    print("\n✓ 评测完成！")


def cmd_analyze(args):
    """分析已有评测结果"""
    analyzer = Analyzer(config.EVAL_RESULTS_PATH, config.CHARTS_DIR)
    analyzer.load_results()
    analyzer.compute_statistics()
    analyzer.compute_category_stats()
    analyzer.analyze_by_difficulty()
    analyzer.compute_retrieval_quality_report()
    low_score = analyzer.find_low_score_cases(threshold=3.0)
    diagnoses = analyzer.diagnose_low_score_cases(threshold=3.0)
    analyzer.generate_all_charts()

    if args.export_low:
        low_path = config.OUTPUTS_DIR / "low_score_cases.csv"
        if len(low_score) > 0:
            low_score.to_csv(low_path, index=False, encoding="utf-8-sig")
            print(f"低分案例已导出至: {low_path}")
        if diagnoses:
            diag_path = config.OUTPUTS_DIR / "low_score_diagnosis.csv"
            pd.DataFrame(diagnoses).to_csv(diag_path, index=False, encoding="utf-8-sig")
            print(f"归因诊断已导出至: {diag_path}")


def main():
    parser = argparse.ArgumentParser(
        description="物流调度领域RAG问答系统与自动化评测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py build-index          # 构建知识库索引
  python main.py ask                  # 交互式问答
  python main.py eval                 # 运行自动化评测
  python main.py eval --ablation      # 评测 + 消融实验
  python main.py eval --analyze       # 评测 + 分析
  python main.py analyze              # 分析已有结果
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    parser_build = subparsers.add_parser("build-index", help="构建知识库索引")

    parser_ask = subparsers.add_parser("ask", help="交互式问答")

    parser_eval = subparsers.add_parser("eval", help="运行自动化评测")
    parser_eval.add_argument("--delay", type=float, default=0.5,
                             help="API调用间隔秒数 (默认: 0.5)")
    parser_eval.add_argument("--ablation", action="store_true",
                             help="运行消融实验 (RAG vs 纯Qwen)")
    parser_eval.add_argument("--analyze", action="store_true",
                             help="评测后自动运行分析")

    parser_analyze = subparsers.add_parser("analyze", help="分析已有评测结果")
    parser_analyze.add_argument("--export-low", action="store_true",
                                help="导出低分案例到CSV")

    args = parser.parse_args()

    if args.command == "build-index":
        cmd_build_index(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()