import re
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class PDFDocument:
    filename: str
    full_text: str
    pages: List[str] = field(default_factory=list)
    page_count: int = 0


class PDFLoader:
    """
    PDF批量加载器，使用 pdfplumber 提取文本，保留段落换行。
    支持过滤页眉页脚（数字页码及固定字符）。
    """

    def __init__(self, pdf_dir: Path):
        self.pdf_dir = Path(pdf_dir)
        if not self.pdf_dir.exists():
            raise FileNotFoundError(f"PDF目录不存在: {self.pdf_dir}")

    def load_all(self) -> List[PDFDocument]:
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"警告: {self.pdf_dir} 目录下未找到PDF文件")
            return []

        documents = []
        for pdf_path in pdf_files:
            doc = self._load_single(pdf_path)
            if doc and doc.full_text.strip():
                documents.append(doc)
                print(f"已加载: {doc.filename} ({doc.page_count} 页, {len(doc.full_text)} 字符)")
        return documents

    def _load_single(self, pdf_path: Path) -> Optional[PDFDocument]:
        try:
            import pdfplumber
        except ImportError:
            print("pdfplumber 未安装，尝试使用 pypdf...")
            return self._load_with_pypdf(pdf_path)

        try:
            pages = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        cleaned = self._clean_page_text(text)
                        pages.append(cleaned)
            full_text = "\n\n".join(pages)
            return PDFDocument(
                filename=pdf_path.name,
                full_text=full_text,
                pages=pages,
                page_count=len(pages),
            )
        except Exception as e:
            print(f"pdfplumber 加载失败 ({pdf_path.name}): {e}，尝试 pypdf...")
            return self._load_with_pypdf(pdf_path)

    def _load_with_pypdf(self, pdf_path: Path) -> Optional[PDFDocument]:
        try:
            from pypdf import PdfReader
        except ImportError:
            print("pypdf 未安装，无法加载PDF")
            return None

        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    cleaned = self._clean_page_text(text)
                    pages.append(cleaned)
            full_text = "\n\n".join(pages)
            return PDFDocument(
                filename=pdf_path.name,
                full_text=full_text,
                pages=pages,
                page_count=len(pages),
            )
        except Exception as e:
            print(f"pypdf 加载失败 ({pdf_path.name}): {e}")
            return None

    @staticmethod
    def _clean_page_text(text: str) -> str:
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()


class TextLoader:
    """
    纯文本文件加载器，支持 .txt 文件。
    """

    def __init__(self, text_dir: Path):
        self.text_dir = Path(text_dir)
        if not self.text_dir.exists():
            raise FileNotFoundError(f"文本目录不存在: {self.text_dir}")

    def load_all(self) -> List[PDFDocument]:
        txt_files = sorted(self.text_dir.glob("*.txt"))
        if not txt_files:
            return []

        documents = []
        for txt_path in txt_files:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if text.strip():
                    doc = PDFDocument(
                        filename=txt_path.name,
                        full_text=text.strip(),
                        pages=[text.strip()],
                        page_count=1,
                    )
                    documents.append(doc)
                    print(f"已加载: {doc.filename} ({len(doc.full_text)} 字符)")
            except Exception as e:
                print(f"加载文本文件失败 ({txt_path.name}): {e}")
        return documents


def load_pdfs(pdf_dir: Path) -> List[PDFDocument]:
    loader = PDFLoader(pdf_dir)
    return loader.load_all()


def load_documents(data_dir: Path) -> List[PDFDocument]:
    """
    统一加载入口：先加载PDF，再加载文本文件，合并返回。
    """
    documents = []

    pdf_loader = PDFLoader(data_dir)
    pdf_docs = pdf_loader.load_all()
    documents.extend(pdf_docs)

    text_loader = TextLoader(data_dir)
    text_docs = text_loader.load_all()
    documents.extend(text_docs)

    if not documents:
        print(f"警告: {data_dir} 目录下未找到PDF或TXT文件")
    else:
        print(f"共加载 {len(documents)} 个文档 ({len(pdf_docs)} PDF, {len(text_docs)} TXT)")

    return documents