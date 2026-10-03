from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载 docs/ 下所有 md 文件
docs_dir = Path(__file__).resolve().parent.parent / "docs"

loader = DirectoryLoader(
    str(docs_dir),
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()
print(f"加载了 {len(documents)} 个文档")
for d in documents:
    print(f"  - {Path(d.metadata['source']).name}: {len(d.page_content)} 字符")

# 2. 切分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
chunks = splitter.split_documents(documents)
print(f"\n切分成 {len(chunks)} 块")
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- 块 {i+1}（{len(chunk.page_content)} 字符）---")
    print(chunk.page_content[:100])