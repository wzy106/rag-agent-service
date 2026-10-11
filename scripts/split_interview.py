"""
把 notes/07_七面试可能被问的问题.md 按 ### 三级标题拆成多个文件。
每道面试题独立成一个 markdown 文件，提升文档级召回判别力。

用法：python scripts/split_interview.py
"""
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET = PROJECT_ROOT / "docs" / "notes" / "07_七面试可能被问的问题附答案要点.md"


def sanitize_filename(title: str) -> str:
    """标题转文件名。"""
    # 去掉 "1. " "2. " 这种序号前缀
    title = re.sub(r"^\d+\.\s*", "", title)
    # 去掉特殊字符
    name = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", title)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:40]


def main():
    if not TARGET.exists():
        print(f"[错误] 找不到 {TARGET}")
        return

    content = TARGET.read_text(encoding="utf-8")

    # 找所有 '### ' 三级标题位置
    pattern = re.compile(r"^### (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(content))

    if not matches:
        print("[错误] 没有找到 ### 三级标题")
        return

    print(f"[拆分] 找到 {len(matches)} 道面试题")

    # 拆出每个面试题（含到下一个 ### 之前的内容）
    out_dir = PROJECT_ROOT / "docs" / "notes_interview"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, m in enumerate(matches, 1):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i].start() if i < len(matches) else len(content)
        body = content[start:end].strip()

        filename = f"q{i:02d}_{sanitize_filename(title)}.md"
        out_path = out_dir / filename

        # 用三级标题作为一级标题
        out_path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
        print(f"  → {filename} ({len(body)} 字符)")

    # 删除原文件
    TARGET.unlink()
    print(f"\n[完成] 原文件已删除，拆成 {len(matches)} 个文件到 docs/notes_interview/")
    print("[注意] 请重建索引：rmdir /s /q faiss_index")


if __name__ == "__main__":
    main()