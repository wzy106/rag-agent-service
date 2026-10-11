"""
把大 markdown 按二级标题(##)拆成多个小文件。
用法：python scripts/split_notes.py
"""
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 待拆分的文件 → 输出目录
SPLIT_TASKS = [
    ("LESSONS_LEARNED.md", "docs/stories"),
    ("PROJECT_NOTES.md", "docs/notes"),
    ("README.md", "docs/readme"),
    ("docs/_old_langchain.md", "docs/langchain"),   # ← 新增：旧笔记
]


def split_by_h2(content: str) -> list:
    """按 '## xxx' 二级标题拆分。返回 [(title, body), ...]"""
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(content))

    if not matches:
        return [("full", content)]

    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def sanitize_filename(title: str) -> str:
    """标题转文件名：去特殊字符，空格换下划线。"""
    name = re.sub(r"[^\w\s\u4e00-\u9fff-]", "", title)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:50]


def main():
    for src_name, out_dir_rel in SPLIT_TASKS:
        src = PROJECT_ROOT / src_name
        out_dir = PROJECT_ROOT / out_dir_rel
        out_dir.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"[跳过] {src} 不存在")
            continue

        content = src.read_text(encoding="utf-8")
        sections = split_by_h2(content)

        print(f"\n[{src_name}] 拆出 {len(sections)} 个章节 → {out_dir_rel}/")
        for i, (title, body) in enumerate(sections, 1):
            filename = f"{i:02d}_{sanitize_filename(title)}.md"
            out_path = out_dir / filename
            out_path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
            print(f"  → {filename} ({len(body)} 字符)")


if __name__ == "__main__":
    main()