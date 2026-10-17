"""Regression tests for HF cache path resolution on non-root CI runners."""
import rag_core.search as search


class _PermissionDeniedPath:
    def exists(self):
        raise PermissionError(13, "Permission denied", "/root/.cache/huggingface")


def test_safe_exists_handles_permission_error():
    assert search._safe_exists(_PermissionDeniedPath()) is False


def test_resolve_hf_home_falls_back_when_root_is_inaccessible(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(search, "_safe_exists", lambda path: False)

    resolved = search._resolve_hf_home()

    assert resolved.endswith("hf_cache")
