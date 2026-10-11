# 19. Windows 下载的 wheel 能给 Linux 容器用吗？

**不能。** 三个维度都必须匹配：

| 维度 | Windows 宿主机 | Linux 容器 |
|---|---|---|
| 操作系统 | `win_amd64` | `manylinux*` |
| Python 版本 | `cp314`（本机 3.14） | `cp312`（容器 3.12） |
| ABI | Windows ABI | Linux ABI |

**验证方法**：看 wheel 文件名后缀。
- `torch-2.14.1+cpu-cp314-cp314-win_amd64.whl` ❌ 容器用不了
- `torch-2.x.x+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` ✅ 容器能用

**教训**：**别在宿主机给容器下 wheel**，除非用 `pip download --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all:` 明确指定平台。
