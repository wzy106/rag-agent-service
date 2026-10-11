# 故事 25：Windows wheel 不能给 Linux 容器用（2026-10-08）

### 问题
为了避开容器内 pip 下载超时，我想先在 Windows 宿主机下载所有 wheel 到 `wheelhouse/`，再让容器离线安装。

宿主机跑：
```cmd
pip download -r requirements.txt -d wheelhouse -i https://mirrors.aliyun.com/pypi/simple/
```

下完了，`dir wheelhouse` 显示 100+ 个 whl。看文件名：
```
torch-2.14.1+cpu-cp314-cp314-win_amd64.whl
faiss_cpu-1.15.1-cp314-cp314-win_amd64.whl
pywin32-312-cp314-cp314-win_amd64.whl
...
```

### 分析
**三个维度都不匹配**：

| 维度 | Windows 宿主机 | Linux 容器 |
|---|---|---|
| 操作系统 | `win_amd64` | `manylinux*` |
| Python 版本 | `cp314`（本机 3.14） | `cp312`（容器 3.12） |
| ABI | Windows ABI | Linux ABI |

wheel 文件名后缀解读：
```
torch-2.14.1+cpu-cp314-cp314-win_amd64.whl
                ^^^^^  ^^^^^  ^^^^^^^^
                CPython 3.14  Windows amd64
```

**Docker 容器里**：
- 平台是 Linux（`manylinux`）
- Python 3.12（`cp312`）
- **一个都装不上**

### 方案
**放弃 wheelhouse 方案，回到容器内直接下载**（配合故事 23 的阿里云源 + 故事 24 的 CPU 版 torch，下载量降到 600 MB，能过）。

**如果非要本地下 wheel 给容器用**，必须用 `--platform` 明确指定：
```cmd
pip download -r requirements.txt -d wheelhouse ^
    --platform manylinux2014_x86_64 ^
    --platform manylinux_2_17_x86_64 ^
    --platform manylinux_2_28_x86_64 ^
    --platform any ^
    --python-version 3.12 ^
    --only-binary=:all: ^
    --timeout 300 --retries 10
```

### 为什么容易踩
- 开发时"下个 wheel 而已"看起来很简单
- 但 pip 默认下载**当前平台**的 wheel
- 宿主机是 Windows/Python 3.14，容器是 Linux/Python 3.12
- **两个环境的 wheel 不通用**

### 收获
**别在宿主机给容器下 wheel**——这是**跨平台工具链陷阱**。除非明确用 `--platform` + `--python-version` 指定目标平台，否则宿主机下下来的都是"自己用得上、容器用不了"的 whl。

**面试时能讲**：
> "为了避开容器内 pip 超时，我试过在 Windows 宿主机下载 wheel 让容器离线安装。结果发现宿主机 Python 3.14 下载的是 `win_amd64` + `cp314` 的 wheel，而容器是 Linux + Python 3.12，需要 `manylinux` + `cp312`，一个都装不上。后来我明白了：wheel 是平台相关的，跨平台必须用 `--platform` 明确指定。"

---
