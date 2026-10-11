# 故事 21：Docker 构建报 `unpigz corrupted`（2026-10-08）

### 问题
Docker 构建到 `exporting to image` / `unpacking to image` 阶段突然失败：

```text
#16 exporting to image
#16 unpacking to docker.io/library/agent-agent-api:latest
#16 ERROR: failed to extract layer sha256:f7dfae7ab81225ebfe4b866133d90a08b596073e07813dcada89a0823070199d: 
    exit status 1: unpigz: skipping: <stdin>: corrupted -- incomplete deflate data
failed to solve: Unavailable: error reading from server: EOF
```

### 分析
- 报错位置在 `unpacking`（写盘阶段），不是编译阶段 → **不是代码问题，是环境问题**
- 报错关键词 `corrupted -- incomplete deflate data` → 写入被截断
- 第一反应查磁盘：
  ```powershell
  Get-PSDrive C
  # 结果：C 盘 200 GB，Free 0.00 GB ← 命中
  wsl -d docker-desktop df -h
  # 结果：/mnt/host/c 200.0G 200.0G 0 100% ← 确认
  ```
- Docker 解压镜像层时要往 C 盘写数据，**C 盘一个字节都写不进去**，管道被截断

### 方案
1. **清 C 盘**（管理员 PowerShell）：
   ```powershell
   cleanmgr /sageset:1    # 弹出窗口，全勾
   cleanmgr /sagerun:1

   Remove-Item "C:\Windows\SoftwareDistribution\Download\*" -Recurse -Force
   Remove-Item "$env:TEMP\*" -Recurse -Force
   Remove-Item "$env:USERPROFILE\.cache\huggingface" -Recurse -Force
   pip cache purge
   ```
   清出约 10 GB。

2. **把 Docker 数据迁到 D 盘**（见故事 22）。

### 为什么是"磁盘满"而不是"改代码"
`unpigz corrupted` 和 `incomplete deflate data` 这种错误，**永远是 I/O 层面的问题**，不是程序 bug。看到这类错误，排查顺序应该是：
1. 磁盘空间
2. 磁盘 I/O 故障（坏道、文件系统损坏）
3. 网络传输中断（如果在 pull 阶段）
4. Docker 缓存损坏

**先查磁盘，再查别的。**

### 收获
**Docker 构建失败的第一排查项是磁盘空间。** 尤其在中国 Windows 环境下，C 盘经常被一堆缓存、Docker 虚拟磁盘、HuggingFace 模型吃满。**磁盘满时，各种"看似离奇"的构建报错都可能是写盘失败导致的。**

---
