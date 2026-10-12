# 17. Docker 构建失败怎么排查？

**案发现场**：
```
#16 exporting to image
#16 ERROR: failed to extract layer sha256:f7df...: 
    exit status 1: unpigz: skipping: <stdin>: corrupted -- incomplete deflate data
failed to solve: Unavailable: error reading from server: EOF
```

**排查思路**：
1. **看错误阶段**：报在 `exporting to image` / `unpacking` → 是写盘阶段，不是编译阶段
2. **查磁盘空间**：`Get-PSDrive C` → 显示 Free = 0.00 GB ✅ 命中
3. **查 Docker 数据位置**：`wsl -d docker-desktop df -h` → `/mnt/host/c` 100% 满

**根因**：C 盘 200 GB 用满，Docker 解压镜像层时写不进去，管道被截断。

**解法**：
1. 清 C 盘（cleanmgr + 手动删 Temp/SoftwareDistribution/HF cache）
2. Docker 数据迁到 D 盘（Settings → Disk image location）
3. C 盘从 10 GB → 65 GB 可用
