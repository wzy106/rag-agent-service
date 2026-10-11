# 4. InMemorySaver 和 PostgresSaver 区别？

- InMemorySaver：存内存，进程结束即丢失，适合开发测试
- PostgresSaver：存数据库，持久化，多实例共享，适合生产
- **本项目用 AsyncSqliteSaver**：存 SQLite 文件，持久化，单机部署方便
