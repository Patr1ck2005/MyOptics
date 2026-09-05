# CI 与自托管 GPU Runner

仓库 CI 由 `.github/workflows/ci.yml` 定义，三个 job：

| Job | 环境 | 内容 |
|-----|------|------|
| `lint` | ubuntu-latest | `ruff check .` |
| `tests-cpu` | ubuntu-latest (py3.10/3.11) | `pytest -m "not gpu"`（GPU 用例自动跳过） |
| `tests-gpu` | 自托管 Windows GPU runner | `pytest -m gpu`（完整物理回归） |

## 自托管 GPU runner 注册（Windows）

1. GitHub → 仓库 → Settings → Actions → Runners → **New self-hosted runner**，
   按向导在 GPU 机器上下载并配置，labels 确保包含 `self-hosted`、`windows`、`gpu`
   （默认自带前两个，`gpu` 需在注册时或 `.\config.cmd` 中追加自定义标签）。
2. 安装为 Windows 服务以常驻：`./svc.sh install && ./svc.sh start`。
3. runner 工作目录会 checkout 独立副本，GPU 测试使用的 Python 环境通过
   **仓库变量** 指定（无需为 runner 单独装依赖）：

   Settings → Secrets and variables → Actions → **Variables**：

   | 变量 | 值示例 | 说明 |
   |------|--------|------|
   | `MYOPTICS_GPU_RUNNER` | `1` | 启用 `tests-gpu` job（未设置时该 job 不运行） |
   | `MYOPTICS_VENV_PYTHON` | `D:\Dev\Projects\Work\MyOptics\.venv\Scripts\python.exe` | 指向主工作区已装好依赖的 venv 解释器 |

4. 安全提示：self-hosted runner 对 fork PR 不自动执行（本工作流仅在
   `push` 到受保护分支时运行 GPU job），且 runner 进程以注册用户权限运行，
   请勿在公开仓库上滥用。

## GPU 用例标记约定

- 需要 GPU 的测试统一打 `@pytest.mark.gpu`（pyproject 已注册 marker）。
- `tests/conftest.py` 在 GPU 不可用时自动跳过这些用例，因此同一套测试
  可同时跑在云端 CPU runner 与本地 GPU runner 上。
- 新增 GPU 测试务必加标记，否则 CPU job 会因缺 CUDA 失败。
