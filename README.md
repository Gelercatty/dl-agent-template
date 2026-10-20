# dl-agent-template

A practical starter for AI agents and humans collaborating on reproducible deep-learning experiments.

一个统一的深度学习启动仓库模版：把开发、源码发布、任务定义和运行产物分开，让人和 Agent 都能*真正*找到
代码、复现任务、追溯结果。提供工作流工具和真实的 PyTorch CPU 示例，可接入已有 trainer。

## 工作方式

```text
Code（本地或远程 Git 工作区）
  └─ freeze → Releases/<source-id>/       冻结源码 + 内容哈希 + Git 来源
                 └─ prepare → Jobs/<id>/ 配置 + 启动脚本 + 计划 + 训练记录
                                  └─ run → Runs/<id>/ 日志 + checkpoint + 指标 + 终态
```

- 开发工作区可以就在远程 CPU 开发机／共享存储上，临时数据处理在 `scratch/` 进行。
- 正式任务从冻结源码启动；快照可以包含未提交修改，不要求每次实验先提交 Git。
- 模型、数据和训练语义写进注册配置；机器路径在私有 bindings 和生成的 Job 中绑定。
- `prepare` 和 `check` 不启动训练。已有明确运行／发布授权时，Agent 验证后继续执行。
- 普通调查在对话中交流；正式训练保留前后记录。退出码不等于训练成功或模型质量通过。

## 五分钟运行 CPU 示例

需要 Git、Bash、Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
支持 Linux 和 macOS。点击 GitHub **Use this template** 建立你的仓库后，在其根目录执行：

```bash
uv sync --locked --extra demo
source .venv/bin/activate
export PYTHON_BIN="$VIRTUAL_ENV/bin/python"
export RUN_ROOT="$(cd .. && pwd)/dl-agent-workspace"

RELEASE_DIR=$(python -m workflow release --root "$RUN_ROOT")
python -m workflow prepare \
  --release "$RELEASE_DIR" --root "$RUN_ROOT" \
  --name demo-001 --pipeline demo --plan plans/demo.md

python -m workflow check --job "$RUN_ROOT/Jobs/demo-001"
bash "$RUN_ROOT/Jobs/demo-001/run.sh"
python -m workflow status --job "$RUN_ROOT/Jobs/demo-001"
```

示例在 CPU 上用合成数据训练一个两层 MLP，共 40 步，无数据下载、GPU 或云服务。
验收要求 loss 下降、验证准确率 ≥ 90%、checkpoint 和步数一致。
重复运行时使用新的 Job 名；已有 Run 不覆盖。

将相同模型继续训练到 80 步：

```bash
python -m workflow prepare \
  --release "$RELEASE_DIR" --root "$RUN_ROOT" \
  --name demo-resume-001 --pipeline demo-long --plan plans/resume-demo.md \
  --resume "$RUN_ROOT/Runs/demo-001/ckpt/last.pt"
python -m workflow check --job "$RUN_ROOT/Jobs/demo-resume-001"
bash "$RUN_ROOT/Jobs/demo-resume-001/run.sh"
```

恢复会校验 checkpoint、元数据和配置兼容性，保留前一次 Job/Run。
测试还会比较恢复至 80 步和连续训练 80 步的全部模型参数、优化器状态和随机状态。

## 仓库入口

| 路径                                                                  | 用途                                     |
| --------------------------------------------------------------------- | ---------------------------------------- |
| [AGENTS.md](AGENTS.md)                                                | 人与 Agent 的协作边界、授权和检查约定    |
| [.agents/skills/dl-experiment](.agents/skills/dl-experiment/SKILL.md) | 仓库内的实验工作流 skill                 |
| [workflow/](workflow/__main__.py)                                     | 源码冻结、任务生成、验证、执行、状态查询 |
| [src/dl_demo/](src/dl_demo/pipeline.py)                               | 可替换的 PyTorch 训练 adapter            |
| [configs/registry.json](configs/registry.json)                        | 正式 pipeline 注册表                     |
| [configs/pipelines/](configs/pipelines/demo.json)                     | 完整实验配置，无命令行隐式覆盖           |
| [plans/](plans/demo.md)                                               | 冻结进任务的实验假设与验收条件           |
| [docs/workflow.md](docs/workflow.md)                                  | 远程开发、发布、运行和中断处理           |
| [docs/adapters.md](docs/adapters.md)                                  | 接入已有模型、数据和分布式 trainer       |
| [docs/records.md](docs/records.md)                                    | 训练前后记录与实验归档约定               |

## 用于自己的项目

1. 修改项目名称、README 和团队的 `AGENTS.md`；MIT 的原始版权声明需保留。
2. 将现有训练代码放入 `src/` 或调整 [workflow.toml](workflow.toml) 的源码包含目录。
3. 按 [adapter 合同](docs/adapters.md) 包装已有入口，注册完整配置，写自己的计划和验收。
4. 在主开发机建立忽略提交的 `.local/`，保存数据路径绑定和平台映射。
5. 用小规模 smoke 验证后，生成正式训练 Job，交给已有调度平台执行 `run.sh`。

不包含云平台账号、SSH 默认值、训练服务、外部 tracker 或自动上传／删除工具。
GitHub 模板复制的是起始文件，后续不会自动同步上游更改。

## 检查

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked --extra demo pytest -q
```

CI 在 Linux CPU 上检查发布、任务校验、失败／中断处理和真实训练恢复。
工作流本身只使用 Python 标准库；PyTorch 仅属于可替换示例。
Linux 示例使用 CPU wheel，其他训练环境应在项目中维护自己的依赖锁或镜像 digest。

[MIT License](LICENSE) · 具体数据、模型权重和外部代码应各自遵守原许可证。
