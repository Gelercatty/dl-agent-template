# dl-agent-template

A practical starter for AI agents and humans collaborating on reproducible deep-learning experiments.

一个统一的深度学习启动仓库模版：把开发、源码发布、任务定义和运行产物分开，让人和 Agent 都能*真正*找到
代码、复现任务、追溯结果。提供工作流工具和真实的 YAML＋Lightning CPU 示例，沿用 BOMB 的 data/model/lightning/pipeline 分层。

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

## 代码组织与协作

```text
configs/pipelines/*.yaml                完整、可审阅的实验配置
src/dl_demo/data/                       LightningDataModule：数据与采样
src/dl_demo/model/                      纯 nn.Module：模型前向
src/dl_demo/lightning/                  LightningModule：loss、优化器、指标
src/dl_demo/pipeline/adapters/          schema、preflight、build、产物验收
src/dl_training/lightning.py            共享 Trainer、完整 baseline、周期评测、checkpoint
```

YAML 是实际训练入口，不维护同一实验的 JSON 副本。JSON 只承担 registry、来源清单、
机器产物及旧 Job 的兼容读取。新 Job 原样冻结为 `config.yaml`。

保留 BOMB 的协作方式：人确定研究目标和验收标准，agent 实现、验证并准备可审阅 Job；
变体展示 YAML 字段差异，已授权的操作不重复询问。正式运行保留原计划和前后记录，
失败尝试及旧快照不覆盖；普通调查直接在聊天中交流。

[demo.yaml](configs/pipelines/demo.yaml) 中的评测周期按 optimizer step 计算。
示例先完整验证，再训练，每 20 次更新验证，最后完整验证；累计 2 个 microbatch 才更新。
这是 CPU 工作流示例，不能据此宣称某个大模型或 FSDP 已验证。

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
  --resume "$RUN_ROOT/Runs/demo-001/ckpt/last.ckpt"
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
| [src/dl_demo/](src/dl_demo/pipeline/adapters/classification.py)                               | DataModule、纯模型、LightningModule 和 adapter            |
| [configs/registry.json](configs/registry.json)                        | 正式 pipeline 注册表                     |
| [configs/pipelines/](configs/pipelines/demo.yaml)                     | 完整实验配置，无命令行隐式覆盖           |
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

不包含云平台账号、SSH 默认值、训练服务或自动上传／删除工具。可选 tracking extra 接入 SwanLab。
GitHub 模板复制的是起始文件，后续不会自动同步上游更改。

## 检查

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked --extra demo pytest -q
```

CI 在 Linux CPU 上检查发布、任务校验、失败／中断处理和真实训练恢复。
配置读取使用 PyYAML safe loader；示例使用 PyTorch Lightning 2.6.5。
Linux 示例使用 CPU wheel，其他训练环境应在项目中维护自己的依赖锁或镜像 digest。

[MIT License](LICENSE) · 具体数据、模型权重和外部代码应各自遵守原许可证。

## 私有运行配置与 SwanLab

机器配置写在忽略提交的 `.local/runtime.yaml`，用 `prepare --runtime .local/runtime.yaml`
绑定到固定 Job。示例支持 `accelerator: cpu`、`devices: 1`、`strategy: auto` 和
`swanlab_mode: disabled|offline|online`；默认 disabled。启用日志需安装 `tracking` extra。
SwanLab project/tags 写在实验 YAML，凭据使用既有环境变量，dry-run 不连接服务。
CUDA adapter 必须实现并验证自己的分布式策略；不要把 demo 的 CPU 配置当成 GPU 实现。
