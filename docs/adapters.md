# 接入已有训练代码

保留 YAML、DataModule、纯模型、LightningModule 和注册 adapter 分层。当前示例是
[分类 adapter](../src/dl_demo/pipeline/adapters/classification.py)，通过 `build` 返回
`TrainingPlan(module, datamodule, resume_key)`；公共 Trainer 位于
[dl_training/lightning.py](../src/dl_training/lightning.py)。adapter 不手写训练循环。

## 注册配置

在 [configs/registry.json](../configs/registry.json) 注册模块及完整 YAML 配置路径。
正式 Job 不接受任意模块名或命令行训练超参覆盖；一个 pipeline 名对应一个可审阅配置。

| 字段                 | 合同                                                        |
| -------------------- | ----------------------------------------------------------- |
| `schema`             | 新 YAML 为整数 2；兼容旧 JSON 的 1                                                |
| `seed`               | 整数；真实 trainer 还需记录 sampler、rank 等随机性来源      |
| `training.max_steps`          | 正整数；表示 optimizer 更新次数，adapter 说明累计与恢复语义 |
| `inputs`             | 输入名到 `{id, sha256}` 的映射；大数据可引用版本化 manifest |
| `expected_artifacts` | Run 内必须存在且非空的相对文件路径，至少一个                |
| 其他字段             | adapter 严格验证的模型、优化器、数据、评测语义              |

对于大数据 manifest，通用工具只验证 manifest 文件；adapter 必须验证其中数据的版本和
完整性，不能声称通用工具已读取所有分片。

私有绑定示例（放到忽略提交的 `.local/inputs.json`，自行填实际路径）：

```json
{ "train": "/your/data/train-manifest.json" }
```

准备时传入 `--bindings .local/inputs.json`；绑定名称必须与 config.inputs 一致。
不要把账号、token、私有 URL 或机器映射放进可发布配置。

## 四个函数

```python
def validate(config: dict) -> None:
    """Strict schema and experiment semantics; raise on unsupported fields."""

def preflight(config: dict, bindings: dict, resume_metadata: dict | None) -> dict:
    """Check dependencies, input metadata and resume compatibility; return version facts."""

def run(config: dict, run_dir: Path, bindings: dict, resume: Path | None) -> None:
    """Train using existing code; write only under run_dir; raise on failure."""

def validate_outputs(config: dict, run_dir: Path) -> dict:
    """Read actual artifacts and return JSON facts including integer completed_steps."""
```

`preflight` 不分配 GPU、不创建模型、不写产物。`validate_outputs` 检查真实 checkpoint 步数、
有限参数、指标和验收，而不是直接回传 `config.training.max_steps`。通用 controller 再检查步数、
所需文件、源码和任务哈希。输出校验只处理自己生成或可信的 checkpoint。

JSON 可用 `workflow.common.atomic_json` 写入。模型文件先写临时文件，完成后原子替换。
controller 独占 `RUN.json`、`train.log`、`TRAIN_COMPLETE.json`、`JOB_SUCCESS.json`、
`JOB_FAILED.json`，adapter 不写这些文件。

恢复 checkpoint 的 `<filename>.json` 至少包含：

```json
{
  "schema": 1,
  "checkpoint_sha256": "64-character SHA-256 of the checkpoint file",
  "completed_steps": 40,
  "resume_key": "adapter-defined model/data/optimizer compatibility identity"
}
```

`resume_key` 的解释属于 adapter；示例还含 `torch` 与 `lightning` 版本。
模型、optimizer、scheduler、scaler、RNG、sampler 和梯度累计位置等状态按实际 trainer 的
机制保存。示例为全批量 CPU 训练，只需要其中适用的状态。

## 分布式与环境

一个 Job 只运行一个 controller；直接以 `torchrun -m workflow` 启动会被拒绝。
多 GPU adapter 内部调用既有分布式 launcher，等待所有 worker 返回，将失败和信号传回
controller，确保子进程停止且只有 global rank 0 发布普通产物。checkpoint 的分片格式、
每个 rank 的恢复和集合完整性由 adapter 验证。

模板尚未提供多 GPU adapter，未验证分布式训练；CPU 示例通过不代表多卡通过。
tracker、缓存服务、对象存储等按实际需要引入，明确资源关闭和失败处理。
长任务需限制缓存、日志和 checkpoint 保留量。

依赖在根目录维护。示例锁文件使用 CPU torch；接入 CUDA trainer 时，修改来源和版本，
重新锁定并检查目标环境，或记录已验证镜像 digest。不要维护相互冲突的依赖说明。


## 分层与配置变体

`data/` 负责样本、split、采样和 DataModule；`model/` 仅依赖 tensor/nn.Module；
`lightning/` 定义 loss、optimizer 和任务指标；`pipeline/adapters/` 严格校验、preflight、
装配和验收。不同超参数使用完整 YAML，展示字段差异，不复制 Lightning 代码。
公共 Trainer 接收真实模块，管理 lifecycle、baseline、评测周期和 checkpoint。

`preflight_runtime(runtime)` 是可选的运行部署合同：检查生成 Job 中的设备/strategy/logger
参数，不能创建模型、触碰 GPU 或初始化 logger。没有此函数的 adapter 不接受 runtime 覆盖。
共享 CPU 示例只实现单进程；H3 的嵌套 FSDP、精度、梯度裁剪和分片 checkpoint 由 H3 的
实际 adapter 管理与验证，不假设 Lightning 默认封装可直接套用所有大模型。

模板尚未实现独立 eval Job 或自动消融 YAML 生成器；不要把设计约定当作已有命令。
