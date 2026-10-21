# 开发、发布与执行

## 主开发工作区

代码可以放在本机，也可以放在长期使用的远程 CPU 开发机／共享存储。
为同一分支选定主工作区，明确本次修改发生在哪里。不要把两个有独立修改的目录互相覆盖；
通过 Git commit、补丁或经过核对的单向同步交换变更。GPU 节点主要执行固定任务。

```text
Code/project/               Git 开发工作区
  .local/                   私有机器、数据路径和调度映射（忽略）
  scratch/                  临时处理脚本与输出（忽略）
project-workspace/
  Releases/<release-id>/    冻结源码，不原地修改
  Jobs/<job-id>/            固定任务定义和可追加训练记录
  Runs/<job-id>/            每次尝试独占的运行产物
```

运行根目录放在源码仓库外。数据仍可保留在已有目录，通过逻辑 ID、内容哈希和私有绑定关联。
临时处理流程稳定后，将脚本、配置和有效测试提升到源码。

## 发布源码

`python -m workflow release --root <运行根目录>` 捕获 Git 可见且由
[workflow.toml](../workflow.toml) 包含的文件，包括 tracked 修改和未忽略的新文件。
删除的 tracked 文件不复制。仓库必须已经 `git init`；尚未提交时 Git commit 记为 null。

发布记录 Git commit、dirty 状态、每个文件的 SHA-256／大小／执行位，生成内容身份，
再检查源文件是否在复制期间变化。完整目录先在 staging 中创建，再原子重命名并设为只读。
相同身份可复用；每次运行检查文件集合和哈希。`RELEASE.json` 是来源清单。

包含目录是显式 allowlist，常见权重和密钥文件即使被 tracked 也会拒绝。这不是通用 secret
扫描器：发布前仍要检查所选源码、配置和文档。只读权限与哈希用于发现误改，不构成恶意代码
的安全隔离。并发编辑应暂停后再 freeze；文件系统不是事务快照。

## 准备和验证 Job

`prepare` 只接受冻结注册表中的 pipeline，配置不能在 Job 中临时覆盖。
修改变量时，改 Code 中的完整配置，再发布新 Release。

```text
Jobs/<job-id>/
  JOB.json                来源、输入绑定、恢复来源和任务文件哈希
  config.yaml             原样冻结 YAML（旧 JSON Job 继续读取）
  PLAN.md                 冻结计划原文
  run.sh                  固定入口，运行时可指定 PYTHON_BIN
  TRAINING_RECORD.md      执行前计划、自动事实和人工／Agent 解释
```

`PLAN.md` 及任务定义不可修改；`TRAINING_RECORD.md` 可追加实际结果。
机器路径只出现在仓库外的 Job 中；复制到不同挂载路径后，要重新准备 Job。

`python -m workflow check --job <Job目录>` 与 `bash run.sh --dry-run` 校验源码和任务哈希、
注册配置、实际输入内容、恢复兼容性、所需 Python 依赖及输出目录冲突。
检查使用冻结源码和指定解释器，不建立 Run、不创建模型、不访问 GPU；adapter 的 preflight
必须遵守此约定。它不证明 GPU 显存足够、分布式通信可用或任务质量通过，仍需受授权的 smoke。

生成脚本使用 `PYTHON_BIN`，默认 `python3`；建议设置为目标环境的绝对解释器路径。
脚手架记录解释器、平台和 adapter 返回的版本摘要，不保存整个进程环境。依赖缺失在目标
开发环境预先解决，不在训练启动脚本中临时在线安装。

## 执行与平台提交

已经授权执行具体任务时，通过检查后直接运行。若没有授权，先交付 Job、具体命令、资源建议
和验收条件，再申请缺失的授权。不要把每次只读检查变成审批。

在云平台、Slurm 或容器等现有环境中，将 `bash <Job目录>/run.sh` 作为一个 controller 入口。
确认目标挂载下 Release 和输入可访问，解释器符合约定。平台资源、镜像 digest、提交 ID
和实际运行地址写入私有 Job 记录。模板没有平台提交器，不默认任何 GPU 数量。
共享存储可直接复用 Release；跨机器复制后运行 `python -m workflow verify <Release目录>`。

每次运行通过独占建目录领取 Run：

```text
Runs/<job-id>/
  RUN.json                来源、启动时间和环境摘要
  train.log               Python 训练日志
  ckpt/                   adapter Lightning checkpoint 和元数据
  metrics.json            示例指标
  TRAIN_COMPLETE.json     通过输出合同后的步数与结果
  JOB_SUCCESS.json        验收终态与产物哈希
  JOB_FAILED.json         异常／可捕获中断终态，与 SUCCESS 互斥
```

平台还应采集原生库和子进程 stdout/stderr。仅进程返回 0 不足以成功：adapter 输出验收必须
通过、实际步数达到计划、所需产物非空、来源在运行后仍校验一致。`status` 再验证成功产物哈希。
视觉质量、泛化和业务目标仍需按计划评审，通用脚手架无法代替。

## 中断与恢复

SIGINT／SIGTERM 记录中断，普通异常记录失败。SIGKILL、掉电或磁盘故障可能来不及写终态；
`running_or_unfinished` 只表示存在 Run 且无终态，不能断言进程仍在运行。
检查平台状态和准确的进程／日志后，在训练记录补充原因，不伪造成功 marker。

恢复总是新 Job、新 Run，通过 `--resume` 引用已有 checkpoint，原运行不覆盖。
checkpoint 必须配套 `.json` 元数据，校验内容、完成步数和 adapter 定义的兼容性。
示例允许改变 training.max_steps 和用于标识新实验的 experiment.name；恢复模型、optimizer、
Lightning loop 和 RNG，要求相同 torch/Lightning 版本。
故意更换模型或数据应定义迁移／初始化语义，不把它声称为精确恢复。

示例每 20 步写独立 checkpoint，`ckpt/latest.json` 指向最新完整一代，最终写 `last.ckpt`。
中断时用完整一代的 `.ckpt` 路径恢复。示例按 YAML keep_checkpoints 保留最近完整周期 checkpoint，最终另存 last；只清理本 callback
在本 Run 创建的周期文件。生产 trainer 需核定完整／分片／仅 adapter checkpoint 的存储预算。归档／清理只针对确认的 Run，不因上传成功自动删除原件。

## 实验 YAML 与运行参数

新实验使用 schema 2：`training.max_steps` 表示累计 optimizer 更新次数。
完整 YAML 在 `configs/pipelines/`，禁止隐式覆盖、YAML alias 和重复键；学习率使用
`1.0e-5` 这样的 YAML 数值写法。Job 哈希覆盖原始 YAML 字节，修改注释同样需要新 Job。
历史 schema 1 JSON 和旧 Job 定义仍可校验；旧 Release 的执行代码保持原样。

`--runtime .local/runtime.yaml` 将资源和日志运行模式固化到 JOB.json 的 execution 中，
与实验 YAML 分离。未实现运行参数合同的 adapter 拒绝非空 execution。
preflight 验证参数但不分配设备、实例化模型、初始化 tracker 或创建 Run。

共享 Trainer 负责完整训练前验证、按 optimizer step 周期验证和训练后验证。
本示例 resume 建立新 Job/Run，继承模型、optimizer、Lightning loop 和任务 RNG 状态；
BOMB 的复用 Run ID / attempt 模式没有原样移入此模板，避免更改旧运行身份和追溯合同。
