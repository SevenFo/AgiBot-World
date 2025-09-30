# LeRobot v0.3.3 训练脚本键名重映射迁移指南

> 目标：在 **lerobot v0.3.3** 的 `src/lerobot/scripts/train.py` 中复用最新版（主干分支）提供的键重命名能力，让旧版数据集的字段可以通过配置映射到模型所需的键名，同时保持训练管线其余部分不变。

## 背景差异速览

- **主干版（2025）** 已在录制脚本与处理器工厂中集成 `rename_map`：通过 `RenameObservationsProcessorStep`+`rename_stats` 即可在运行时改写观测键。
- **v0.3.3** 虽然包含 `RenameProcessor`，但 `train.py` 没有入口透传映射；`DatasetConfig` 也缺少对应字段，因此训练批次仍使用原始键。
- 训练管线关键节点：
  1. `make_dataset(cfg)` 加载 `LeRobotDataset`；
  2. `make_policy(cfg.policy, ds_meta=dataset.meta)` 依据 **原始** `meta.features` 构造模型；
  3. Dataloader 直接返回未重命名的样本。

要实现动态映射，需要在 **配置层 → 数据集 → Policy → 批次** 的链路上同步改写键名与统计量。

## 修改总览

| 步骤 | 文件                                    | 目的                                                          |
| ---- | --------------------------------------- | ------------------------------------------------------------- |
| 1    | `src/lerobot/configs/default.py`        | 为 `DatasetConfig` 增加 `rename_map` 配置字段（默认空字典）。 |
| 2    | `src/lerobot/scripts/train.py`          | 引入重命名工具函数 / 数据集封装，并在训练开始前应用映射。     |
| 3    | （可选）`src/lerobot/datasets/utils.py` | 追加 `rename_stats` 等复用函数，避免在脚本内重复实现。        |

下文提供推荐实现细节与代码片段，按需拣选。

## Step 1. 扩展 DatasetConfig

在 `DatasetConfig` 中添加一个 `rename_map` 字段即可让 Draccus CLI / 配置文件直接接受映射字典：

```python
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    repo_id: str
    # ...保留现有字段...
    video_backend: str = field(default_factory=get_safe_default_codec)
    rename_map: dict[str, str] = field(default_factory=dict)  # <- 新增
```

使用示例：

```bash
python -m lerobot.scripts.train \
  --config-path configs/train/diffusion.json \
  --dataset.rename-map='{"cam_front": "observation.images.front", "action_joint": "action"}'
```

Draccus 会自动把 JSON 字符串转换成字典传入 `cfg.dataset.rename_map`。

## Step 2. 在 train.py 中接入重命名逻辑

### 2.1 放置辅助函数

建议在 `train.py` 顶部（imports 之后）粘贴以下工具：

```python
from copy import deepcopy


def rename_sample_keys(sample: dict[str, Any], rename_map: dict[str, str]) -> dict[str, Any]:
    if not rename_map:
        return sample
    renamed = {}
    for key, value in sample.items():
        target = rename_map.get(key, key)
        renamed[target] = value
    return renamed


def rename_meta_in_place(meta, rename_map: dict[str, str]):
    if not rename_map:
        return
    features = meta.info["features"]
    stats = meta.stats or {}
    # episodes_stats 里同样存有逐集统计量，逐键同步更安全
    episodes_stats = meta.episodes_stats or {}

    for src, dst in rename_map.items():
        if src == dst or src not in features:
            continue
        features[dst] = deepcopy(features.pop(src))

        if src in stats:
            stats[dst] = stats.pop(src)

        for ep_idx, ep_stats in episodes_stats.items():
            if src in ep_stats:
                ep_stats[dst] = ep_stats.pop(src)


class RenamedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, rename_map: dict[str, str]):
        self.dataset = dataset
        self.rename_map = rename_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        return rename_sample_keys(sample, self.rename_map)
```

> 如果不想在脚本中定义类，也可以把这些函数放到新模块（例如 `lerobot/datasets/rename_utils.py`），在 `train.py` 中导入。

### 2.2 在 `train()` 中应用映射

在 `dataset = make_dataset(cfg)` 之后立即插入以下逻辑：

```python
    rename_map = getattr(cfg.dataset, "rename_map", {})
    if rename_map:
        rename_meta_in_place(dataset.meta, rename_map)
        dataset = RenamedDataset(dataset, rename_map)

        # 训练过程中会基于 meta.stats 做归一化，为了避免旧键残留，再复制一份改名后字典
        dataset.meta.stats = deepcopy(dataset.meta.stats)
        dataset.meta.episodes_stats = deepcopy(dataset.meta.episodes_stats)
```

要点：

- **先改 meta，再封装 dataset**，确保 `make_policy(cfg.policy, ds_meta=dataset.meta)` 读取到的是新键名。
- `deepcopy` 主要是防止潜在的共享引用导致重命名后出现难以追踪的副作用。
- `RenamedDataset` 会在 `DataLoader` 取样本时删除旧键并写入新键。

### 2.3 传递到 DataLoader

其余训练循环不需要调整，`DataLoader` 会迭代包装后的数据集。若想确认映射生效，可在进入 `update_policy` 前打印一次 `batch.keys()`。

## Step 3. （可选）复用最新版工具

若希望与主干版保持一致，可直接拷贝下列函数，放在 `lerobot_v0.3.3/src/lerobot/processor/rename_processor.py` 或新模块使用：

- `rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]`
- `RenameObservationsProcessorStep`（主干名为 `RenameObservationsProcessorStep`，v0.3.3 的 `RenameProcessor` 已可复用）

替换 `rename_meta_in_place` 中对 `stats` 的处理即可复用官方实现。

## Step 4. 验证流程

1. 通过 CLI 指定映射字典：
   ```bash
   python -m lerobot.scripts.train \
     --config-path path/to/train_config.json \
     --dataset.rename-map='{"camera_front": "observation.images.front", "torque": "observation.state.torque"}'
   ```
2. 启动训练后，在首次 batch 里确认键名已替换：
   ```python
   logging.debug(sorted(batch.keys()))
   ```
3. 若模型使用归一化，请确认 `dataset.meta.stats` 中也已出现新键。

## 注意事项

- **保持映射唯一性**：不要将两个不同原始键映射到同一目标键，否则后写入的值会覆盖先写入的值。
- **保证统计量同步**：若跳过 `rename_meta_in_place`，Policy 仍会依据旧键初始化结构，导致训练过程找不到新键或归一化出错。
- **视频键与图像键**：重命名后，`meta.camera_keys` 会自动返回新键名；若使用 `use_videos=True`，确保重命名字典同样覆盖视频键。
- **推理阶段一致性**：部署或评估脚本也需要套用同样的 `rename_map`，否则序列化的 `dataset_stats.json` 与训练时不一致。

按照以上步骤，即可在 v0.3.3 版本中复现主干 repo 的键重映射功能，而无需升级整套代码。