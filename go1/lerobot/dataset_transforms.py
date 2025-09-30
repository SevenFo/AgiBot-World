import dataclasses
import numpy as np
from collections.abc import Sequence
from typing import (
    Any,
    Dict,
    Protocol,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    Union,
    runtime_checkable,
)

DataDict: TypeAlias = Dict[str, Any]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """
        ...


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class TransformedDataset(Dataset[DataDict]):
    def __init__(
        self, dataset: Dataset, transforms: Sequence[DataTransformFn], num_frames: int
    ):
        self._dataset = dataset
        self._transform = compose(transforms)
        self.num_frames = num_frames

    def __getitem__(self, index: SupportsIndex) -> DataDict:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: Dict
    # 要进行normalize的键名，可以是字符串或字符串列表，默认是"action"
    key: str | list[str] = "action"

    def __call__(self, data: DataDict) -> DataDict:
        # 确保key是列表格式
        keys = [self.key] if isinstance(self.key, str) else self.key

        for k in keys:
            if k in data and k in self.norm_stats:
                data[k] = self._normalize(data[k], self.norm_stats[k])

        return data

    def _normalize(self, x, stats):
        return (x - stats["mean"]) / (stats["std"] + 1e-6)


@dataclasses.dataclass(frozen=True)
class SelectDim(DataTransformFn):
    """select specific dimensions from a given key in the data dictionary."""

    key: str
    index: Union[tuple, list, slice, int]

    def __call__(self, data: DataDict) -> DataDict:
        if self.key in data:
            data[self.key] = data[self.key][self.index]
        return data


@dataclasses.dataclass(frozen=True)
class Padding(DataTransformFn):
    """Pad the data to a fixed length."""

    key: str
    target_shape: tuple | int  # length of target_shape must correspond to target_dim
    target_dim: tuple | int = -1
    pad_value: float = 0.0

    def __call__(self, data: DataDict) -> DataDict:
        if self.key in data:
            # 确保target_shape和target_dim是元组格式
            target_shapes = (
                self.target_shape
                if isinstance(self.target_shape, tuple)
                else (self.target_shape,)
            )
            target_dims = (
                self.target_dim
                if isinstance(self.target_dim, tuple)
                else (self.target_dim,)
            )

            # 处理负数索引
            array_shape = data[self.key].shape
            target_dims = tuple(
                dim if dim >= 0 else len(array_shape) + dim for dim in target_dims
            )

            # 计算需要的padding
            pad_width = []
            for i, dim_size in enumerate(array_shape):
                if i in target_dims:
                    dim_index = target_dims.index(i)
                    target_size = target_shapes[dim_index]
                    if dim_size < target_size:
                        pad_width.append((0, target_size - dim_size))
                    else:
                        pad_width.append((0, 0))
                else:
                    pad_width.append((0, 0))

            # 应用padding
            data[self.key] = np.pad(
                data[self.key], pad_width, constant_values=self.pad_value
            )

            # 如果需要截断到目标形状
            slices = []
            for i, dim_size in enumerate(data[self.key].shape):
                if i in target_dims:
                    dim_index = target_dims.index(i)
                    target_size = target_shapes[dim_index]
                    slices.append(slice(None, target_size))
                else:
                    slices.append(slice(None))

            data[self.key] = data[self.key][tuple(slices)]

        return data


def make_conversation(prompt: str, conversation_type: int = 0) -> str:
    if conversation_type == 0:
        return f"What action should the robot take to {prompt}?"
    elif conversation_type == 1:
        return f"What action should the robot take to {prompt}?"
    elif conversation_type == 2:
        return f"{prompt}"
    else:
        print(f"Conversation Type {conversation_type} is not implemented.")
        raise NotImplementedError()


def test_padding_transform():
    """测试 Padding 类的各种用例"""
    print("=" * 50)
    print("测试 Padding 变换")
    print("=" * 50)

    # 测试用例1: 简单的1D padding (扩展最后一维)
    print("\n测试1: 1D数组padding")
    original1 = np.array([1, 2, 3])
    data1 = {"action": original1.copy()}  # 使用副本
    padding1 = Padding(key="action", target_shape=5, target_dim=-1)
    result1 = padding1(data1)
    print(f"输入: {original1}, shape: {original1.shape}")
    print(f"输出: {result1['action']}, shape: {result1['action'].shape}")
    print("期望: [1 2 3 0 0], shape: (5,)")
    assert result1["action"].shape == (5,), (
        f"期望shape (5,), 实际 {result1['action'].shape}"
    )
    assert np.array_equal(result1["action"], [1, 2, 3, 0, 0]), "padding结果不正确"
    print("✅ 测试1通过")

    # 测试用例2: 2D padding (扩展指定维度)
    print("\n测试2: 2D数组padding特定维度")
    original2 = np.array([[1, 2], [3, 4], [5, 6]])
    data2 = {"state": original2.copy()}
    padding2 = Padding(key="state", target_shape=5, target_dim=0)  # 扩展第0维到5
    result2 = padding2(data2)
    print(f"输入shape: {original2.shape}")
    print(f"输出shape: {result2['state'].shape}")
    print("期望shape: (5, 2) - 第0维扩展到5，用0填充")
    assert result2["state"].shape == (5, 2), (
        f"期望shape (5, 2), 实际 {result2['state'].shape}"
    )
    print("✅ 测试2通过")

    # 测试用例3: 多维度同时padding
    print("\n测试3: 多维度同时padding")
    original3 = np.ones((2, 3, 4))
    data3 = {"image": original3.copy()}
    padding3 = Padding(key="image", target_shape=(4, 6), target_dim=(0, 2))
    result3 = padding3(data3)
    print(f"输入shape: {original3.shape}")
    print(f"输出shape: {result3['image'].shape}")
    print("期望shape: (4, 3, 6) - 第0维扩展到4，第2维扩展到6")
    assert result3["image"].shape == (4, 3, 6), (
        f"期望shape (4, 3, 6), 实际 {result3['image'].shape}"
    )
    print("✅ 测试3通过")

    # 测试用例4: 截断测试 (当原始大小超过目标大小)
    print("\n测试4: 截断测试")
    original4 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    data4 = {"long_array": original4.copy()}
    padding4 = Padding(key="long_array", target_shape=5, target_dim=-1)
    result4 = padding4(data4)
    print(f"输入: {original4}, shape: {original4.shape}")
    print(f"输出: {result4['long_array']}, shape: {result4['long_array'].shape}")
    print("期望: [1 2 3 4 5], shape: (5,) - 截断到目标长度")
    assert result4["long_array"].shape == (5,), (
        f"期望shape (5,), 实际 {result4['long_array'].shape}"
    )
    assert np.array_equal(result4["long_array"], [1, 2, 3, 4, 5]), "截断结果不正确"
    print("✅ 测试4通过")

    # 测试用例5: 负数索引测试
    print("\n测试5: 负数索引测试")
    original5 = np.array([[1, 2, 3], [4, 5, 6]])
    data5 = {"tensor": original5.copy()}
    padding5 = Padding(key="tensor", target_shape=5, target_dim=-1)  # 最后一维
    result5 = padding5(data5)
    print(f"输入shape: {original5.shape}")
    print(f"输出shape: {result5['tensor'].shape}")
    print("期望shape: (2, 5) - 最后一维扩展到5")
    assert result5["tensor"].shape == (2, 5), (
        f"期望shape (2, 5), 实际 {result5['tensor'].shape}"
    )
    print("✅ 测试5通过")

    # 测试用例6: 不存在的键测试
    print("\n测试6: 不存在的键测试")
    data6 = {"existing_key": np.array([1, 2, 3])}
    data6_original = data6.copy()
    padding6 = Padding(key="non_existing_key", target_shape=5, target_dim=-1)
    result6 = padding6(data6)
    print(f"输入键: {list(data6_original.keys())}")
    print(f"输出键: {list(result6.keys())}")
    print("期望: 数据不变，因为键不存在")
    assert result6 == data6_original, "数据应该保持不变"
    print("✅ 测试6通过")

    # 测试用例7: 自定义padding值
    print("\n测试7: 自定义padding值")
    original7 = np.array([1, 2, 3])
    data7 = {"values": original7.copy()}
    padding7 = Padding(key="values", target_shape=6, target_dim=-1, pad_value=-1)
    result7 = padding7(data7)
    print(f"输入: {original7}, shape: {original7.shape}")
    print(f"输出: {result7['values']}, shape: {result7['values'].shape}")
    print("期望: [1 2 3 -1 -1 -1], 用-1填充")
    expected = np.array([1, 2, 3, -1, -1, -1])
    assert np.array_equal(result7["values"], expected), (
        f"期望 {expected}, 实际 {result7['values']}"
    )
    print("✅ 测试7通过")


def test_select_dim_transform():
    """测试 SelectDim 类的各种用例"""
    print("\n" + "=" * 50)
    print("测试 SelectDim 变换")
    print("=" * 50)

    # 测试用例1: 基本索引选择
    print("\n测试1: 基本索引选择")
    original1 = np.array([10, 20, 30, 40, 50])
    data1 = {"state": original1.copy()}
    select1 = SelectDim(key="state", index=[0, 2, 4])
    result1 = select1(data1)
    print(f"输入: {original1}")
    print(f"输出: {result1['state']}")
    print("期望: [10 30 50]")
    expected = np.array([10, 30, 50])
    assert np.array_equal(result1["state"], expected), (
        f"期望 {expected}, 实际 {result1['state']}"
    )
    print("✅ 测试1通过")

    # 测试用例2: 切片选择
    print("\n测试2: 切片选择")
    original2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    data2 = {"action": original2.copy()}
    select2 = SelectDim(key="action", index=(slice(None), slice(1, 3)))
    result2 = select2(data2)
    print(f"输入shape: {original2.shape}")
    print(f"输出shape: {result2['action'].shape}")
    print("期望shape: (3, 2) - 选择所有行，列1-2")
    assert result2["action"].shape == (3, 2), (
        f"期望shape (3, 2), 实际 {result2['action'].shape}"
    )
    expected = original2[:, 1:3]
    assert np.array_equal(result2["action"], expected), "切片结果不正确"
    print("✅ 测试2通过")

    # 测试用例3: 元组索引 (你配置文件中的用法)
    print("\n测试3: 元组索引 (配置文件用法)")
    original3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    data3 = {"observation.state": original3.copy()}
    select3 = SelectDim(key="observation.state", index=([0, 1, 2, 3, 4, 5, -1], ...))
    result3 = select3(data3)
    print(f"输入: {original3}, shape: {original3.shape}")
    print(
        f"输出: {result3['observation.state']}, shape: {result3['observation.state'].shape}"
    )
    print("期望: 选择前6个和最后1个元素 [1 2 3 4 5 6 14]")
    expected = np.array([1, 2, 3, 4, 5, 6, 14])
    assert np.array_equal(result3["observation.state"], expected), (
        f"期望 {expected}, 实际 {result3['observation.state']}"
    )
    print("✅ 测试3通过")

    # 测试用例4: 测试错误的配置场景
    print("\n测试4: 测试你原配置中的问题")
    original4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # 14维state
    data4 = {"observation.state": original4.copy()}

    # 这是你原来的配置，应该选择7个元素
    select4 = SelectDim(key="observation.state", index=([0, 1, 2, 3, 4, 5, -1], ...))
    result4 = select4(data4)
    print(f"原始14维状态: {original4}")
    print(f"选择后7维状态: {result4['observation.state']}")
    print(f"输出维度: {result4['observation.state'].shape[0]}")

    # 验证这确实解决了你的维度问题
    assert result4["observation.state"].shape[0] == 7, "应该输出7维状态"
    print("✅ 测试4通过 - 这解决了你的state_dim=7的问题!")


if __name__ == "__main__":
    # 运行所有测试
    test_padding_transform()
    test_select_dim_transform()

    print("\n" + "=" * 50)
    print("所有测试完成! 🎉")
    print("=" * 50)


if __name__ == "__main__":
    # 运行所有测试
    test_padding_transform()
    test_select_dim_transform()

    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)
