## 训练自己的数据集
1. 将图片、标签和list文件放入 `data/` 下
2. 在 `experiments/cityscapes` 下新建一份[test.yaml](experiments/cityscapes/test.yaml)
3. 在 `lib/datasets/` 下复制一份cityscapes.py 重命名为 infrared_images.py，并修改其中的参数
4. 在 `lib/datasets/__init__.py` 中导入刚才建立的数据集
5. `lib/models/ddrnet_23_slim.py` 中修改 num_classes

### 训练
```bash
PYTHONDONTWRITEBYTECODE=1 python -B tools/train.py --cfg experiments/cityscapes/test.yaml TRAIN.BATCH_SIZE_PER_GPU 12 && /usr/bin/shutdown
```