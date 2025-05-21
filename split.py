# import os
# import random
# import shutil
#
# random.seed(42)
#
# source_base = "/root/autodl-tmp/val_036"
# dest_val_base = "/root/autodl-tmp/val_final"
# dest_test_base = "/root/autodl-tmp/test_final"  # 修正可能的拼写错误（test_final → test_final）
# categories = ["0_real", "1_fake"]
#
# for category in categories:
#     src_dir = os.path.join(source_base, category)
#     files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
#     random.shuffle(files)
#
#     # 处理奇数文件数量
#     split_idx = len(files) // 2
#     if len(files) % 2 != 0:
#         split_idx += random.choice([0, 1])  # 随机分配余数
#
#     val_files = files[:split_idx]
#     test_files = files[split_idx:]
#
#     val_dest = os.path.join(dest_val_base, category)
#     test_dest = os.path.join(dest_test_base, category)
#     os.makedirs(val_dest, exist_ok=True)
#     os.makedirs(test_dest, exist_ok=True)
#
#     # 复制文件并检查冲突
#     for f in val_files:
#         src = os.path.join(src_dir, f)
#         dst = os.path.join(val_dest, f)
#         if os.path.exists(dst):
#             print(f"跳过已存在文件: {dst}")
#             continue
#         shutil.copy(src, dst)
#
#     for f in test_files:
#         src = os.path.join(src_dir, f)
#         dst = os.path.join(test_dest, f)
#         if os.path.exists(dst):
#             print(f"跳过已存在文件: {dst}")
#             continue
#         shutil.copy(src, dst)
#
# print("文件划分完成！")
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
