import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob


class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label
        self.real_list = glob.glob(opt.real_list_path+"/*.png")
        self.fake_list = glob.glob(opt.fake_list_path+"/*.png")
        self.label_dict = dict()
        for i in self.real_list:
            self.label_dict[i] = 0
        for i in self.fake_list:
            self.label_dict[i] = 1
        self.total_list = self.real_list + self.fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        #print(f"Attempting to read file: {img_path}")
        label = self.label_dict[img_path]
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])(img)
        # crop images
        # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
        crops = [[transforms.Resize((224, 224))(img[:, 500:, i:i + 500]) for i in range(5)], [], []]
        crop_idx = [(28, 196), (61, 163)]
        for i in range(len(crops[0])):
            crops[1].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            crops[2].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
        img = transforms.Resize((1120, 1120))(img)

        return img, crops, label

# import torch
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# import glob
# from mtcnn import MTCNN
# import cv2
#
#
# class AVLip(Dataset):
#     def __init__(self, opt):
#         assert opt.data_label in ["train", "val"]
#         self.data_label = opt.data_label
#         self.real_list = glob.glob(opt.real_list_path + "/*.png")
#         self.fake_list = glob.glob(opt.fake_list_path + "/*.png")
#         self.real_list = sorted(self.real_list)
#         self.fake_list = sorted(self.fake_list)
#         self.label_dict = dict()
#         for i in self.real_list:
#             self.label_dict[i] = 0
#         for i in self.fake_list:
#             self.label_dict[i] = 1
#         self.total_list = self.real_list + self.fake_list
#         if torch.cuda.is_available():
#             device_str = '/GPU:0'
#         else:
#             device_str = '/CPU:0'
#         self.detector = MTCNN(device=device_str)
#
#     def __len__(self):
#         return len(self.total_list)
#
#     def __getitem__(self, idx):
#         img_path = self.total_list[idx]
#         label = self.label_dict[img_path]
#         img = torch.tensor(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), dtype=torch.float32)
#         img = img.permute(2, 0, 1)
#         # crop images
#         crops = [[transforms.Resize((224, 224))(img[:, 500:, i * 500:(i * 500) + 500]) for i in range(5)], [], []]
#
#         for i in range(len(crops[0])):
#             face_img_np = crops[0][i].permute(1, 2, 0).numpy().astype('uint8')
#             faces = self.detector.detect_faces(face_img_np)
#             if faces:
#                 face = faces[0]
#                 x, y, width, height = face['box']
#                 face_img = crops[0][i][:, y:y + height, x:x + width]
#                 crops[1].append(transforms.Resize((224, 224))(face_img))
#
#                 keypoints = face['keypoints']
#                 mouth_left = keypoints['mouth_left']
#                 mouth_right = keypoints['mouth_right']
#
#                 mouth_top = int(0.5 * face_img.shape[0])
#                 mouth_bottom = face_img.shape[0]
#                 mouth_start_x = mouth_left[0] - x
#                 mouth_end_x = mouth_right[0] - x
#
#                 mouth_region = face_img[mouth_top-5:mouth_bottom+5, mouth_start_x-5:mouth_end_x+5]
#                 if mouth_region.shape[1] > 0 and mouth_region.shape[2] > 0:
#                     crops[2].append(transforms.Resize((224, 224))(mouth_region))
#                 else:
#                     crops[2].append(transforms.Resize((224, 224))(crops[0][i]))
#             else:
#                 crops[1].append(transforms.Resize((224, 224))(crops[0][i]))
#                 crops[2].append(transforms.Resize((224, 224))(crops[0][i]))
#
#         #crops[1].append(transforms.Resize((224, 224))
#                         #(crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
#         #crops[2].append(transforms.Resize((224, 224))
#                         #(crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
#         normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
#                                          std=[0.26862954, 0.26130258, 0.27577711])
#         # 遍历 crops 列表进行归一化
#         for j in range(len(crops)):
#             for k in range(len(crops[j])):
#                 crops[j][k] = crops[j][k] / 255.0
#                 crops[j][k] = normalize(crops[j][k]).clone().detach()
#
#         # 对 img 进行归一化和尺寸调整
#         img = img / 255.0
#         img = transforms.Resize((1120, 1120))(img)
#         img = normalize(img)
#         label = torch.tensor(label, dtype=torch.float32)
#         return img, crops, label
