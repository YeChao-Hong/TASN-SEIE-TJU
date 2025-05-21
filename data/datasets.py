import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
from mtcnn import MTCNN
import cv2
from facenet_pytorch import MTCNN



class AVLip(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label
        self.real_list = glob.glob(opt.real_list_path + "/*.png")
        self.fake_list = glob.glob(opt.fake_list_path + "/*.png")
        self.real_list = sorted(self.real_list)
        self.fake_list = sorted(self.fake_list)
        self.label_dict = dict()
        for i in self.real_list:
            self.label_dict[i] = 0
        for i in self.fake_list:
            self.label_dict[i] = 1
        self.total_list = self.real_list + self.fake_list
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = MTCNN(device=device_str)  # 缩进和上面一致，都是4个空格
    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.label_dict[img_path]
        img = torch.tensor(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        # crop images
        crops = [[transforms.Resize((224, 224))(img[:, 500:, i * 500:(i * 500) + 500]) for i in range(5)], [], []]
        crop_idx = [(28, 196), (61, 163)]

        for i in range(len(crops[0])):
            face_tensor = crops[0][i]  # C×H×W tensor
            face_np = face_tensor.permute(1, 2, 0).numpy().astype('uint8')

            # 用 facenet_pytorch 的 MTCNN.detect 获取人脸框和关键点
            boxes, probs, landmarks = self.detector.detect(face_np, landmarks=True)

            if boxes is not None and len(boxes) > 0:
                # 只处理第一个检测到的人脸
                x, y, x2, y2 = boxes[0]
                w = int(x2 - x)
                h = int(y2 - y)
                x = int(x)
                y = int(y)
                # 计算正方形边长和中心
                side = max(w, h)
                cx = x + w // 2
                cy = y + h // 2
                # 计算正方形框坐标
                x1 = max(0, cx - side // 2)
                y1 = max(0, cy - side // 2)
                x2 = x1 + side
                y2 = y1 + side

                H, W = face_tensor.shape[1], face_tensor.shape[2]
                if x2 > W:
                    x2 = W
                    x1 = W - side
                if y2 > H:
                    y2 = H
                    y1 = H - side

                sq_face = face_tensor[:, y1:y2, x1:x2]  # 正方形脸部区域
                crops[1].append(transforms.Resize((224, 224))(sq_face))

                # 关键点 landmarks[0] 顺序是 左眼、右眼、鼻子、左嘴角、右嘴角
                kps = landmarks[0]
                ml = kps[3]  # mouth_left (x,y)
                mr = kps[4]  # mouth_right (x,y)
                mlx, mly = int(ml[0]), int(ml[1])
                mrx, mry = int(mr[0]), int(mr[1])

                # 嘴部区域的上边界，参考人脸框
                top = y + int(0.5 * h)
                mw = mrx - mlx
                mh = y + h - top
                mside = int(1.1*max(mw, mh))
                mcx = mlx + mw // 2
                mcy = top + mh // 2

                mx1 = max(0, mcx - mside // 2)
                my1 = max(0, mcy - mside // 2)
                mx2 = mx1 + mside
                my2 = my1 + mside

                if mx2 > W:
                    mx2 = W
                    mx1 = W - mside
                if my2 > H:
                    my2 = H
                    my1 = H - mside

                sq_mouth = face_tensor[:, my1:my2, mx1:mx2]
                crops[2].append(transforms.Resize((224, 224))(sq_mouth))

            else:
                # 无检测人脸时回退到原始分块
                crops[1].append(transforms.Resize((224, 224))
                                (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
                crops[2].append(transforms.Resize((224, 224))
                                (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))

        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                         std=[0.26862954, 0.26130258, 0.27577711])
        # 归一化
        for j in range(len(crops)):
            for k in range(len(crops[j])):
                crops[j][k] = crops[j][k] / 255.0
                crops[j][k] = normalize(crops[j][k]).clone().detach()

        img = img / 255.0
        img = transforms.Resize((1120, 1120))(img)
        img = normalize(img)
        label = torch.tensor(label, dtype=torch.float32)
        return img, crops, label

