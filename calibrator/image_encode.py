from ytools.tensorrt.quant import CalibratorDatasetObject
import torch
from torchvision.transforms import transforms
import numpy as np
import cv2
import os
from typing import cast


class ImageEncodeCalibratorDataset(CalibratorDatasetObject):
    def __init__(
        self,
        calibration_image_folder,
        input_shapes=[(-1, 3, 1024, 1024)],
        names=["image"],
        batch_size=1,
        skip_frame=1,
        dataset_limit=1 * 1000,
        do_norm=False,
    ):
        super().__init__()

        self.image_folder = calibration_image_folder
        self.names = names
        (_, _, self.height, self.width) = input_shapes[0]

        self.dataset_limit = dataset_limit
        self.skip_frame = skip_frame

        self.batch_size = batch_size

        if do_norm:
            self.norm = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.norm = lambda x: x

        self.init_data()

    def init_data(self):
        imgs = self.load_pre_data(
            self.image_folder, size_limit=self.dataset_limit, skip=self.skip_frame
        )  # (k*b+m,c,h,w)
        datasize = len(imgs) // self.batch_size * self.batch_size // self.batch_size

        img_batchs = np.split(
            imgs[: datasize * self.batch_size, ...],
            datasize,
            axis=0,
        )

        self.datasets = [[np.ascontiguousarray(img_batch)] for img_batch in img_batchs]
        print(
            f"finish init calibration in cpu: datasize={len(self)}*{self.shape(0)}, type={self.dtype(0)}"
        )

    def preprocess(self, np_image: np.ndarray, bgr=True):
        """
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        """
        if bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image

        input_image = cv2.resize(np_image_rgb, (self.width, self.height))
        input_image = self.norm(
            (torch.from_numpy(input_image) / 255.0).moveaxis(-1, 0)
        )  # type: torch.Tensor

        return input_image.cpu().numpy()

    def load_pre_data(self, imgs_folder, size_limit=0, skip=20):
        img_names = os.listdir(imgs_folder)
        imgs = []
        idx = -1
        for img_name in img_names:
            if not cast(str, img_name).endswith(".jpg") and not cast(
                str, img_name
            ).endswith(".png"):
                continue
            img_path = os.path.join(imgs_folder, img_name)
            img = cv2.imread(img_path)
            idx += 1

            if idx % skip == 0:
                print(f"load {img_path}")
                imgs.append(self.preprocess(img))
                idx = 0

            if size_limit > 0 and len(imgs) >= size_limit:
                break

        assert len(imgs) > 0, "empty datas"

        return np.stack(imgs)
