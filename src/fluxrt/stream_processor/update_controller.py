import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as FF

import time


class UpdateController:
    def __init__(
        self,
        height: int,
        width: int,
        compression_ratio: int,
        text_seq_len: int = 512,
        device="cuda",
        dtype=torch.bfloat16,
        reset_period=None,
        reference_image_seq_len: int = None,
    ):
        self.height = height
        self.width = width
        self.compression_ratio = compression_ratio
        self.text_seq_len = text_seq_len
        self.device = device
        self.dtype = dtype
        self.reference_image_seq_len = reference_image_seq_len

        self.mask_height = height // compression_ratio
        self.mask_width = width // compression_ratio

        self.cached_frame = torch.zeros(1, 3, height, width, device=device, dtype=dtype)

        if reset_period is not None:
            self.previous_reset = time.time()

        self.reset_period = reset_period
        self.requires_reset = False
        self.text_is_valid = False
        self.reference_image_is_valid = False

    def update_and_get_mask(self, frame: torch.Tensor):
        """
        Args:
            frame: tensor of shape (1, 3, h, w)
        Returns:
            mask: binary bool tensor of shape (1, h // compression_ratio, w // compression_ratio)
        """
        if self.reset_period is not None:
            if time.time() - self.previous_reset > self.reset_period:
                self.requires_reset = True

        if self.requires_reset:
            self.cached_frame = frame
            self.previous_reset = time.time()
            self.requires_reset = False
            return torch.ones(
                1,
                self.mask_height,
                self.mask_width,
                device=self.device,
                dtype=torch.bool,
            )

        frame_blurred = F.gaussian_blur(frame, kernel_size=3, sigma=0.5)
        cached_blurred = F.gaussian_blur(self.cached_frame, kernel_size=3, sigma=0.5)

        difference = (cached_blurred - frame_blurred) ** 2
        difference = difference.mean(dim=1, keepdim=True)

        difference_mask = torch.max_pool2d(
            difference, (self.compression_ratio, self.compression_ratio)
        )
        difference_mask = difference_mask > 0.1
        difference_mask_dilated = (
            FF.max_pool2d(difference_mask.float(), kernel_size=3, stride=1, padding=1)
            > 0
        )
        difference_mask_dilated = (
            FF.max_pool2d(
                difference_mask_dilated.float(), kernel_size=3, stride=1, padding=1
            )
            > 0
        )

        difference_mask_upsampled = FF.interpolate(
            difference_mask_dilated.float(),
            size=(self.height, self.width),
            mode="nearest",
        )
        self.cached_frame = torch.where(
            difference_mask_upsampled.to(torch.bool).expand(-1, 3, -1, -1),
            frame,
            self.cached_frame,
        )

        return difference_mask_dilated.squeeze(1)

    def use_text_mask(self):
        if self.text_is_valid:
            mask = torch.zeros(
                1, self.text_seq_len, device=self.device, dtype=self.dtype
            )
        else:
            mask = torch.ones(
                1, self.text_seq_len, device=self.device, dtype=self.dtype
            )
            self.text_is_valid = True

        return mask

    def use_reference_image_mask(self):
        if self.reference_image_seq_len is None:
            return None

        if self.reference_image_is_valid:
            mask = torch.zeros(
                1, self.reference_image_seq_len, device=self.device, dtype=self.dtype
            )
        else:
            mask = torch.ones(
                1, self.reference_image_seq_len, device=self.device, dtype=self.dtype
            )
            self.reference_image_is_valid = True

        return mask

    def reset_cache(self):
        self.requires_reset = True
        self.text_is_valid = False
        self.reference_image_is_valid = False
