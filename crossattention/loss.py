import torch
import torch.nn as nn
import torch.nn.functional as F

class SAMLoss(nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, img1, img2):
        # 입력 이미지의 차원 검증
        if not img1.size() == img2.size():
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim != 4 or img1.shape[1] <= 1:
            raise ValueError("Input dimension should be BxCxHxW and n_channels should be greater than 1")

        # 이미지를 float64로 변환
        img1_ = img1.to(torch.float64)
        img2_ = img2.to(torch.float64)

        # 내적과 스펙트럼 노름 계산
        inner_product = (img1_ * img2_).sum(dim=1)  # BxHxW
        img1_spectral_norm = torch.sqrt((img1_ ** 2).sum(dim=1))
        img2_spectral_norm = torch.sqrt((img2_ ** 2).sum(dim=1))

        # 코사인 유사도 계산과 손실 계산을 위한 수치 안정성 보장
        eps = torch.finfo(torch.float64).eps
        # eps = 0
        cos_theta = inner_product / (img1_spectral_norm * img2_spectral_norm + eps)
        cos_theta = torch.clamp(cos_theta, min=0, max=1)

        # 1에서 코사인 유사도를 빼서 손실 계산
        loss = 1 - cos_theta**2

        # 배치 내 모든 이미지의 평균 손실을 반환
        return loss.mean()

# 사용 예시
if __name__ == "__main__":
    # 두 배치 이미지 텐서 생성 (예시)
    img1 = torch.rand(64, 8, 256, 256, dtype=torch.float32)  # BxCxHxW
    img2 = torch.rand(64, 8, 256, 256, dtype=torch.float32)

    # 손실 함수 인스턴스화
    sam_loss = SAMLoss()

    # 손실 계산
    loss = sam_loss(img1, img2)

    print(f'SAM Loss: {loss.item()}')