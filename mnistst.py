import os
from torchvision import datasets

output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)

for i in range(10):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)

mnist = datasets.MNIST(root="./data", train=True, download=True)

for idx, (img, label) in enumerate(mnist):
    filename = f"{idx}.png"
    save_path = os.path.join(output_dir, str(label), filename)
    img.save(save_path)   # ✅ 바로 저장 가능

print("MNIST 이미지 저장 완료!")
