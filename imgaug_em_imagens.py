import imgaug.augmenters as iaa
import cv2
import os
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 0.5)),
    iaa.ContrastNormalization((0.9, 1.1)),
    iaa.Affine(rotate=(-10, 10))
])
input_dir = r"C:\Users\vmarq\Documents\Aprendizado de Maquina\Modelo_CNN\dataset_train\Barbara"
output_dir = r"C:\Users\vmarq\Documents\Aprendizado de Maquina\Modelo_CNN\dataset_train\Barbara_augmented"
os.makedirs(output_dir, exist_ok=True)
for img_name in os.listdir(input_dir):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        for i in range(4):  # Gerar 4 variações por imagem
            augmented = seq(image=image)
            cv2.imwrite(os.path.join(output_dir, f"aug_{i}_{img_name}"), augmented)