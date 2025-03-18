# GenAI-Assignment2
Generating realistic images with DCGAN and enhancing them using ESRGAN for AHFQ dataset.
# **DCGAN + ESRGAN: High-Resolution Image Generation**

This repository contains a Deep Convolutional Generative Adversarial Network (DCGAN) for generating images from random noise, followed by Enhanced Super-Resolution GAN (ESRGAN) to upscale the generated images for better visualization. The project includes a web-based demo deployed on Hugging Face Spaces.

**Project Overview:**

Train DCGAN to generate 64×64 images from the latent noise using the AFHQ (Animal Faces-HQ) dataset.

Enhance images using ESRGAN to improve resolution and clarity.

Deploy a web app to generate and display high-resolution images.

**Web App:**
You can try out the model here: Live Demo on Hugging Face Spaces
External URL: http://52.6.180.21:8501

**Dataset: AFHQ**

Source: Animal Faces (AFHQ) https://huggingface.co/datasets/huggan/AFHQ 

Description: A high-quality dataset of animal faces (Cats, Dogs, and Wild Animals).

Preprocessing: Images were resized to 64×64 before training DCGAN.

# Model Training Details:
**1. DCGAN (Deep Convolutional GAN)**

Epochs: 200

Image Size: 64×64

Latent Dimension (z_dim): 100

Optimizer: Adam (lr = 2e-4, β1 = 0.5, β2 = 0.999)

Loss Function: Binary Cross-Entropy (BCE)

Framework: PyTorch

The DCGAN was trained to generate 64×64 images. However, for better visualization, the images were later upscaled using ESRGAN. The model training took approximately 5 hours, and around epoch 50, it started generating reasonable animal images with distinguishable features.

Below are the generated images at different epochs:

Epoch1:
![fake_epoch_0](https://github.com/user-attachments/assets/5eb58cf8-766d-4f56-ac92-c2c7818861b9)

Epoch 50:
![fake_epoch_49](https://github.com/user-attachments/assets/619db893-e8a7-46d7-8f4e-eb718819efcf)

Epoch 100:
![fake_epoch_99](https://github.com/user-attachments/assets/267e8566-e081-4f67-b0ab-aefaf5b80b7d)

Epoch 150:
![fake_epoch_149](https://github.com/user-attachments/assets/66570325-75b8-4e52-97e9-ab3a8413af31)

Epoch 200:
![fake_epoch_199](https://github.com/user-attachments/assets/b19e0d92-740a-4d3e-a819-ad3d737ee971)

# Loss Curves & Training Observations
During training, the generator loss was continuously increasing, while the discriminator loss was decreasing. Although the model was learning new features, the rising generator loss indicated a potential risk of overfitting. To prevent degradation in model performance, training was stopped at 200 epochs.

**Loss Curves:**
![image](https://github.com/user-attachments/assets/6268a85a-f503-4724-a301-1948ce48e0b4)

Despite achieving reasonable results, further improvements could be made by tuning hyperparameters to enhance image quality and capture finer details. Future work could focus on adjusting the learning rate, optimizing loss balancing, and experimenting with different architectures.

**2. ESRGAN (Enhanced Super-Resolution GAN)**

Used pretrained model weights from Xintao's ESRGAN Repository

Upscaling Factor: ×4

Final Image Resolution: 256×256

Output Quality: Improved texture, sharpness, and clarity

**Sample Results**
Generated images before and after super-resolution enhancement:







