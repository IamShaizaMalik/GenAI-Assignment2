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

The DCGAN generates 64×64 images, but for better visualization, they were upscaled using ESRGAN.
The model was trained for 200 epochs and it took around 5 hours in training and started to displaying the reasonable animal pictures with fine features. The images obtained at various epochs are:
Epoch1:
![fake_epoch_0](https://github.com/user-attachments/assets/5eb58cf8-766d-4f56-ac92-c2c7818861b9)


**2. ESRGAN (Enhanced Super-Resolution GAN)**

Used pretrained model weights from Xintao's ESRGAN Repository

Upscaling Factor: ×4

Final Image Resolution: 256×256

Output Quality: Improved texture, sharpness, and clarity

**Sample Results**
Generated images before and after super-resolution enhancement:







