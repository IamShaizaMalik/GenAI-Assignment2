import os
import subprocess
import torch
import torch.nn as nn
import torchvision
import streamlit as st
import urllib.request
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# ===============================
# 1. Ensure Basicsr Import Works 
# ===============================
def fix_basicsr_import():
    file_path = "/usr/local/lib/python3.10/site-packages/basicsr/data/degradations.py"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            file_content = file.read()
        new_content = file_content.replace(
            "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
            "from torchvision.transforms._functional_tensor import rgb_to_grayscale"
        )
        with open(file_path, "w") as file:
            file.write(new_content)
        print(" Basicsr import issue fixed.")
    else:
        print(" Basicsr degradation file not found!")

fix_basicsr_import()

# ===============================
# 2. Download & Load ESRGAN Model
# ===============================
os.makedirs("experiments/pretrained_models", exist_ok=True)
model_path = "experiments/pretrained_models/RealESRGAN_x4plus.pth"
if not os.path.exists(model_path):
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    print("Downloading ESRGAN model...")
    urllib.request.urlretrieve(url, model_path)
    print(" Download complete!")
else:
    print("ESRGAN model already exists.")

# ===============================
# 3. Define DCGAN Generator Model
# ===============================
class GeneratorDCGAN(nn.Module):
    def __init__(self, noise_dim, output_channels, feature_map_gen):
        super(GeneratorDCGAN, self).__init__()
        self.generator_network = nn.Sequential(
            self._gen_layer(noise_dim, feature_map_gen * 16, 4, 1, 0),
            self._gen_layer(feature_map_gen * 16, feature_map_gen * 8, 4, 2, 1),
            self._gen_layer(feature_map_gen * 8, feature_map_gen * 4, 4, 2, 1),
            self._gen_layer(feature_map_gen * 4, feature_map_gen * 2, 4, 2, 1),
            nn.ConvTranspose2d(feature_map_gen * 2, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _gen_layer(self, input_dim, output_dim, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

    def forward(self, latent_vector):
        return self.generator_network(latent_vector)

# ===============================
# 4. Load Pretrained DCGAN Generator
# ===============================
dcgan_model_path = "generator_epoch140_210.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 50
channels_img = 3
features_g = 64

gen = GeneratorDCGAN(z_dim, channels_img, features_g).to(device)
gen.load_state_dict(torch.load(dcgan_model_path, map_location=device))
gen.eval()

# ===============================
# 5. Generate Image using DCGAN
# ===============================
def generate_image():
    noise = torch.randn((1, z_dim, 1, 1)).to(device)
    with torch.no_grad():
        generated_image = gen(noise).cpu()
    
    generated_image = (generated_image + 1) / 2
    generated_image = torch.clamp(generated_image, 0, 1)
    save_path = "generated_image.png"
    torchvision.utils.save_image(generated_image, save_path)

    return save_path


# ===============================
# 6. Clone & Setup Real-ESRGAN
# ===============================
if not os.path.exists("Real-ESRGAN"):
    subprocess.run(["git", "clone", "https://github.com/xinntao/Real-ESRGAN.git"])

os.chdir("Real-ESRGAN")

# Install dependencies
subprocess.run(["pip", "install", "-r", "requirements.txt"])
subprocess.run(["python", "setup.py", "develop"])

os.chdir("..")  # Move back to the main directory


# ===============================
# 7. Enhance Image using ESRGAN
# ===============================


# Ensure basicsr is installed (run separately if needed)
# !pip install basicsr

from basicsr.archs.rrdbnet_arch import RRDBNet  # Import RRDBNet correctly

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ESRGAN model (Real-ESRGAN)
generator = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32).to(device)

# Load pretrained weights
weights_path = "RealESRGAN_x4plus.pth"

# Load checkpoint
checkpoint = torch.load(weights_path, map_location=device)

# Check if 'params_ema' exists and use that, otherwise use the full checkpoint
state_dict = checkpoint['params_ema'] if 'params_ema' in checkpoint else checkpoint

# Fix wrapped parameter keys (e.g., remove 'module.' prefix if necessary)
new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

# Load fixed state dict into the model
generator.load_state_dict(new_state_dict, strict=False)  # Use strict=False to avoid minor mismatches
generator.eval()

print(" Model loaded successfully with 'params_ema' handling!")

# ===============================
# 8. Enhance Image using ESRGAN
# ===============================
def enhance_image(image_path):
    lr_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    lr_tensor = transform(lr_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        sr_tensor = generator(lr_tensor)
    
    sr_image = sr_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    sr_image = np.clip(sr_image, 0, 1)
    
    return lr_image, sr_image

# ===============================
# 9. Streamlit UI
# ===============================
st.title("DCGAN + ESRGAN Image Generator")
st.write("Generate an image using DCGAN and enhance it with ESRGAN.")

col1, col2 = st.columns([1, 1])

if st.button("Generate & Enhance Image", use_container_width=True):
    image_path = generate_image()
    lr_img, sr_img = enhance_image(image_path)
    
    col1.image(lr_img, caption="Generated Image", use_container_width=True)
    col2.image(sr_img, caption="Super-Resolved Image", use_container_width=True)
