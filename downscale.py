import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_utils import ImportDataset
from torchvision.transforms import ToTensor, ToPILImage


parser = argparse.ArgumentParser(description='Downscale Images')
parser.add_argument('--downscale_factor', default=4, type=int, help='downscale factor')
# parser.add_argument('--folder_path', type=str, help='folder containing images to downscale')
# parser.add_argument('--output_folder', type=str, help='output folder for downscaled images')
opt = parser.parse_args()

DOWNSCALE_FACTOR = opt.downscale_factor
# FOLDER_PATH = opt.folder_path
# OUTPUT_FOLDER = opt.output_folder

FOLDER_PATH = 'data/REDS/val_REDS4_sharp_bicubic/X4'
OUTPUT_FOLDER = 'data/REDS/low_res_val_bicubic'

transform = transforms.Compose([
    transforms.ToTensor()
])

# Create output directory if it does not exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

dataset = ImportDataset(FOLDER_PATH, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for img, path in loader:
    # Resize image using PIL
    original_img = ToPILImage()(img[0].data.cpu())
    # print(original_img.shape)
    width, height = original_img.size
    new_width = width // DOWNSCALE_FACTOR
    new_height = height // DOWNSCALE_FACTOR
    downscaled_img = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    downscaled_img = original_img.resize((new_width, new_height), Image.Resampling.BICUBIC)

    # Save downscaled image
    
    parts = path[0].split(os.sep)
    new_file_name = f"{parts[-2]}/{os.path.basename(path[0])}"
    save_path = os.path.join(OUTPUT_FOLDER, new_file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    downscaled_img.save(save_path)

print("All images have been downscaled and saved.")
