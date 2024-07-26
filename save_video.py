import cv2
import os

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs
out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (320, 180))  # Adjust the frame size
# out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (1280, 720))  # Adjust the frame size


path = '/home/gridsan/jmeindl/SRGAN/data/REDS/pred_val/003'
# path = '/home/gridsan/jmeindl/SRGAN/data/REDS/val_REDS4_sharp_bicubic/X4/001'
# Read each file and write to video
for filename in sorted(os.listdir(path)):
    if filename.endswith(".png"):
        img = cv2.imread(os.path.join(path, filename))
        out.write(img)

out.release()