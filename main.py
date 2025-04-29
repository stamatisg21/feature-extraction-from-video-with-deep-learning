import torch
import numpy as np
from PIL import Image
import os, cv2
from mtcnn import MTCNN
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights 

# Directories
rgb_frames_dir = "/home/stamatis/4th Assignment/BioVid_rgb_thermal/rgb/071309_w_21/PA4-020"
depth_frames_dir = "/home/stamatis/4th Assignment/depth"
thermal_frames_dir = "/home/stamatis/4th Assignment/BioVid_rgb_thermal/thermal/071309_w_21/PA4-020"
output_dir_rgb_faces = "/home/stamatis/4th Assignment/out_rgb_faces"
output_dir_thermal_faces = "/home/stamatis/4th Assignment/out_thermal_faces"
output_dir_depth_faces = "/home/stamatis/4th Assignment/out_depth_faces"
output_dir_rgb_features = "/home/stamatis/4th Assignment/rgb_features"
output_dir_thermal_features = "/home/stamatis/4th Assignment/thermal_features"
output_dir_depth_features = "/home/stamatis/4th Assignment/depth_features"

# # Load image processor and depth estimation model
# image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
# model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

# # Get list of file names in the dataset directory
# image_files = sorted(os.listdir(rgb_frames_dir))

# # Process each image in the dataset
# for image_file in image_files:
#     # Load image
#     image_path = os.path.join(rgb_frames_dir, image_file)
#     image = Image.open(image_path)
    
#     # Prepare image for the model
#     inputs = image_processor(images=image, return_tensors="pt")

#     # Perform depth estimation
#     with torch.no_grad():
#         outputs = model(**inputs)
#         predicted_depth = outputs.predicted_depth

#     # Interpolate to original size
#     prediction = torch.nn.functional.interpolate(
#         predicted_depth.unsqueeze(1),
#         size=image.size[::-1],
#         mode="bicubic",
#         align_corners=False,
#     )

#     # Convert depth map to image
#     output = prediction.squeeze().cpu().numpy()
#     # Normalize depth values to 0-255
#     output_normalized = ((output - output.min()) / (output.max() - output.min()) * 255).astype(np.uint8)
#     depth_image = Image.fromarray(output_normalized)

#     # Save depth map
#     depth_map_filename = f"{os.path.splitext(image_file)[0]}_depth_map.jpg"
#     depth_map_path = os.path.join(depth_frames_dir, depth_map_filename)
#     depth_image.save(depth_map_path)

#     print(f"Depth map saved: {depth_map_path}")

# print("Depth map generation completed.")

# Create output directories if they don't exist
os.makedirs(output_dir_rgb_faces, exist_ok=True)
os.makedirs(output_dir_thermal_faces, exist_ok=True)
os.makedirs(output_dir_depth_faces, exist_ok=True)
os.makedirs(output_dir_rgb_features, exist_ok=True)
os.makedirs(output_dir_thermal_features, exist_ok=True)
os.makedirs(output_dir_depth_features, exist_ok=True)


# # Initialize MTCNN for face detection
# detector = MTCNN()

# # Function to detect faces and save them
# def detect_and_save_faces(rgb_frame, thermal_frame, depth_frame, frame_filename):
#     # Detect faces in RGB frame
#     rgb_faces = detector.detect_faces(rgb_frame)
    
#     # Process each detected face
#     for i, face in enumerate(rgb_faces):
#         # Get bounding box coordinates
#         x, y, w, h = face['box']
        
#         # Crop faces from thermal and depth frames using RGB face coordinates
#         thermal_face = thermal_frame[y:y+h, x:x+w]
#         depth_face = depth_frame[y:y+h, x:x+w]
        
#         # Save detected faces
#         cv2.imwrite(os.path.join(output_dir_rgb_faces, f"{frame_filename}_rgb_face_{i}.jpg"), rgb_frame[y:y+h, x:x+w])
#         cv2.imwrite(os.path.join(output_dir_thermal_faces, f"{frame_filename}_thermal_face_{i}.jpg"), thermal_face)
#         cv2.imwrite(os.path.join(output_dir_depth_faces, f"{frame_filename}_depth_face_{i}.jpg"), depth_face)

# # Process each frame in the directories
# for rgb_frame_file, thermal_frame_file, depth_frame_file in zip(sorted(os.listdir(rgb_frames_dir)),
#                                                                sorted(os.listdir(thermal_frames_dir)),
#                                                                sorted(os.listdir(depth_frames_dir))):
#     # Read frames
#     rgb_frame = cv2.imread(os.path.join(rgb_frames_dir, rgb_frame_file))
#     thermal_frame = cv2.imread(os.path.join(thermal_frames_dir, thermal_frame_file))
#     depth_frame = cv2.imread(os.path.join(depth_frames_dir, depth_frame_file))
    
#     # Get the filename for the current frame
#     frame_filename = os.path.splitext(rgb_frame_file)[0]
    
#     # Detect faces and save them
#     detect_and_save_faces(rgb_frame, thermal_frame, depth_frame, frame_filename)

# Load pre-trained ResNet model
# Load pre-trained ResNet model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features from frames
def extract_features(frames_dir, output_dir):
    for frame_file in os.listdir(frames_dir):
        # Load frame
        frame_path = os.path.join(frames_dir, frame_file)
        frame = Image.open(frame_path).convert("RGB")
        
        # Apply transformation
        frame_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension
        
        # Extract features using ResNet
        with torch.no_grad():
            features = model(frame_tensor)
        
        # Convert feature tensor to numpy array
        feature_vector = features.squeeze().numpy()
        
        # Save feature vector as CSV
        output_csv_path = os.path.join(output_dir, os.path.splitext(frame_file)[0] + ".csv")
        np.savetxt(output_csv_path, feature_vector, delimiter=",")

# Extract features for RGB frames
extract_features(rgb_frames_dir, output_dir_rgb_features)

# Extract features for thermal frames
extract_features(thermal_frames_dir, output_dir_thermal_features)

# Extract features for depth frames
extract_features(depth_frames_dir, output_dir_depth_features)