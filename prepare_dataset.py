import os
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ================= CONFIGURATION =================
VIDEO_ROOT = 'archive'
OUTPUT_ROOT = 'datasets/CDF2'

FRAME_INTERVAL = 10 
FACE_SIZE = 224      
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =================================================

# Lowering thresholds to 0.5 because c23 compression is very "blurry"
mtcnn = MTCNN(
    image_size=FACE_SIZE, margin=20, post_process=True, 
    device=DEVICE, selection_method='largest',
    thresholds=[0.5, 0.6, 0.6] 
)

def get_video_paths():
    return [
        (os.path.join(VIDEO_ROOT, 'Celeb-real'), 'Celeb-real'),
        (os.path.join(VIDEO_ROOT, 'Celeb-synthesis'), 'Celeb-synthesis'),
        (os.path.join(VIDEO_ROOT, 'YouTube-real'), 'YouTube-real'),
        # (os.path.join(VIDEO_ROOT, 'manipulated_sequences/NeuralTextures/c23/videos'), 'NT'),
        # (os.path.join(VIDEO_ROOT, 'original_sequences/youtube/c23/videos'), 'real'),
    ]

def process_videos():
    video_tasks = get_video_paths()

    for source_path, target_name in video_tasks:
        if not os.path.exists(source_path):
            print(f"❌ ERROR: Path does not exist: {source_path}")
            continue

        videos = [f for f in os.listdir(source_path) if f.endswith('.mp4')]
        print(f"\n--- Found {len(videos)} videos in {target_name} ---")
        
        for video_file in tqdm(videos[:40]): # Testing with first 10 videos
            video_id = video_file.split('.')[0]
            video_full_path = os.path.join(source_path, video_file)
            
            save_dir = os.path.join(OUTPUT_ROOT, target_name, video_id)
            os.makedirs(save_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_full_path)
            if not cap.isOpened():
                print(f"⚠️ Could not open video: {video_full_path}")
                continue

            frame_count = 0
            saved_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_count % FRAME_INTERVAL == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    
                    face_save_path = os.path.join(save_dir, f"{saved_count:03d}.png")
                    
                    # Manual detection check for debugging
                    boxes, _ = mtcnn.detect(img)
                    if boxes is not None:
                        mtcnn(img, save_path=face_save_path)
                        saved_count += 1
                
                frame_count += 1
                #if saved_count >= 10: break # Keep it small for the test

            cap.release()
            if saved_count == 0:
                print(f"❗ No faces found in {video_file} (Checked {frame_count} frames)")

if __name__ == "__main__":
    print(f"Running on: {DEVICE}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    process_videos()
