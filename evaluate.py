import os
import numpy as np
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import faiss
import clip
from tqdm import tqdm #เพื่อดูแถบความคืบหน้า

VAL_PATH = r"data/val"       
TRAIN_PATH = r"data/train"  

EMB_FILE = "clip_embeddings.npy"
FN_FILE = "clip_filenames.npy"
INDEX_FILE = "clip_image_index.faiss"


K = 100 # 👈 วัดผล 100 อันดับแรก 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# โหลด Index และ Filenames
try:
    index = faiss.read_index(INDEX_FILE)
    # path ในไฟล์เป็น relative กับ TRAIN_PATH แปลงเป็น absolute เพื่อเทียบกับ query ได้
    rel_filenames = np.load(FN_FILE, allow_pickle=True)
    filenames = [os.path.abspath(os.path.join(TRAIN_PATH, rel)) for rel in rel_filenames]
    print(f"โหลด index และไฟล์สำเร็จ (มี {index.ntotal} รูป)")
except Exception as e:
    print(f" ไม่พบไฟล์ดัชนี ({e})")
    exit()

# - Helper 
def load_rgb(image_source) -> Image.Image:
    try:
        with Image.open(image_source) as im:
            im = ImageOps.exif_transpose(im)
            return im.convert("RGB")
    except Exception:
        return None

def get_embedding(image: Image.Image) -> np.ndarray:
    if image is None: return None
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    return image_features.squeeze(0).cpu().numpy().astype("float32")

# ======== 2. ฟังก์ชันสำคัญ: ตัวแยกแยะ "คลาส" (ต้องแก้เอง) ========

def get_class_from_path(path: str) -> str:
    try:
        # โครงสร้าง .../val/class_A/file.jpg
        # os.path.dirname(path) = .../val/class_A
        # os.path.basename(...) =class_A
        class_name = os.path.basename(os.path.dirname(path))
        return class_name
    except Exception:
        return "unknown" 

print(f"โหลดโมเดลและฟังก์ชันพร้อม")
print(f"--- เริ่มการประเมินผล (Mean Precision@{K}) ---")

# วนลูปประเมินผลในโฟลเดอร์

all_precision_scores = []
query_files = []

# ค้นหาไฟล์ทั้งหมดใน VAL_PATH
for root, _, files in os.walk(VAL_PATH):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            query_files.append(os.path.join(root, fname))

if not query_files:
    print(f"ไม่พบไฟล์รูปภาพใดๆ ในโฟลเดอร์ '{VAL_PATH}'")
    exit()

print(f"พบรูปภาพสำหรับทดสอบ (จำนวนรูปที่ใช้ทดสอบ) ทั้งหมด {len(query_files)} รูป")

# ใช้ tqdm แสดงแถบความคืบหน้า
for query_path in tqdm(query_files, desc="กำลังประเมินผล"):
    
    # หา "เฉลย" ของรูปนี้
    expected_class = get_class_from_path(query_path)
    if expected_class == "unknown":
        print(f"ข้ามไฟล์:ไม่สามารถหาคลาสได้จาก {query_path}")
        continue

    # สร้าง Embedding ของ จำนวนรูปที่ใช้ทดสอบ
    query_img = load_rgb(query_path)
    if query_img is None:
        print(f"ข้ามไฟล์:เปิดรูปไม่ได้ {query_path}")
        continue
        
    q_emb = get_embedding(query_img).reshape(1, -1)
    faiss.normalize_L2(q_emb) # Normalize สำหรับ Cosine Sim

    # ค้นหา 100 อันดับแรก
    D, I = index.search(q_emb, K + 1)
    
    # 4. ตรวจคำตอบ
    correct_count = 0
    result_indices = I[0] # indexของผลลัพธ์ 100 อันดับ

    for i in result_indices:
        result_path = filenames[i]
        result_class = get_class_from_path(result_path)
        
        # กรองกรณีที่รูป จำนวนรูปที่ใช้ทดสอบ อยู่ใน train 
        if result_path == os.path.abspath(query_path):
            continue
            
        # ถ้าคลาสของผลลัพธ์ตรงกับคลาสที่คาดหวัง = ถูกต้อง
        if result_class == expected_class:
            correct_count += 1
            
    # 5. คำนวณ Precision สำหรับ จำนวนรูปที่ใช้ทดสอบ นี้
    p_at_k = correct_count / K
    all_precision_scores.append(p_at_k)

# ======== 4. สรุปผล ========

if not all_precision_scores:
    print("ไม่สามารถคำนวณคะแนนได้เลย (อาจจะหาคลาสไม่เจอ?)")
else:
    mean_precision_at_k = np.mean(all_precision_scores)
    
    print("\nสรุปผลการประเมิน Clip")
    print(f"จำนวนรูปที่ใช้ทดสอบ: {len(all_precision_scores)}")
    print(f"จำนวนอันดับที่พิจารณา:   {K}")
    print(f"Mean Precision: {mean_precision_at_k * 100:.2f} %")
    print(f"โดยเฉลี่ยแล้ว ใน {K} อันดับแรกที่ระบบค้นหามาให้ มีความถูกต้อง {mean_precision_at_k * 100:.2f} %")