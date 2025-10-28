import streamlit as st
import os
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image, ImageFile, ImageOps
import torch
import faiss
import clip
ImageFile.LOAD_TRUNCATED_IMAGES = True #กัน error ถ้าไฟล์เสีย code ไม่ล่ม

# ตั้งค่า Path 
BASE_PATH = "data/train"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EMB_FILE = "clip_embeddings.npy"     
FN_FILE = "clip_filenames.npy"    
INDEX_FILE = "clip_image_index.faiss" 

# ตรวจสอบและเลือก (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ใช้แปลง path เนื่องจาก อัพลง streamlit
def to_posix(p) -> str:
    if isinstance(p, (str, Path)):
        return str(p).replace("\\", "/")
    return p

@st.cache_resource  # โหลดโมเดล CLIP (Cache ไว้)
def load_model():
    st.write(" กำลังโหลดโมเดล CLIP 'ViT-B/32' ")
    # โหลดโมเดลด้วย clip.load()
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    st.write(" โหลดโมเดล CLIP สำเร็จ ")
    return model, preprocess

model, preprocess = load_model()

# ฟังก์ชัน จัดการรูปภาพ 

def load_rgb(image_source) -> Image.Image:
    try:
        if isinstance(image_source, (str, Path)):

            with Image.open(to_posix(image_source)) as im:
                im = ImageOps.exif_transpose(im)
                return im.convert("RGB")
        else:
            if hasattr(image_source, "seek"):
                try:
                    image_source.seek(0)
                except Exception:
                    pass
            with Image.open(image_source) as im:
                im = ImageOps.exif_transpose(im)
                return im.convert("RGB")   #แปลงภาพเป็นโหมด RGB
    except Exception as e:
        st.error(f"ไม่สามารถเปิดไฟล์รูปได้: {e}")
        return None

def get_embedding(image: Image.Image) -> np.ndarray:  #แปลงเป็น vector

    if image is None:
        return None
        
    # ใช้ preprocess ที่ได้มาจาก clip.load()
    image_input = preprocess(image).unsqueeze(0).to(device)  #resize, normalize ตามที่โมเดลต้องการ.เพิ่มมิติ batch.ย้ายไป GPU หรือ CPU
    
    with torch.no_grad(): #เนื่องจาก ไม่ได้ train ทำให้ไม่ต้องไปคำนวณ gradient
        image_features = model.encode_image(image_input) # ใช้ แปลงภาพเป็นเวกเตอร์
        
    return image_features.squeeze(0).cpu().numpy().astype("float32")  #squeeze เพื่อให้เหลือ (512,) จาก (1,512)
    #เนื่องจาก FAISS ใช้ float32 จึงกำหนดชนิดตัวเลขให้เป็น float32 และคืนเป็น array ของ numpy 

def _list_all_images(base_path: str):
    all_files = []
     # วนลูปหาไฟล์ภาพทั้งหมดใน data/train
    for root, _, files in os.walk(base_path):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                all_files.append(os.path.join(root, fname))
    return all_files

def _abs_from_rel(base_path: str, rel: str) -> str:
    return to_posix(os.path.join(base_path, rel))

#  สร้าง/โหลด ดัชนี FAISS (Cache ไว้)
@st.cache_data(show_spinner=False)
def create_or_load_index(base_path: str):
    base_path = to_posix(Path(base_path).resolve())


    if all(os.path.exists(p) for p in (INDEX_FILE, EMB_FILE, FN_FILE)):
        try:
            index = faiss.read_index(INDEX_FILE)
            embeddings = np.load(EMB_FILE)
            rel_filenames = np.load(FN_FILE, allow_pickle=True)  # เก็บเป็น relative
            # สร้าง absolute สำหรับแสดงผล/โหลดภาพภายหลัง
            abs_filenames = np.array([_abs_from_rel(base_path, str(r)) for r in rel_filenames])
            st.write(f" โหลดสำเร็จ (มี {index.ntotal} รูป, ขนาด {index.d},)")
            return index, embeddings, abs_filenames
        except Exception as e:
            st.warning(f"โหลดไฟล์ดัชนีไม่สำเร็จ จะสร้างใหม่: {e}")

    # ถ้าไม่เจอ
    st.write(f" กำลังสร้างCLIP ใหม่จาก '{base_path}'")
    
    embeddings_list, filenames = [], []
    all_files = _list_all_images(base_path)

    if not all_files:
        st.error(f"ไม่พบไฟล์รูปภาพใน: {base_path}")
        return None, None, None
    

    embeddings_list, abs_filenames = [], []

    progress_bar = st.progress(0, text="กำลังประมวลผลรูปภาพด้วย CLIP")
    total_files = len(all_files)
    #แปลงเป็น embedding
    for i, path in enumerate(all_files):
        try:
            img = load_rgb(path)
            if img is not None:
                emb = get_embedding(img)
                if emb is not None:
                    embeddings_list.append(emb)
                    abs_filenames.append(to_posix(path))
        except Exception as e:
            st.warning(f" ข้ามไฟล์เสีย: {path} ({e})")
        
        progress_bar.progress(i / total, text=f"ประมวลผล: {i}/{total}")

    progress_bar.empty()

    if not embeddings_list:
        st.error("ไม่สามารถสร้าง embedding ได้")
        return None, None, None

    embeddings = np.vstack(embeddings_list) #embeddings ทั้งหมดเป็นเมทริกซ์ขนาด (N, 512)
    
    #  Normalize แต่ละเวกเตอร์ให้มี length = 1 เพิ่อไปใช้สำหรับ cosine similarity
    faiss.normalize_L2(embeddings) 

    st.write(f" ประมวลผลเสร็จ: {embeddings.shape[0]} รูป, ขนาด {embeddings.shape[1]}-d")
    st.write("กำลังสร้าง FAISS")

    #  เปลี่ยนมิติเป็น 512 และใช้ IndexFlatIP
    index = faiss.IndexFlatIP(embeddings.shape[1]) # สร้างกล่องไว้เก็บเวกเตอร์ทุกภาพ
    index.add(embeddings)  # อาเวกเตอร์ของรูปทั้งหมด ใส่เข้าไปในดัชนี FAISS


    rel_filenames = []
    for p in filenames:  # p เป็น absolute path ที่เราใช้ตอนประมวลผล
        rel = os.path.relpath(p, start=base_path)   # เป็น path ใต้ BASE_PATH
        rel_filenames.append(to_posix(rel))

    # save embed และ fn เพื่อนำไปใช้ครั้งต่อไป
    np.save(EMB_FILE, embeddings)  
    np.save(FN_FILE, np.array(rel_filenames)) #เซฟรายชื่อไฟล์รูป เพื่อรู้ว่า vector นี้มาจากรูปอะไร
    faiss.write_index(index, INDEX_FILE) #เก็บโครงสร้างภายในของ FAISS

    st.success(f" สร้างและบันทึก ( {index.ntotal} รูป) สำเร็จ")
    return index, embeddings, filenames

#  โหลดข้อมูล 
base_abs = Path(BASE_PATH).resolve()
index, embeddings, abs_filenames = create_or_load_index(str(base_abs))


st.title(" Image Search Engine (CLIP + FAISS)")
st.write("อัปโหลดรูปภาพเพื่อค้นหารูปที่ 'สถานที่เดียวกัน' ")

if index is None:
    st.error("ไม่สามารถเริ่มได้เนื่องจากไม่มีindex")
    st.stop()

uploaded_file = st.file_uploader(
    "เลือกรูปภาพที่ต้องการค้นหา", 
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

#  ปรับ Threshold (ค่า Cosine Sim 0.0 - 1.0)
threshold = st.slider(
    "ตั้งค่าเกณฑ์ความมั่นใจ", 
    min_value=0.1, max_value=1.0, value=0.80, step=0.01,
)

# 
if uploaded_file is not None:
    query_img = load_rgb(uploaded_file)
    if query_img:
        st.subheader("รูปต้นฉบับ :")
        st.image(query_img, width=200)

        # สร้าง Embedding และ Normalize
        q_emb = get_embedding(query_img).reshape(1, -1).astype("float32")
        faiss.normalize_L2(q_emb)  #ทำให้เวกเตอร์แต่ละตัวมีความยาว

        k = index.ntotal #จำนวนรูปทั้งหมดในฐานข้อมูล
        
        # D = Distances คือค่า Cosine, I = ลำดับ index
        D, I = index.search(q_emb, k)   #ให้ FAISS หารูปที่ “คล้ายกับ q_emb ทั้งหมด [[]] , [[]]

        D_list = D[0] #ให้เป็น 1 D []
        I_list = I[0]

        # แปลงค่า\ (D คือ Cosine Sim อยู่แล้ว)
        # แปลงเป็น % และกรอง
        filtered_results = [] #กรองเฉพาะรูปที่ “คล้าย” ตามเกณฑ์
        for i, conf_score in zip(I_list, D_list):
            # ไม่เอารูปตัวเอง (ถ้าเผอิญรูป query อยู่ใน index)
            if conf_score < 0.9999 and conf_score >= threshold:
                filtered_results.append((abs_filenames[idx], float(sim) * 100.0))

        # เรียงลำดับจากมากไปน้อย
        filtered_results.sort(key=lambda x: x[1], reverse=True)

        # แสดงผลลัพธ์
        st.subheader(f"ผลการค้นหา (พบ {len(filtered_results)} รูป ที่สถานที่เดียวกัน ≥ {threshold*100:.0f}%):")

        if not filtered_results:
            st.warning(" ไม่พบภาพที่สถานที่เดียวกันเกิน threshold ที่ตั้งไว้")
        else:
            num_cols = 4
            cols = st.columns(num_cols)
            for j, (path, conf) in enumerate(filtered_results):
                with cols[j % num_cols]:
                    try:
                        img_result = load_rgb(path)
                        if img_result:
                            st.image(img_result, caption=f"{conf:.1f}%")
                            st.caption(os.path.basename(path), help=path)
                    except Exception:
                        st.error(f"Error loading {path}")