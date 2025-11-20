# ===================== BLIP v1 captioning (no prompt, 5 diverse captions for BM25) =====================
!pip -q install "transformers>=4.44" "accelerate>=0.34" pillow tqdm

import os, io, re, json, glob, zipfile, time
from typing import List, Dict, Set
from PIL import Image
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from config.config import Config

# ---------------- Config ----------------
ZIP_GLOB        = "/content/flickr30k_part_*_of_10.zip"
OUT_JSON        = "/content/flickr30k_blip_v1_diverse.json"
MODEL_ID        = Config.HF_MODEL_BLIP
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CAPTIONS    = 5
MAX_NEW_TOKENS  = 40
SAVE_EVERY      = 10
# ----------------------------------------

def is_image_name(name): return name.lower().endswith((".jpg",".jpeg",".png",".webp",".bmp"))
def normalize_caption(s): return re.sub(r"\s+", " ", s.strip().lower()).rstrip(" .")

def save_json_atomic(data, path):
    tmp = path + ".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
    os.replace(tmp,path)

def load_json_or_empty(p): return json.load(open(p,"r",encoding="utf-8")) if os.path.exists(p) else {}

@torch.inference_mode()
def make_model(model_id, device="cuda"):
    dtype = torch.float16 if device=="cuda" else torch.float32
    proc  = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
    if device=="cuda": torch.backends.cuda.matmul.allow_tf32=True
    model.eval()
    return proc, model

@torch.inference_mode()
def generate_diverse_captions(image, processor, model, device="cuda", num_caps=5, max_new_tokens=40):
    """
    Generate N diverse captions with stochastic decoding only (no text prompt).
    Diversity comes from varying sampling hyperparams each time.
    """
    caps, seen = [], set()
    settings = [
        dict(temperature=1.0, top_p=0.95, top_k=60),
        dict(temperature=1.2, top_p=0.9,  top_k=80),
        dict(temperature=1.4, top_p=0.85, top_k=100),
        dict(temperature=1.6, top_p=0.9,  top_k=120),
        dict(temperature=1.8, top_p=0.8,  top_k=150),
    ]

    for i in range(num_caps):
        cfg = settings[i % len(settings)]
        inputs = processor(images=image, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            do_sample=True,
            num_beams=1,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            **cfg
        )
        text = processor.decode(out[0], skip_special_tokens=True)
        norm = normalize_caption(text)
        if norm and norm not in seen:
            caps.append(norm); seen.add(norm)
    return caps

# ---------------- Main ----------------
results = load_json_or_empty(OUT_JSON)
processed = set(results.keys())
print(f"[Resume] loaded {len(results)} filenames from {OUT_JSON}")

zip_paths = sorted(glob.glob(ZIP_GLOB))
if not zip_paths: raise FileNotFoundError(f"No ZIPs match {ZIP_GLOB}")

# Count valid zips and total images
total_imgs = 0; valid_zips = []
for z in zip_paths:
    try:
        with zipfile.ZipFile(z, "r") as zf:
            count = sum(1 for m in zf.namelist() if is_image_name(m))
            total_imgs += count; valid_zips.append(z)
    except Exception as e:
        print(f"[WARN] Skipping invalid zip {z}: {e}")
if not valid_zips: raise RuntimeError("No valid ZIPs to process.")
print(f"Found {len(valid_zips)} valid zip parts, total {total_imgs} images")

print("Loading BLIP v1 model...")
proc, model = make_model(MODEL_ID, DEVICE)
print("Model ready.")

done=skipped=0; since_save=0; processed_global=len(processed)
start_time = time.time()

try:
    for zp in valid_zips:
        print(f"\nProcessing {zp}")
        with zipfile.ZipFile(zp,"r") as zf:
            members=[m for m in zf.namelist() if is_image_name(m)]
            for member in tqdm(members):
                fname=os.path.basename(member)
                if fname in processed: skipped+=1; continue
                try:
                    with zf.open(member) as fh:
                        img=Image.open(io.BytesIO(fh.read())).convert("RGB")
                except Exception as e:
                    print(f"[WARN] read fail {fname}: {e}")
                    results[fname]=[]; processed.add(fname); since_save+=1; continue
                try:
                    caps=generate_diverse_captions(img,proc,model,DEVICE,NUM_CAPTIONS,MAX_NEW_TOKENS)
                except Exception as e:
                    print(f"[WARN] caption fail {fname}: {e}")
                    caps=[]
                results[fname]=caps; processed.add(fname)
                done+=1; since_save+=1; processed_global+=1

                elapsed=time.time()-start_time
                ips=(processed_global/elapsed) if elapsed>0 else 0
                spit=(elapsed/processed_global) if processed_global>0 else 0
                pct=(processed_global/total_imgs)*100 if total_imgs>0 else 0
                remaining=max(total_imgs-processed_global,0)
                eta=(remaining/ips) if ips>0 else float('inf')

                print(f"Progress: {processed_global}/{total_imgs} ({pct:.2f}%)  |  "
                      f"{ips:.2f} it/s  |  {spit:.3f} s/it  |  ETA: {eta/60:.1f} min", flush=True)

                if since_save>=SAVE_EVERY:
                    save_json_atomic(results,OUT_JSON); since_save=0
    if since_save>0: save_json_atomic(results,OUT_JSON)
finally:
    if since_save>0: save_json_atomic(results,OUT_JSON)

print(f"\n=== Summary ===\nProcessed new: {done}\nSkipped: {skipped}\nTotal JSON entries: {len(results)}\nOutput: {OUT_JSON}")
# ===============================================================================================================
