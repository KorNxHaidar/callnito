
import os
import glob
import json
import torch
import librosa
import soundfile as sf
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import nemo.collections.asr as nemo_asr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "./chroma_db_thairath"
SCAM_DATA_DIR = "./scam_data"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# --- 1. Load Models ---

def load_asr_model():
    print("Loading Typhoon ASR model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="scb10x/typhoon-asr-realtime",
            map_location=device
        )
        print(f"ASR Model loaded on {device}")
        return model
    except Exception as e:
        print(f"Failed to load ASR model: {e}")
        return None

def load_rag_system():
    print("Loading RAG System (ChromaDB + Embeddings)...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        if os.path.exists(CHROMA_PATH):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name="thairath_news")
            print("RAG System loaded.")
            return db
        print("ChromaDB path not found.")
        return None
    except Exception as e:
        print(f"Failed to load Knowledge Base: {e}")
        return None

# --- 2. Helper Functions ---

def prepare_audio(input_path, output_path, target_sr=16000):
    """
    Prepare audio for Typhoon ASR (Resample to 16kHz, Normalize, Save as WAV)
    """
    try:
        # Load
        y, sr = librosa.load(input_path, sr=None)
        
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Normalize
        y = y / max(abs(y))
        
        # Save as WAV
        sf.write(output_path, y, target_sr)
        return output_path
    
    except Exception as e:
        print(f"Error preparing audio {input_path}: {e}")
        return None

def transcribe_audio(asr_model, file_path):
    """
    Transcribe audio file using Typhoon ASR
    """
    temp_wav = "temp_eval.wav"
    try:
        processed_wav = prepare_audio(file_path, temp_wav)
        if processed_wav:
            transcriptions = asr_model.transcribe(audio=[processed_wav])
            if transcriptions:
                return transcriptions[0].text
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
    return ""

def analyze_scam_with_llm(vector_db, text_to_analyze, use_rag=True):
    """
    Analyze text with Gemini, optionally using RAG.
    """
    retrieved_context = ""
    references = []
    
    if use_rag and vector_db:
        try:
            docs = vector_db.similarity_search(text_to_analyze, k=3)
            if docs:
                retrieved_context = "\n".join([f"- {d.page_content}" for d in docs])
                references = [d.metadata.get('title', 'ไม่ระบุแหล่งที่มา') for d in docs]
        except Exception as e:
            print(f"RAG Error: {e}")
 
    ref_section = ""
    if use_rag:
        ref_section = f"""
        [ข้อมูลอ้างอิงจากฐานข้อมูลข่าว/การเตือนภัย]:
        {retrieved_context if retrieved_context else "ไม่พบข้อมูลที่ตรงกันในฐานข้อมูล"}
        
        คำแนะนำเพิ่มเติม: หากข้อมูลใน [ข้อมูลอ้างอิง] สอดคล้องกับข้อความที่วิเคราะห์ ให้ระบุในเหตุผลด้วย
        """

    system_prompt = f"""
    คุณคือผู้เชี่ยวชาญด้านความปลอดภัยทางไซเบอร์และการตรวจจับการหลอกลวง (Scam)
    หน้าที่ของคุณคือวิเคราะห์บทสนทนาที่ได้รับ และประเมินว่ามีโอกาสเป็นมิจฉาชีพหรือไม่
    
    {ref_section}
    
    โปรดตอบกลับเป็นภาษาไทย และใช้โครงสร้าง JSON ตามที่กำหนดเท่านั้น
    วิเคราะห์โดยพิจารณาปัจจัยต่างๆ เช่น:
    - ความเร่งด่วนที่ผิดปกติ (เช่น "ต้องทำทันที", "บัญชีจะถูกระงับ")
    - การขอข้อมูลส่วนตัว (เช่น รหัสผ่าน, เลขบัตรประชาชน, รหัส OTP)
    - การอ้างตัวเป็นเจ้าหน้าที่ (เช่น ตำรวจ, ธนาคาร, กรมสรรพากร)
    - การเสนอผลประโยชน์ที่น่าสงสัย (เช่น ถูกรางวัล, ได้เงินคืน)
    - การข่มขู่ (เช่น "จะถูกดำเนินคดี")
    """

    json_schema = {
        "type": "OBJECT",
        "properties": {
            "verdict": {
                "type": "STRING",
                "description": "ผลการประเมิน (เช่น 'มีโอกาสเป็นมิจฉาชีพสูง', 'ไม่น่าจะเป็นมิจฉาชีพ', 'ข้อมูลไม่เพียงพอ')"
            },
            "confidence": {
                "type": "STRING",
                "description": "ระดับความมั่นใจ (เช่น 'สูง', 'ปานกลาง', 'ต่ำ')"
            },
            "reasoning": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "เหตุผลสนับสนุนการประเมินเป็นข้อๆ"
            },
             "warning_signs": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
                "description": "สัญญาณเตือนที่ตรวจพบ (ถ้ามี)"
            }
        },
        "required": ["verdict", "confidence", "reasoning"]
    }

    generation_config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=json_schema
    )

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
        generation_config=generation_config
    )

    prompt = f"โปรดวิเคราะห์บทสนทนานี้: \"{text_to_analyze}\""

    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        return result
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def normalize_verdict(verdict_text):
    """
    Normalize LLM verdict to binary label: 1 (Scam), 0 (Safe), -1 (Unknown)
    """
    if not verdict_text:
        return -1
    v = verdict_text.lower()
    if "สูง" in v or "high" in v or "มิจฉาชีพ" in v and "ไม่น่าจะ" not in v:
        return 1 # Scam
    if "ปานกลาง" in v:
        return 1 # Treat medium risk as positive for scam detection (better safe than sorry) or maybe separate class
    if "ไม่น่าจะ" in v or "ต่ำ" in v or "safe" in v:
        return 0 # Safe
    return -1

def get_ground_truth(filename):
    """
    Determine ground truth from filename.
    scam*.wav -> 1 (Scam)
    general*.wav -> 0 (Safe)
    """
    name = os.path.basename(filename).lower()
    if "scam" in name:
        return 1
    elif "general" in name:
        return 0
    return -1

# --- 3. Main Evaluation Loop ---

def main():
    # Load resources
    asr_model = load_asr_model()
    vector_db = load_rag_system()
    
    if not asr_model:
        print("Critical Error: Could not load ASR model.")
        return

    # Get files
    audio_files = glob.glob(os.path.join(SCAM_DATA_DIR, "*.wav")) + glob.glob(os.path.join(SCAM_DATA_DIR, "*.mp3"))
    if not audio_files:
        print(f"No audio files found in {SCAM_DATA_DIR}")
        return

    print(f"Found {len(audio_files)} files.")
    
    # Main Evaluation Loop
    results = []

    DELAY_SECONDS = 2

    for file_path in tqdm(audio_files, desc="Processing Files"):
        filename = os.path.basename(file_path)
        ground_truth = get_ground_truth(filename)
        
        # 1. Transcribe (Once per file)
        print(f"\nProcessing {filename}...")
        transcript = transcribe_audio(asr_model, file_path)
        if not transcript:
            print(f"Skipping {filename}: Transcription failed.")
            continue
        
        # 2. Analyze 5 times
        for i in range(1, 6):
            print(f"  Run {i}/5...")
            
            # Use a dummy retry wrapper or just raw call, but we need to sleep after each call
            
            # --- Condition 1: No RAG ---
            try:
                res_no_rag = analyze_scam_with_llm(vector_db, transcript, use_rag=False)
                # Sleep to respect rate limit
                time.sleep(DELAY_SECONDS)
            except Exception as e:
                print(f"Error in No RAG run {i}: {e}")
                res_no_rag = None
                
            verdict_no_rag = res_no_rag.get('verdict') if res_no_rag else "Error"
            pred_no_rag = normalize_verdict(verdict_no_rag)
            
            # --- Condition 2: With RAG ---
            try:
                res_rag = analyze_scam_with_llm(vector_db, transcript, use_rag=True)
                # Sleep to respect rate limit
                time.sleep(DELAY_SECONDS)
            except Exception as e:
                print(f"Error in RAG run {i}: {e}")
                res_rag = None

            verdict_rag = res_rag.get('verdict') if res_rag else "Error"
            pred_rag = normalize_verdict(verdict_rag)
            
            results.append({
                "filename": filename,
                "run_id": i,
                "transcript": transcript,
                "ground_truth": ground_truth,
                "verdict_no_rag": verdict_no_rag,
                "pred_no_rag": pred_no_rag,
                "verdict_rag": verdict_rag,
                "pred_rag": pred_rag,
                "correct_no_rag": pred_no_rag == ground_truth,
                "correct_rag": pred_rag == ground_truth
            })

    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv("eval_results.csv", index=False)
    
    # Calculate Metrics
    print("\n--- Evaluation Results ---")
    
    if len(df_res) > 0:
        # Group by filename to see consistency? Or just overall accuracy?
        # Overall Accuracy
        acc_no_rag = df_res['correct_no_rag'].mean()
        acc_rag = df_res['correct_rag'].mean()
        
        print(f"Total Prediction Rows: {len(df_res)}")
        print(f"Overall Accuracy (LLM Only): {acc_no_rag:.2%}")
        print(f"Overall Accuracy (LLM + RAG): {acc_rag:.2%}")
        
        print("\n--- Detailed Results (First 10 rows) ---")
        print(df_res[['filename', 'run_id', 'ground_truth', 'pred_no_rag', 'pred_rag']].head(10))
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
