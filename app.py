import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Scam Detector", page_icon="🕵️‍♂️", layout="wide")

import nemo.collections.asr as nemo_asr
import torch
import librosa
import soundfile as sf
import os
import tempfile
import json
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from st_audiorec import st_audiorec
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import requests

load_dotenv()

CHROMA_PATH = "./chroma_db_thairath"
BACKEND_API_URL = "http://localhost:8000/notify"
DEFAULT_TARGET_GROUP_ID = "YOUR_TARGET_GROUP_ID_HERE"

def init_session_state():
    # อ่านค่าจาก URL (st.query_params แบบใหม่)
    query_params = st.query_params
    
    url_user_id = query_params.get("line_user_id", None)
    url_group_id = query_params.get("target_group_id", None)
    
    # 1. จัดการ User ID
    if url_user_id:
        st.session_state['line_user_id'] = url_user_id
    
    # 2. จัดการ Group ID (สำคัญมาก: ถ้ามีใน URL ให้ทับใน Session เลย)
    if url_group_id:
        st.session_state['target_group_id'] = url_group_id
    
    # ถ้าใน Session ยังไม่มี ให้กำหนดเป็นค่าว่าง
    if 'target_group_id' not in st.session_state:
        st.session_state['target_group_id'] = ""
        
    if 'line_user_id' not in st.session_state:
        st.session_state['line_user_id'] = ""

# เรียกใช้งานทันทีที่เริ่มโหลดหน้า
init_session_state()

# ดึงค่าจาก Session มาใช้ (ตัวแปรหลักที่จะเอาไปใช้ทั้งแอป)
user_id = st.session_state['line_user_id']
target_group_id = st.session_state['target_group_id']
display_name = "User" # Default

# --- 3. Sidebar Setup ---
st.sidebar.header("👤 สถานะการเชื่อมต่อ")

if user_id:
    st.sidebar.success(f"✅ User ID: ...{user_id[-4:]}")
else:
    st.sidebar.warning("⚠️ Guest Mode (ไม่ได้ Login)")

if target_group_id:
    st.sidebar.success(f"✅ Group ID: ...{target_group_id[-4:]}")
    st.sidebar.caption(f"Full ID: {target_group_id}") # แสดงเต็มเพื่อ debug
else:
    st.sidebar.error("❌ ไม่พบ Group ID")
    st.sidebar.info("กรุณาเข้าผ่านลิงก์จาก LINE Bot ในกลุ่ม")
    
    # ให้กรอกเองได้กรณีฉุกเฉิน
    target_group_id = st.sidebar.text_input("ใส่ Group ID เอง (ถ้าจำเป็น):", value=st.session_state['target_group_id'])
    # อัปเดตกลับเข้า session ถ้ามีการพิมพ์แก้
    if target_group_id:
        st.session_state['target_group_id'] = target_group_id

st.sidebar.divider()
use_rag_feature = st.sidebar.toggle("📚 Use RAG", value=True)


@st.cache_resource
def load_rag_system():
    """Load Embedding model and Vector DB"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        if os.path.exists(CHROMA_PATH):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings, collection_name="thairath_news")
            return db
        return None
    except Exception as e:
        st.error(f"Failed to load Knowledge Base: {e}")
        return None

vector_db = load_rag_system()
# --- การตั้งค่า Gemini API ---
@st.cache_resource
def setup_gemini_client():
    """
    ตรวจสอบ API key
    """
    api_key = None
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
    except FileNotFoundError:
        st.sidebar.warning("ไม่พบไฟล์ .env")
    except KeyError:
        st.sidebar.warning("ไม่พบ GOOGLE_API_KEY ใน .env")

    if not api_key:
        api_key = st.sidebar.text_input("กรุณาป้อน Gemini API Key ของคุณ:", type="password", key="api_key_input")

    if not api_key:
        st.error("จำเป็นต้องใช้ Gemini API Key เพื่อวิเคราะห์ข้อความ")
        st.stop()
    
    try:
        genai.configure(api_key=api_key)
        return genai
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการตั้งค่า Gemini: {e}")
        st.stop()

@st.cache_resource
def load_asr_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="scb10x/typhoon-asr-realtime",
        map_location=device
    )
    return model, device

# --- Setup ---
genai_client = setup_gemini_client()

asr_model_status = st.sidebar.info("กำลังโหลดโมเดล Typhoon ASR...")
asr_model, device = load_asr_model()
asr_model_status.success(f"โมเดล ASR พร้อมใช้งาน! ({device.upper()})")

def prepare_audio(input_path, output_path, target_sr=16000):
    """
    เตรียมไฟล์เสียงสำหรับ Typhoon ASR (ดัดแปลงจากโค้ดของคุณ)
    """
    try:
        # Load (รองรับ MP3/WAV)
        y, sr = librosa.load(input_path, sr=None)
        
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Normalize
        y = y / max(abs(y))
        
        # Save เป็น WAV
        sf.write(output_path, y, target_sr)
        return output_path
    
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผลไฟล์เสียง: {e}")
        return None

def run_transcription(asr_model, uploaded_file):
    """
    รับไฟล์ที่อัปโหลด (MP3/WAV) บันทึกชั่วคราว ประมวลผล และถอดเสียง
    """
    file_suffix = os.path.splitext(uploaded_file.name)[-1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_in:
        tmp_in.write(uploaded_file.getvalue())
        input_audio_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        output_wav_path = tmp_out.name

    try:
        with st.spinner("กำลังประมวลผลไฟล์เสียง..."):
            processed_wav = prepare_audio(input_audio_path, output_wav_path)
        
        if processed_wav:
            with st.spinner("กำลังถอดเสียงด้วยโมเดล Typhoon..."):
                transcriptions = asr_model.transcribe(audio=[processed_wav])
                if transcriptions:
                    return transcriptions[0].text
                else:
                    return "[ไม่สามารถถอดเสียงได้]"
        else:
            return "[เกิดข้อผิดพลาดในการเตรียมไฟล์เสียง]"

    finally:
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)
        if os.path.exists(output_wav_path):
            os.remove(output_wav_path)

# --- ฟังก์ชันใหม่สำหรับไมโครโฟน ---
def run_transcription_from_mic(asr_model, audio_bytes):
    """
    รับ audio bytes (WAV) จาก st_audiorec, ประมวลผล, และถอดเสียง
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_in:
        tmp_in.write(audio_bytes)
        input_wav_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        output_wav_path = tmp_out.name

    try:
        with st.spinner("กำลังประมวลผลประโยคสนทนา..."):
            processed_wav = prepare_audio(input_wav_path, output_wav_path)
        
        if processed_wav:
            with st.spinner("กำลังถอดเสียงด้วยโมเดล Typhoon..."):
                transcriptions = asr_model.transcribe(audio=[processed_wav])
                if transcriptions:
                    return transcriptions[0].text
                else:
                    return "[ไม่สามารถถอดเสียงได้]"
        else:
            return "[เกิดข้อผิดพลาดในการเตรียมไฟล์เสียง]"

    finally:
        if os.path.exists(input_wav_path):
            os.remove(input_wav_path)
        if os.path.exists(output_wav_path):
            os.remove(output_wav_path)
            

@st.cache_data(show_spinner=False)
def analyze_scam_with_llm(_genai_client, text_to_analyze, use_rag=True):
    """
    เรียก Gemini API เพื่อวิเคราะห์ข้อความว่าเป็น Scammer หรือไม่
    """

    retrieved_context = ""
    references = []
    
    if use_rag and vector_db:
        docs = vector_db.similarity_search(text_to_analyze, k=3)
        if docs:
            retrieved_context = "\n".join([f"- {d.page_content}" for d in docs])
            references = [d.metadata.get('title', 'ไม่ระบุแหล่งที่มา') for d in docs]
 
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

    # 3. ตั้งค่า GenerationConfig
    generation_config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=json_schema
    )

    # 4. สร้างโมเดล
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
        generation_config=generation_config
    )

    # 5. สร้าง Prompt
    prompt = f"โปรดวิเคราะห์บทสนทนานี้: \"{text_to_analyze}\""

    # 6. เรียก API
    try:
        response = model.generate_content(prompt)
        # Parse JSON
        result = json.loads(response.text)
        result['references'] = references
        return result
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเรียก LLM: {e}")
        if "response" in locals():
            st.error(f"Response ที่ได้รับ: {response.parts}")
        return None

def send_alert_to_line(message, result, user_name, target_id):
    """
    ส่งข้อมูลไปยัง FastAPI service เพื่อแจ้งเตือน Line
    รับค่า user_name และ target_id เข้ามาโดยตรงเพื่อความชัวร์
    """
    # Fallback: ถ้า target_id ที่ส่งมาเป็นค่าว่าง ให้ลองดึงจาก Session อีกรอบ
    if not target_id:
        target_id = st.session_state.get('target_group_id')
    
    # ดึง User ID (Reporter) จาก Session เพื่อใช้เป็น Fallback
    reporter_id = st.session_state.get('line_user_id')

    payload = {
        "message": message,
        "fraud_details": result,
        "user_name": user_name if user_name else "Group Member",
        "target_id": target_id,
        "reporter_id": reporter_id 
    }

    try:
        # 🟢 แก้ไข 1: ใช้ชื่อตัวแปร BACKEND_API_URL ให้ตรงกับที่ประกาศไว้ข้างบน
        response = requests.post(BACKEND_API_URL, json=payload)
            
        if response.status_code == 200:
            st.toast("✅ ส่งแจ้งเตือนไปยัง Group Line เรียบร้อยแล้ว!", icon="📨")
        else:
            st.toast("⚠️ ไม่สามารถส่งแจ้งเตือน Line ได้", icon="❌")
            # แสดง Error จาก Server เพื่อให้รู้สาเหตุ
            st.error(f"Server Error ({response.status_code}): {response.text}")
            
    except Exception as e:
        st.error(f"Connection Error: {e}")

def display_analysis_results(result, analyzed_text=None):
    """
    แสดงผลการวิเคราะห์ในรูปแบบที่สวยงาม
    """
    st.subheader("ผลการวิเคราะห์:")
    
    verdict = result.get("verdict", "ไม่ทราบผล")
    confidence = result.get("confidence", "ไม่ทราบ")

    current_group_id = st.session_state.get('target_group_id')
    
    if "สูง" in verdict or "High" in verdict:
        st.error(f"🚨 **{verdict}** (ความมั่นใจ: {confidence})")
        
        # *** AUTO ALERT ***
        if current_group_id and DEFAULT_TARGET_GROUP_ID != "Cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx":
             with st.spinner("🚀 พบความเสี่ยงสูง! กำลังแจ้งเตือนเข้ากลุ่ม LINE..."):
                send_alert_to_line(analyzed_text, result, display_name, current_group_id)
        elif current_group_id:
             st.warning("⚠️ พบความเสี่ยงสูง แต่ยังไม่ได้ตั้งค่า Group ID ที่ถูกต้อง จึงไม่ได้ส่งแจ้งเตือน")
                
    elif "ปานกลาง" in verdict:
        st.warning(f"⚠️ **{verdict}** (ความมั่นใจ: {confidence})")
    else:
        st.success(f"✅ **{verdict}** (ความมั่นใจ: {confidence})")

    # แสดงรายละเอียด
    st.markdown("**เหตุผล:**")
    for r in result.get("reasoning", []):
        st.markdown(f"- {r}")
        
    if result.get("warning_signs"):
        st.markdown("**สัญญาณเตือน:**")
        for s in result.get("warning_signs", []):
            st.markdown(f"- {s}")

# --- หน้าจอหลัก Streamlit ---

st.title("🕵️‍♂️ แอปตรวจจับบทสนทนามิจฉาชีพ (Scam Detector)")
st.caption("ขับเคลื่อนด้วย Typhoon ASR + Gemini LLM")

# --- UI Tabs ---
tab1, tab2, tab3 = st.tabs(["📁 อัปโหลดไฟล์เสียง", "✏️ ป้อนข้อความ", "🎙️ เสียงสดจากไมโครโฟน"])

with tab1:
    st.header("วิเคราะห์จากไฟล์เสียง")
    uploaded_file = st.file_uploader("เลือกไฟล์เสียง .mp3 หรือ .wav", type=["mp3", "wav"])
    
    analyze_audio_button = st.button("ถอดเสียงและวิเคราะห์", key="analyze_audio")

    if analyze_audio_button and uploaded_file:
        transcript = run_transcription(asr_model, uploaded_file)
        
        if transcript and not transcript.startswith("["):
            st.info(f"**ข้อความที่ถอดได้:**\n\n{transcript}")
            st.divider()
            

            with st.spinner(f"กำลังส่งข้อความให้ LLM วิเคราะห์... (RAG: {'ON' if use_rag_feature else 'OFF'})"):
                analysis_result = analyze_scam_with_llm(genai_client, transcript, use_rag=use_rag_feature)
            
            if analysis_result:
                display_analysis_results(analysis_result, transcript)
        else:
            st.error(transcript) 

with tab2:
    st.header("วิเคราะห์จากข้อความ")
    text_input = st.text_area("ป้อนบทสนทนาที่ต้องการวิเคราะห์:", height=200, key="text_input_area")
    
    analyze_text_button = st.button("วิเคราะห์ข้อความ", key="analyze_text")

    if analyze_text_button and text_input:
        with st.spinner(f"กำลังส่งข้อความให้ LLM วิเคราะห์... (RAG: {'ON' if use_rag_feature else 'OFF'})"):
            analysis_result = analyze_scam_with_llm(genai_client, text_input, use_rag=use_rag_feature)
        
        if analysis_result:
            display_analysis_results(analysis_result, text_input)

with tab3:
    st.header("วิเคราะห์จากเสียงไมโครโฟน")
    st.write("กดปุ่มอัดเสียง เริ่มพูด และกดปุ่มหยุดเมื่อพูดจบครับ")
    
    # เรียก component อัดเสียง
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        
        transcript = run_transcription_from_mic(asr_model, wav_audio_data)
        
        if transcript and not transcript.startswith("["):
            st.info(f"**ข้อความที่ถอดได้:**\n\n{transcript}")
            st.divider()
            
            with st.spinner(f"กำลังส่งข้อความให้ LLM วิเคราะห์... (RAG: {'ON' if use_rag_feature else 'OFF'})"):
                analysis_result = analyze_scam_with_llm(genai_client, transcript, use_rag=use_rag_feature)
            
            if analysis_result:
                display_analysis_results(analysis_result, transcript)
        else:
            st.error(transcript)