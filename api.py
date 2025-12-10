import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from linebot.v3 import WebhookHandler
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage
)
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    JoinEvent,
    GroupSource,
    UserSource
)

load_dotenv()

app = FastAPI()

# --- CONFIGURATION ---
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
DEFAULT_USER_ID = os.getenv("LINE_USER_ID") 

STREAMLIT_WEB_URL = os.getenv("WEB_URL")

configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET) if CHANNEL_SECRET else None

# --- Data Models ---
class AlertRequest(BaseModel):
    message: str
    fraud_details: dict | None = None
    user_name: str | None = "Group Member"
    target_id: str | None = None
    reporter_id: str | None = None 

# --- Endpoints ---

@app.post("/notify")
async def send_notification(request: AlertRequest, background_tasks: BackgroundTasks):
    if not CHANNEL_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Line Bot credentials not configured.")
    
    try:
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            
            # 1. ระบุผู้รับ (Recipient Logic)
            recipient_id = request.target_id
            if not recipient_id or recipient_id.strip() == "":
                recipient_id = request.reporter_id
            if not recipient_id or recipient_id.strip() == "":
                recipient_id = DEFAULT_USER_ID
            
            if not recipient_id:
                raise HTTPException(status_code=400, detail="No recipient ID provided.")

            # 🔥 NEW LOGIC: ดึงชื่อจริงจาก LINE Profile 🔥
            # เราจะใช้ request.reporter_id (User ID คนแจ้ง) ไปถาม LINE ว่าเขาชื่ออะไร
            real_display_name = request.user_name # ค่า Default คือ "Group Member"
            
            if request.reporter_id:
                try:
                    # เรียก LINE API เพื่อขอ Profile
                    user_profile = line_bot_api.get_profile(request.reporter_id)
                    real_display_name = user_profile.display_name
                    print(f"✅ Fetched User Name: {real_display_name}")
                except Exception as e:
                    print(f"⚠️ Could not fetch user profile: {e}")
                    # ถ้าดึงไม่ได้ ให้ใช้ค่าเดิมไปก่อน

            # --- สร้างข้อความ ---
            details = request.fraud_details or {}
            verdict = details.get('verdict', 'ไม่ระบุ')
            confidence = details.get('confidence', 'ไม่ระบุ')
            reasoning_list = details.get('reasoning', [])
            warning_signs = details.get('warning_signs', [])
            
            text = f"🚨 ALERT: เตือนภัยกลุ่ม! 🚨\n"
            # ใช้ชื่อจริงที่ดึงมาได้
            text += f"สมาชิกที่พบความเสี่ยง: คุณ {real_display_name}\n" 
            text += "━━━━━━━━━━━━━━━━━━\n\n"
            
            text += f"📊 ผลการประเมิน:\n{verdict}\n"
            text += f"🔥 ระดับความมั่นใจ: {confidence}\n\n"

            if reasoning_list:
                text += "🧐 เหตุผลวิเคราะห์:\n"
                for item in reasoning_list:
                    text += f"• {item}\n"
                text += "\n"

            if warning_signs:
                text += "⚠️ สัญญาณเตือนที่พบ:\n"
                for sign in warning_signs:
                    text += f"- {sign}\n"
                text += "\n"

            original_msg = request.message
            if len(original_msg) > 100:
                original_msg = original_msg[:100] + "..."
            
            text += f"📝 ข้อความต้นเรื่อง:\n\"{original_msg}\""

            line_bot_api.push_message(
                PushMessageRequest(
                    to=recipient_id,
                    messages=[TextMessage(text=text)]
                )
            )
            
        return {"status": "success", "message": f"Sent to {recipient_id}"}
    except Exception as e:
        print(f"❌ Error sending Line message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/callback")
async def callback(request: Request):
    if not handler:
        raise HTTPException(status_code=500, detail="Secret not configured.")
    
    signature = request.headers.get('X-Line-Signature')
    body = await request.body()
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return 'OK'

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    msg_text = event.message.text.strip()
    
    if msg_text in ["เริ่มใช้งาน", "start", "scam"]:
        user_id = event.source.user_id
        group_id = None
        
        source_type = "Private Chat"
        if isinstance(event.source, GroupSource):
            group_id = event.source.group_id
            source_type = f"Group ({group_id})"
        
        print(f"📢 Request from: {source_type}")

        # สร้าง Link
        target_url = f"{STREAMLIT_WEB_URL}?line_user_id={user_id}"
        
        if group_id:
            target_url += f"&target_group_id={group_id}"
        
        reply_msg = f"🔎 กดลิ้งค์นี้เพื่อเริ่มใช้งาน ({source_type}):\n👉 {target_url}"
        
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_msg)]
                )
            )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)