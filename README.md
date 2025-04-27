# 🚀 RAG LINE Bot App

ระบบ AI Bot ตอบคำถามผ่าน LINE โดยใช้ Retrieval-Augmented Generation (RAG) เชื่อมกับ FAISS Vector Database และ OpenRouter LLM

---

## 📚 คุณสมบัติ

- ใช้ FastAPI สร้าง Webhook Server
- ใช้ FAISS เป็น Vector Database ฝังข้อความ
- ใช้ OpenRouter Free LLM (gpt-3.5-turbo) สำหรับสร้างคำตอบ
- เชื่อมต่อ LINE Bot ผ่าน Messaging API
- Deploy ได้ง่ายบน Railway ฟรี

---

## 📂 โครงสร้างโปรเจกต์

```plaintext
rag-linebot-app/
├── app/
│   ├── main.py
│   ├── rag_chain.py
│   ├── line_webhook.py
│   └── vector_store.py
├── data/docs.txt
├── requirements.txt
├── Procfile
├── .env.example
├── README.md
