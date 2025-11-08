# Gemma PDF Translator (Streamlit)

این پروژه یک اپلیکیشن Streamlit برای ترجمه صفحات PDF با استفاده از مدل‌های Gemma است.

## ویژگی‌ها
- استخراج متن صفحه به صفحه از PDF
- ساخت embedding با `google/embeddinggemma-300m`
- رتریوال با FAISS
- ترجمه متن صفحه با `google/gemma-3-27b-it`
- تولید فایل Word از ترجمه

## اجرا روی Hugging Face Spaces
- Runtime: Python 3.10+
- GPU: توصیه می‌شود (مثلاً A10G یا بهتر)

## اجرا محلی
1. ساخت virtualenv
2. نصب `requirements.txt`
3. اجرای `streamlit run app.py`

## متغیرهای محیطی
- مدل‌ها و توکن Hugging Face در `.env` یا `.env.example` قرار می‌گیرد
