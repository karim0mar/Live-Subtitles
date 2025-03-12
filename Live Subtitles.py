import sounddevice as sd
import numpy as np
import queue
import webrtcvad
import torch
import whisper
import keyboard
from scipy.signal import butter, lfilter
import time
import tkinter as tk
from tkinter import ttk, Label, Button, Frame, StringVar, OptionMenu, messagebox
import threading
import screeninfo
import os
import json

#pyinstaller --onefile --windowed --icon=app_icon.ico --add-data "C:\Users\karim\OneDrive\Desktop\Python\.venv\Lib\site-packages\whisper\assets;whisper\assets" --hidden-import=requests --hidden-import=whisper --hidden-import=torch --hidden-import=numpy main.py
# الثوابت العامة

import logging
import sys

# إنشاء مجلد للسجلات إذا لم يكن موجودًا
log_dir = os.path.join(os.path.expanduser("~"), ".live_subtitles", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")

# إعداد نظام التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # سيعمل فقط في الوضع غير النافذة
    ]
)
logger = logging.getLogger("LiveSubtitles")

APP_NAME = "Live Subtitles"
APP_VERSION = "2.0"
DEFAULT_LANGUAGE = "de"  # اللغة الافتراضية
CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".live_subtitles")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
MODELS_DIR = os.path.join(CONFIG_DIR, "models")

# الإعدادات الأساسية للصوت
SAMPLE_RATE = 16000
FRAME_DURATION = 30  # مللي ثانية
FRAME_SIZE = int(SAMPLE_RATE * (FRAME_DURATION / 1000))
BLOCK_SIZE = 480
CHANNELS = 1
SILENCE_THRESHOLD = 0.3
MAX_SEGMENT_DURATION = 3.0
MAX_QUEUE_SIZE = 10

settings_window_open = False
settings_window = None

# الألوان والمظهر العام (تصميم 2025)
THEME = {
    "dark": {
        "bg": "#121212",
        "card_bg": "#1E1E1E",
        "glass_bg": "rgba(30, 30, 30, 0.7)",
        "text": "#FFFFFF",
        "text_secondary": "#AAAAAA",
        "accent": "#7C4DFF",  # لون بنفسجي عصري
        "accent_gradient": ["#7C4DFF", "#2196F3"],  # تدرج بنفسجي إلى أزرق
        "success": "#00E676",  # أخضر نيون
        "warning": "#FFAB00",  # أصفر برتقالي
        "error": "#FF5252",    # أحمر نيون
        "surface": "#252525",  # لون للعناصر المرتفعة
        "border": "#333333"    # لون الإطارات
    },
    "light": {
        "bg": "#F5F7FA",
        "card_bg": "#FFFFFF",
        "glass_bg": "rgba(255, 255, 255, 0.7)",
        "text": "#212121",
        "text_secondary": "#5F6368",
        "accent": "#7C4DFF",  # نفس اللون في الوضع الداكن للتناسق
        "accent_gradient": ["#7C4DFF", "#2196F3"],
        "success": "#00C853",
        "warning": "#FF9100",
        "error": "#F44336",
        "surface": "#E9ECEF",
        "border": "#DADCE0"
    }
}

# الخيارات المتاحة للمودل
AVAILABLE_MODELS = {
    "tiny": {"size": "~75MB", "description": "متوسط - سريع مع دقة مقبولة", "arabic_name": "متوسط"},
    "base": {"size": "~150MB", "description": "دقيق - متوازن بين السرعة والدقة", "arabic_name": "دقيق"},
    "small": {"size": "~500MB", "description": "دقيق جدا - بطيء مع دقة عالية", "arabic_name": "دقيق جدا"}
}

# اللغات المدعومة
SUPPORTED_LANGUAGES = {
    "ar": "Arabisch",
    "de": "Deutsch",
    "en": "Englisch",
    "es": "Spanisch",
    "fr": "Französisch",
    "ru": "Russisch",
    "zh": "Chinesisch",
    "ja": "Japanisch",
    "it": "Italienisch"
}


# تهيئة المتغيرات العامة
audio_queue = queue.Queue()
processing_queue = queue.Queue()
subtitle_queue = queue.Queue()
audio_buffer = []
silence_counter = 0
is_speaking = False
last_speech_time = time.time()
last_segment_time = time.time()
previous_text = ""
context_buffer = []
config = {}
MODEL = None
current_theme = "dark"  # الوضع الافتراضي
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
vad = None  # سيتم تعريفه لاحقاً في start_transcription

# إنشاء مجلد الإعدادات إذا لم يكن موجوداً
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# تحميل الإعدادات
def load_config():
    global config
    default_config = {
        "model_size": "tiny",
        "language": DEFAULT_LANGUAGE,
        "theme": "dark",
        "opacity": 0.85,
        "font_size": 18,
        "last_device_id": None,
        "models_downloaded": []
    }
    
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = default_config
    except Exception as e:
        print(f"⚠ خطأ في تحميل الإعدادات: {e}")
        config = default_config
    
    # تأكد من وجود جميع المفاتيح الأساسية
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    return config

# حفظ الإعدادات
def save_config():
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("✅ تم حفظ الإعدادات")
    except Exception as e:
        print(f"⚠ خطأ في حفظ الإعدادات: {e}")

# التحقق من توفر المودل محلياً
def check_model_downloaded(model_name):
    if not os.path.exists(CONFIG_FILE):
        return False
    
    if "models_downloaded" not in config:
        return False
    
    return model_name in config["models_downloaded"]

# تنزيل المودل باستخدام آلية التنزيل الرسمية من مكتبة Whisper مع عرض شريط التقدم
def download_model(model_name, progress_var, status_label):
    global MODELS_DIR
    
    try:
        print(f"بدء تحميل النموذج {model_name}...")
        
        # التأكد من وجود المجلد ووجود صلاحيات الكتابة
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR, exist_ok=True)
        
        # فحص صلاحيات الكتابة في المجلد
        if not os.access(MODELS_DIR, os.W_OK):
            print(f"ليس لديك صلاحية الكتابة في المجلد: {MODELS_DIR}")
            
            # محاولة استخدام مجلد بديل
            temp_dir = None
            
            # محاولة استخدام المجلد الحالي للتطبيق
            if getattr(sys, 'frozen', False):
                app_dir = os.path.dirname(sys.executable)
                temp_dir = os.path.join(app_dir, "models")
            else:
                temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            
            try:
                os.makedirs(temp_dir, exist_ok=True)
                if os.access(temp_dir, os.W_OK):
                    MODELS_DIR = temp_dir
                    print(f"تم تغيير مجلد النماذج إلى: {MODELS_DIR}")
                else:
                    print(f"أيضاً ليس لديك صلاحية الكتابة في المجلد البديل: {temp_dir}")
                    messagebox.showerror("خطأ في الصلاحيات", 
                                       f"ليس لديك صلاحية الكتابة في المجلدات")
                    return False
            except Exception as e:
                print(f"خطأ في إنشاء المجلد البديل: {e}")
                messagebox.showerror("خطأ في الصلاحيات", 
                                   f"ليس لديك صلاحية الكتابة في المجلد")
                return False
        
        # تحديث حالة UI
        status_label.config(text=f"جاري التحضير لتحميل نموذج {model_name}...")
        
        # تقدير أحجام الموديل (بالميجابايت)
        estimated_sizes = {
            "tiny": 75,      # ~75MB
            "base": 150,     # ~150MB
            "small": 500,    # ~500MB
            "medium": 1500,  # ~1.5GB
            "large": 3000    # ~3GB
        }
        
        # إعداد متغيرات المراقبة
        progress_thread_running = True
        download_progress = 0
        last_update_time = time.time()
        
        # وظيفة لتحديث شريط التقدم
        def update_progress():
            nonlocal download_progress, last_update_time
            
            while progress_thread_running and download_progress < 100:
                # رفع التقدم ببطء للإشارة إلى التقدم
                current_time = time.time()
                if current_time - last_update_time >= 0.5:  # تحديث كل نصف ثانية
                    if download_progress < 95:  # لا تصل إلى 100% حتى يكتمل التنزيل
                        download_progress += 0.5
                        progress_var.set(download_progress)
                        
                        # تحديث النص
                        status_label.config(text=f"جاري تحميل نموذج {model_name}... {download_progress:.0f}%")
                    
                    last_update_time = current_time
                
                time.sleep(0.1)
        
        # بدء خيط تحديث التقدم
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        try:
            # تحميل النموذج باستخدام مكتبة whisper
            print(f"بدء تحميل النموذج {model_name} باستخدام whisper.load_model")
            model = whisper.load_model(model_name, download_root=MODELS_DIR)
            
            # إيقاف خيط التحديث
            progress_thread_running = False
            progress_thread.join(timeout=1.0)
            
            # تحديث شريط التقدم إلى 100%
            progress_var.set(100)
            status_label.config(text=f"تم تحميل نموذج {model_name} بنجاح!")
            
            # تحديث الإعدادات
            if "models_downloaded" not in config:
                config["models_downloaded"] = []
            
            if model_name not in config["models_downloaded"]:
                config["models_downloaded"].append(model_name)
            
            config["model_size"] = model_name
            save_config()
            
            print(f"تم تحميل النموذج {model_name} بنجاح!")
            return True
            
        except Exception as e:
            # إيقاف خيط التحديث
            progress_thread_running = False
            
            print(f"خطأ في تحميل نموذج {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # في حالة فشل التنزيل التلقائي، اطلب من المستخدم تنزيل النموذج يدويًا
            manual_download_msg = f"""
لم نتمكن من تنزيل النموذج تلقائيًا.

1. يرجى تنزيل ملف النموذج {model_name}.pt يدويًا.
2. قم بنسخ الملف إلى المجلد:
   {MODELS_DIR}
            """
            messagebox.showinfo("تنزيل يدوي مطلوب", manual_download_msg)
            
            # فتح مجلد النماذج
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(MODELS_DIR)
                elif os.name == 'posix':  # Linux/Mac
                    import subprocess
                    subprocess.Popen(['xdg-open', MODELS_DIR])
            except:
                pass
            
            progress_var.set(0)
            status_label.config(text=f"فشل تحميل النموذج: {str(e)}")
            return False
            
    except Exception as e:
        print(f"خطأ في إعداد التحميل: {e}")
        import traceback
        traceback.print_exc()
        status_label.config(text=f"خطأ في التحميل: {str(e)}")
        return False

# معالجة الصوت وإزالة الضوضاء
def process_audio(audio_data):
    # تطبيع مستوى الصوت
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        normalized = audio_data / max_val * 0.9
    else:
        normalized = audio_data
    
    # تطبيق فلتر لإزالة الضوضاء المنخفضة التردد
    b, a = butter_highpass(cutoff=100, fs=SAMPLE_RATE, order=2)
    filtered = lfilter(b, a, normalized)
    
    return filtered.astype(np.float32)

def butter_highpass(cutoff=100, fs=SAMPLE_RATE, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

# البحث عن جهاز الصوت
def get_system_audio_device():
    devices = sd.query_devices()
    print("\n🔍 الأجهزة الصوتية المتوفرة:")
    
    for i, device in enumerate(devices):
        print(f"   {i}: {device['name']} (Max ch: {device['max_input_channels']})")
    
    # البحث عن أجهزة النظام المعروفة
    for i, device in enumerate(devices):
        if ("stereo mix" in device['name'].lower() or 
            "stereomix" in device['name'].lower() or 
            "wasapi" in device['name'].lower() or 
            "wave" in device['name'].lower() or
            "loopback" in device['name'].lower() or
            "system" in device['name'].lower()):
            print(f"✅ جهاز الصوت المستخدم: {device['name']} (ID: {i})")
            return i
    
    # استخدام الجهاز المحفوظ سابقاً
    if config.get("last_device_id") is not None:
        device_id = config["last_device_id"]
        if 0 <= device_id < len(devices):
            print(f"✅ استخدام الجهاز المحفوظ: {devices[device_id]['name']} (ID: {device_id})")
            return device_id
    
    print("⚠ لم يتم العثور على جهاز صوت النظام، سيتم استخدام الإدخال الافتراضي.")
    return None

# تحسين اكتشاف الكلام
def is_speech(audio_chunk, vad):
    if len(audio_chunk) != FRAME_SIZE:
        return False
    
    # التحقق من مستوى الصوت
    audio_level = np.sqrt(np.mean(np.square(audio_chunk)))
    if audio_level < 0.01:  # تجاهل الصوت الخافت جداً
        return False
    
    # التحقق باستخدام VAD
    audio_int16 = (audio_chunk * 32768).astype(np.int16)
    try:
        return vad.is_speech(audio_int16.tobytes(), SAMPLE_RATE)
    except Exception:
        # إذا فشل VAD، استخدم فقط مستوى الصوت
        return audio_level > 0.02

# وظيفة التقاط الصوت المحسنة
def audio_callback(indata, frames, time_info, status):
    global silence_counter, last_speech_time, last_segment_time, audio_buffer, is_speaking
    
    if status and (status.input_overflow or "error" in str(status).lower()):
        print(f"⚠ خطأ في الصوت: {status}")
        return
    
    try:
        # تحويل البيانات إلى شكل مناسب للمعالجة
        audio_data = indata.flatten().astype(np.float32)
        
        # معالجة الصوت في إطارات
        for i in range(0, len(audio_data) - FRAME_SIZE + 1, FRAME_SIZE // 2):  # تداخل الإطارات لتحسين الدقة
            frame = audio_data[i:i + FRAME_SIZE]
            
            # تحديد إذا كان الإطار يحتوي على كلام
            speech_detected = is_speech(frame, vad)
            
            if speech_detected:
                if not is_speaking:
                    # بداية مقطع جديد
                    is_speaking = True
                    subtitle_queue.put(("status", "Listening..."))
                
                # إضافة الإطار إلى المخزن المؤقت
                audio_buffer.append(frame)
                silence_counter = 0
                last_speech_time = time.time()
                
                # إرسال أجزاء صغيرة بشكل مستمر للمعالجة المبكرة
                if len(audio_buffer) >= 15 and len(audio_buffer) % 5 == 0:  # كل ~150ms من الكلام المتواصل
                    if processing_queue.qsize() < 2:  # تجنب الازدحام
                        # أخذ نسخة من المخزن المؤقت الحالي للمعالجة المبكرة
                        current_segment = np.concatenate(audio_buffer)
                        processing_queue.put((current_segment, False))  # False = ليس نهاية الجملة
            else:
                # تحديث عداد الصمت
                silence_counter += 1
                
                # حساب مدة الصمت
                silence_duration = silence_counter * (FRAME_SIZE / SAMPLE_RATE / 2)  # تعديل بسبب التداخل
                
                # إرسال المقطع عند اكتشاف صمت أو تجاوز الحد الأقصى للمدة
                if is_speaking and (silence_duration >= SILENCE_THRESHOLD or 
                                   time.time() - last_segment_time >= MAX_SEGMENT_DURATION) and len(audio_buffer) > 10:
                    
                    if audio_queue.qsize() < MAX_QUEUE_SIZE:
                        full_segment = np.concatenate(audio_buffer)
                        is_speaking = False
                        audio_queue.put(full_segment)
                        processing_queue.put((full_segment, True))  # True = نهاية الجملة
                    else:
                        print("⚠ قائمة الانتظار ممتلئة، تجاهل المقطع الصوتي")
                    
                    # إعادة تعيين المخزن المؤقت
                    audio_buffer.clear()
                    silence_counter = 0
                    last_segment_time = time.time()
                    subtitle_queue.put(("status", "Processing..."))
    
    except Exception as e:
        print(f"⚠ خطأ في معالجة الصوت: {e}")

# مهمة نسخ الصوت محسنة
def transcribe_task():
    global previous_text, context_buffer
    
    # إعداد خيارات النسخ
    transcribe_options = {
        "fp16": (torch.cuda.is_available()),
        "language": config.get("language", "de"),
        "task": "transcribe",
        "beam_size": 3,
        "best_of": 1,
        "temperature": 0.0
    }
    
    while True:
        try:
            if not processing_queue.empty():
                audio_segment, is_final = processing_queue.get()
                
                # إذا كان المقطع صغيراً جداً، تجاهله
                if len(audio_segment) < SAMPLE_RATE * 0.3:  # أقل من 300 مللي ثانية
                    continue
                
                # معالجة الصوت
                audio_processed = process_audio(audio_segment)
                
                # تحديث حالة المعالجة
                subtitle_queue.put(("status", "Transcribing..."))
                
                # إعداد النص السابق كسياق إذا كان متوفراً
                prompt = " ".join(context_buffer[-3:]) if context_buffer else ""
                if prompt:
                    transcribe_options["prompt"] = prompt
                
                # النسخ باستخدام Whisper
                result = MODEL.transcribe(
                    audio_processed,
                    **transcribe_options
                )
                
                # الحصول على النص
                detected_text = result["text"].strip()
                
                # تحديث فقط إذا كان هناك نص
                if detected_text:
                    if is_final:
                        # تخزين النص في buffer السياق
                        context_buffer.append(detected_text)
                        # الاحتفاظ بآخر 5 جمل فقط
                        if len(context_buffer) > 5:
                            context_buffer = context_buffer[-5:]
                    
                    # إرسال النص للعرض
                    subtitle_queue.put(("text", detected_text, is_final))
                    
                    # طباعة في الكونسول للتصحيح
                    status = "FINAL" if is_final else "PARTIAL"
                    print(f"📝 [{status}]: {detected_text}")
            else:
                time.sleep(0.05)
                
        except Exception as e:
            print(f"⚠ خطأ في النسخ: {e}")
            time.sleep(0.1)

# ----- واجهة المستخدم المحسنة -----

# إنشاء نافذة اختيار المودل واللغة
def create_model_selector():
    root = tk.Tk()
    root.title(f"{APP_NAME} - إعداد أول مرة")
    
    # ضبط حجم النافذة وموقعها
    width, height = 550, 550
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # تعيين الألوان من السمة الحالية
    colors = THEME[current_theme]
    root.configure(bg=colors["bg"])
    
    # إنشاء Canvas مع شريط تمرير
    main_canvas = tk.Canvas(root, bg=colors["bg"], highlightthickness=0)
    main_canvas.pack(side="left", fill="both", expand=True)
    
    # إضافة شريط تمرير للcanvas
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    scrollbar.pack(side="right", fill="y")
    
    # ربط الـCanvas بشريط التمرير
    main_canvas.configure(yscrollcommand=scrollbar.set)
    
    # إنشاء الإطار الرئيسي داخل الـCanvas
    main_frame = Frame(main_canvas, bg=colors["bg"], padx=30, pady=30)
    
    # إضافة الإطار إلى Canvas
    canvas_frame = main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
    
    # تحديث أبعاد المنطقة القابلة للتمرير عند تغير حجم الإطار
    def configure_scroll_region(event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))
    
    main_frame.bind("<Configure>", configure_scroll_region)
    
    # ضبط عرض الإطار عند تغيير حجم النافذة
    def on_canvas_configure(event):
        main_canvas.itemconfig(canvas_frame, width=event.width)
    
    main_canvas.bind("<Configure>", on_canvas_configure)
    
    # تمكين التمرير بعجلة الماوس
    def on_mousewheel(event):
        main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
    root.bind_all("<MouseWheel>", on_mousewheel)  # Windows
    root.bind_all("<Button-4>", lambda e: main_canvas.yview_scroll(-1, "units"))  # Linux
    root.bind_all("<Button-5>", lambda e: main_canvas.yview_scroll(1, "units"))  # Linux
    
    # إطار الرأس
    header_frame = Frame(main_frame, bg=colors["bg"])
    header_frame.pack(fill="x", pady=(0, 25))
    
    # عنوان التطبيق ومعلومات المطور
    title_version_frame = Frame(header_frame, bg=colors["bg"])
    title_version_frame.pack(side="left")
    
    Label(
        title_version_frame, 
        text=APP_NAME, 
        font=("Segoe UI", 24, "bold"),
        fg=colors["text"], 
        bg=colors["bg"]
    ).pack(anchor="w")
    
    Label(
        title_version_frame,
        text=f"v{APP_VERSION}",
        font=("Segoe UI", 10),
        fg=colors["text_secondary"],
        bg=colors["bg"]
    ).pack(anchor="w")
    
    # معلومات ترحيبية
    welcome_frame = create_glass_frame(main_frame)
    welcome_frame.pack(fill="x", pady=(0, 25))
    
    Label(
        welcome_frame,
        text="مرحباً بك في تجربة ترجمة صوتية جديدة",
        font=("Segoe UI", 14, "bold"),
        fg=colors["text"],
        bg=colors["card_bg"]
    ).pack(pady=(0, 10))
    
    Label(
        welcome_frame,
        text="اختر نموذج الذكاء الاصطناعي واللغة المفضلة لديك لبدء تجربة الترجمة الصوتية المباشرة.",
        wraplength=width-100,
        justify="center",
        font=("Segoe UI", 10),
        fg=colors["text_secondary"],
        bg=colors["card_bg"]
    ).pack(pady=(0, 5))
    
    # إضافة العبارة الإسلامية بعد إطار التحميل
    islamic_frame = Frame(main_frame, bg=colors["bg"], pady=10)
    islamic_frame.pack(fill="x", pady=(5, 15))
    
    Label(
        islamic_frame,
        text="❤️ولا تنسونا من صالح دعائكم🌙",
        font=("Traditional Arabic", 14, "bold"),
        fg=colors["accent"],
        bg=colors["bg"],
        justify="center"
    ).pack(fill="x")
    
    # إطار اختيار المودل
    create_section_title(main_frame, "اختر نموذج الذكاء الاصطناعي")

    model_frame = create_glass_frame(main_frame, padx=20, pady=15)
    model_frame.pack(fill="x", pady=(0, 15))

    # متغير لتخزين اختيار المودل
    model_var = StringVar(root)
    model_var.set(config["model_size"])  # القيمة الافتراضية

    # إنشاء اختيارات المودل بتصميم بسيط
    for i, (model_name, model_info) in enumerate(AVAILABLE_MODELS.items()):
        model_row = Frame(model_frame, bg=colors["card_bg"], pady=7)
        model_row.pack(fill="x")
        
        radio_button = ttk.Radiobutton(
            model_row,
            text="",
            variable=model_var,
            value=model_name
        )
        radio_button.pack(side="left")
        
        # معلومات النموذج
        model_info_frame = Frame(model_row, bg=colors["card_bg"])
        model_info_frame.pack(side="left", padx=10, fill="x", expand=True)
        
        Label(
            model_info_frame,
            text=f"{model_info.get('arabic_name', model_name.capitalize())}",
            font=("Segoe UI", 11, "bold"),
            fg=colors["text"],
            bg=colors["card_bg"],
            anchor="w"
        ).pack(fill="x")
        
        Label(
            model_info_frame,
            text=f"{model_info['description']} • {model_info['size']}",
            font=("Segoe UI", 9),
            fg=colors["text_secondary"],
            bg=colors["card_bg"],
            anchor="w"
        ).pack(fill="x")

        # إطار اختيار اللغة
    create_section_title(main_frame, "اختر اللغة المستهدفة")

    lang_frame = create_glass_frame(main_frame, padx=20, pady=15)
    lang_frame.pack(fill="x", pady=(0, 20))

    # متغير لتخزين اختيار اللغة
    lang_var = StringVar(root)
    lang_var.set(config["language"])  # القيمة الافتراضية

    # ترتيب اللغات في صفوف
    lang_grid_frame = Frame(lang_frame, bg=colors["card_bg"])
    lang_grid_frame.pack(fill="x")

    row, col = 0, 0
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        lang_box = Frame(lang_grid_frame, bg=colors["card_bg"], padx=10, pady=5)
        lang_box.grid(row=row, column=col, sticky="w", padx=5, pady=5)
        
        ttk.Radiobutton(
            lang_box,
            text=f"{lang_name}",
            variable=lang_var,
            value=lang_code
        ).pack(side="left")
        
        Label(
            lang_box,
            text=f"({lang_code})",
            font=("Segoe UI", 8),
            fg=colors["text_secondary"],
            bg=colors["card_bg"]
        ).pack(side="left", padx=(3, 0))
        
        col += 1
        if col >= 3:  # عرض 3 أعمدة
            col = 0
            row += 1    
    # إطار التحميل والإعدادخض
    download_frame = create_glass_frame(main_frame, padx=20, pady=15)
    download_frame.pack(fill="x", pady=(0, 20))
    
    Label(
        download_frame,
        text="جاهز للتحميل والإعداد",
        font=("Segoe UI", 11, "bold"),
        fg=colors["text"],
        bg=colors["card_bg"]
    ).pack(pady=(0, 10))
    
    # شريط التقدم بتصميم بسيط
    progress_frame = Frame(download_frame, bg=colors["card_bg"], height=30)
    progress_frame.pack(fill="x", pady=5)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        progress_frame, 
        variable=progress_var,
        orient="horizontal",
        length=100, 
        mode="determinate"
    )
    progress_bar.pack(fill="x")
    
    status_label = Label(
        download_frame,
        text="اختر النموذج واللغة ثم انقر على زر البدء",
        font=("Segoe UI", 9),
        fg=colors["text_secondary"],
        bg=colors["card_bg"]
    )
    status_label.pack(pady=10)
    
    # إطار الأزرار
    buttons_frame = Frame(main_frame, bg=colors["bg"], pady=10)
    buttons_frame.pack(fill="x")
    
    # زر التأكيد
    def on_confirm():
        model_name = model_var.get()
        lang_code = lang_var.get()
        
        # تحديث الإعدادات
        config["model_size"] = model_name
        config["language"] = lang_code
        save_config()
        
        # إلغاء ربط أحداث التمرير قبل إنشاء نوافذ جديدة
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Button-4>")
        root.unbind_all("<Button-5>")
        
        # التحقق من وجود المودل وتحميله إذا لزم الأمر
        if not check_model_downloaded(model_name):
            threading.Thread(
                target=lambda: download_and_continue(model_name, progress_var, status_label, root),
                daemon=True
            ).start()
        else:
            root.destroy()
            start_transcription()
    
    # زر الإلغاء
    cancel_btn = Button(
        buttons_frame,
        text="إلغاء",
        font=("Segoe UI", 10),
        bg=colors["surface"],
        fg=colors["text"],
        bd=0,
        padx=15,
        pady=8,
        command=root.destroy
    )
    cancel_btn.pack(side="left", padx=5)
    
    # زر التأكيد
    confirm_btn = Button(
        buttons_frame,
        text="بدء التشغيل",
        font=("Segoe UI", 10),
        bg=colors["accent"],
        fg="white",
        bd=0,
        padx=15,
        pady=8,
        command=on_confirm
    )
    confirm_btn.pack(side="right", padx=5)
    
    # تخصيص نمط شريط التقدم
    style = ttk.Style()
    style.theme_use("clam")
    
    # تخصيص نمط شريط التقدم
    style.configure(
        "TProgressbar",
        troughcolor=colors["bg"],
        background=colors["accent"],
        thickness=10,
        borderwidth=0
    )
    
    # تخصيص أزرار الاختيار
    style.configure(
        "TRadiobutton",
        background=colors["card_bg"],
        foreground=colors["text"],
        font=("Segoe UI", 10)
    )
    
    root.mainloop()

# إضافة شريط تحميل وهمي يتوقف عند 99%
def show_fake_progress(progress_var, status_label, duration=20):
    total_steps = 99  # نتوقف عند 99% فقط
    step_time = duration / total_steps
    
    def update_progress(step=0):
        if step <= total_steps:
            # حساب النسبة المئوية
            percent = step
            
            # حساب الوقت المتبقي
            time_left = round((total_steps - step) * step_time)
            
            # تحديث شريط التقدم وحالة التحميل (نص مبسط)
            status_text = f"جاري التحميل: {percent}% - الوقت المتبقي: {time_left} ثانية"
            
            status_label.config(text=status_text)
            progress_var.set(percent)
            
            # جدولة التحديث التالي
            status_label.after(int(step_time * 1000), lambda: update_progress(step + 1))
    
    # بدء التحديثات
    update_progress(0)

# تحميل المودل وبدء التطبيق
def download_and_continue(model_name, progress_var, status_label, root):
    global MODEL
    
    # تحديث واجهة المستخدم
    status_label.config(text=f"جاري تحميل نموذج {model_name}...")
    progress_var.set(0)
    
    # البدء بالتحميل الفعلي في الخلفية
    def real_download():
        success = download_model(model_name, progress_var, status_label)
        
        if success:
            # تحميل النموذج
            try:
                global MODEL
                MODEL = whisper.load_model(model_name, download_root=MODELS_DIR)
                print(f"تم تحميل نموذج Whisper {model_name}")
            except Exception as e:
                print(f"خطأ أثناء تحميل النموذج: {e}")
                if root.winfo_exists():
                    root.after(0, lambda: messagebox.showerror("خطأ تحميل النموذج", 
                             f"تم تنزيل النموذج ولكن هناك مشكلة في تحميله: {str(e)}"))
                return
            
            # اكتمال التحميل
            if root.winfo_exists():
                root.after(0, lambda: progress_var.set(100))
                root.after(0, lambda: status_label.config(text=f"تم تحميل نموذج {model_name} بنجاح!"))
                root.after(1000, lambda: complete_download())
        else:
            # في حالة فشل التحميل
            if root.winfo_exists():
                root.after(0, lambda: messagebox.showerror("خطأ في التحميل", 
                                       "حدث خطأ أثناء تحميل النموذج. الرجاء التحقق من اتصال الإنترنت والمحاولة مرة أخرى."))
    
    # وظيفة لإكمال التحميل وبدء التطبيق
    def complete_download():
        try:
            # إغلاق نافذة التحميل أولاً
            root.destroy()
            
            # بدء نافذة الترجمة الجديدة
            start_transcription()
        except Exception as e:
            print(f"خطأ أثناء بدء التطبيق: {e}")
            messagebox.showerror("خطأ", f"حدث خطأ أثناء بدء التطبيق: {str(e)}")
    
    # بدء التحميل الحقيقي في خيط منفصل
    threading.Thread(target=real_download, daemon=True).start()

# دالة إعداد واجهة الترجمة
def setup_subtitles_ui():
    root = tk.Tk()
    root.title(APP_NAME)
    root.overrideredirect(True)  # إزالة شريط العنوان الافتراضي
    root.attributes("-topmost", True)  # جعل النافذة دائمًا في الأعلى
    root.attributes("-alpha", config.get("opacity", 0.85))  # الشفافية
    
    # تطبيق ألوان السمة
    colors = THEME[current_theme]
    root.configure(bg=colors["bg"])
    
    # ضبط حجم النافذة وموقعها
    screen = screeninfo.get_monitors()[0]
    window_width = 800
    window_height = 130
    x = (screen.width - window_width) // 2
    y = screen.height - window_height - 100
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # الإطار الرئيسي
    main_frame = Frame(root, bg=colors["bg"], padx=15, pady=10)
    main_frame.pack(fill="both", expand=True)
    
    # شريط العنوان
    title_frame = Frame(main_frame, bg=colors["bg"])
    title_frame.pack(fill="x", pady=(0, 5))
    
    # عنوان التطبيق مع اللغة
    lang_name = SUPPORTED_LANGUAGES.get(config["language"], config["language"].upper())
    title_label = Label(
        title_frame,
        text=f"{APP_NAME} ({lang_name})",
        font=("Segoe UI", 10),
        fg=colors["text_secondary"],
        bg=colors["bg"]
    )
    title_label.pack(side="left")
    
    # اسم المطور
    dev_label = Label(
        title_frame,
        text="Eng.Karim Omar",
        font=("Segoe UI", 8),
        fg=colors["text_secondary"],
        bg=colors["bg"]
    )
    dev_label.pack(side="right", padx=10)
    
    # مؤشر الحالة
    status_indicator = Frame(title_frame, width=6, height=8, bg=colors["success"])
    status_indicator.pack(side="left", padx=5)
    
    # إطار الترجمة الرئيسي
    subtitle_frame = Frame(main_frame, bg=colors["card_bg"], padx=15, pady=10)
    subtitle_frame.pack(fill="both", expand=True)
    
    # نص الترجمة
    font_size = config.get("font_size", 18)
    subtitle_label = Label(
        subtitle_frame,
        text="Hören...",  # النص الافتراضي
        font=("Segoe UI", font_size, "bold"),
        fg=colors["text"],
        bg=colors["card_bg"],
        wraplength=window_width - 10,
        justify="center"
    )
    subtitle_label.pack(fill="both", expand=True)
    
    # شريط التحكم
    control_frame = Frame(main_frame, bg=colors["bg"], height=20)
    control_frame.pack(fill="x", side="bottom")
    control_frame.pack_forget()
    
    # حالة العمل
    status_label = Label(
        control_frame,
        text="Status: Ready",
        font=("Segoe UI", 8),
        fg=colors["success"],
        bg=colors["bg"]
    )
    status_label.pack(side="left")
    
    # أزرار التحكم
    Button(
        control_frame,
        text="إخفاء [S]",
        font=("Segoe UI", 8),
        bg=colors["card_bg"],
        fg=colors["text"],
        activebackground=colors["accent"],
        bd=0, padx=8, pady=2,
        command=lambda: toggle_translation(root)
    ).pack(side="right", padx=2)
    
    Button(
        control_frame,
        text="الإعدادات [O]",
        font=("Segoe UI", 8),
        bg=colors["card_bg"],
        fg=colors["text"],
        activebackground=colors["accent"],
        bd=0, padx=8, pady=2,
        command=lambda: print("Open settings")  # يمكنك استبدالها بدالتك
    ).pack(side="right", padx=2)
    
    Button(
        control_frame,
        text="خروج [Q]",
        font=("Segoe UI", 8),
        bg=colors["card_bg"],
        fg=colors["text"],
        activebackground=colors["error"],
        bd=0, padx=8, pady=2,
        command=root.destroy
    ).pack(side="right", padx=2)
    
    root.bind("<Enter>", lambda e: control_frame.pack(fill="x", side="bottom"))
    root.bind("<Leave>", lambda e: control_frame.pack_forget())
    # إضافة دعم السحب والإفلات
    add_drag_support(root, title_frame)
    
    return root, subtitle_label, status_label, status_indicator
# إضافة دعم السحب والإفلات
def add_drag_support(root, drag_frame):
    drag_x, drag_y = 0, 0
    
    def start_drag(event):
        nonlocal drag_x, drag_y
        drag_x, drag_y = event.x, event.y
    
    def on_drag(event):
        nonlocal drag_x, drag_y
        x = root.winfo_x() + (event.x - drag_x)
        y = root.winfo_y() + (event.y - drag_y)
        root.geometry(f"+{x}+{y}")
    
    drag_frame.bind("<ButtonPress-1>", start_drag)
    drag_frame.bind("<B1-Motion>", on_drag)

# دالة لتمكين/تعطيل أحداث التمرير
def toggle_scrolling(root, canvas, enable=True):
    """دالة لتمكين أو تعطيل التمرير في النافذة"""
    if enable:
        # تمكين التمرير بعجلة الماوس
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # ربط أحداث التمرير بالنافذة
        root.bind_all("<MouseWheel>", on_mousewheel)  # Windows
        root.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux
        root.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))  # Linux
    else:
        # إلغاء ربط أحداث التمرير
        root.unbind_all("<MouseWheel>")
        root.unbind_all("<Button-4>")
        root.unbind_all("<Button-5>")

# تعديل دالة toggle_translation لإضافة منطق تمكين/تعطيل التمرير
def toggle_translation(root, subtitle_canvas=None):
    if root.winfo_viewable():
        # قبل الإخفاء، نلغي أحداث التمرير لتجنب التداخل
        if subtitle_canvas:
            toggle_scrolling(root, subtitle_canvas, False)
        root.withdraw()
    else:
        root.deiconify()
        # بعد الإظهار، نعيد تمكين أحداث التمرير
        if subtitle_canvas:
            toggle_scrolling(root, subtitle_canvas, True)

# عرض نافذة الإعدادات
def show_settings(parent_root):
    global settings_window_open, settings_window
    
    if settings_window_open and settings_window is not None:
        # إذا كانت النافذة مفتوحة، أغلقها
        try:
            settings_window.destroy()
        except:
            pass  # تجاهل الأخطاء إذا كانت النافذة قد أغلقت بالفعل
        settings_window_open = False
        settings_window = None
    else:
        # إذا لم تكن مفتوحة، افتح نافذة جديدة
        settings_window = tk.Toplevel(parent_root)
        settings_window.title(f"{APP_NAME} - الإعدادات")
        settings_window.attributes("-topmost", True)
        
        # تطبيق ألوان السمة
        colors = THEME[current_theme]
        settings_window.configure(bg=colors["bg"])
        
        # تعيين حجم وموقع النافذة
        width, height = 450, 500
        root_x, root_y = parent_root.winfo_x(), parent_root.winfo_y()
        settings_window.geometry(f"{width}x{height}+{root_x + 50}+{root_y - height // 2}")
        
        # إنشاء Canvas مع شريط تمرير
        settings_canvas = tk.Canvas(settings_window, bg=colors["bg"], highlightthickness=0)
        settings_canvas.pack(side="left", fill="both", expand=True)
        
        # إضافة شريط تمرير
        scrollbar = ttk.Scrollbar(settings_window, orient="vertical", command=settings_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        
        settings_canvas.configure(yscrollcommand=scrollbar.set)
        
        # إنشاء الإطار الرئيسي داخل الـCanvas
        main_frame = Frame(settings_canvas, bg=colors["bg"], padx=25, pady=25)
        canvas_frame = settings_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        
        # تحديث أبعاد المنطقة القابلة للتمرير
        def configure_scroll_region(event):
            settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))
        
        main_frame.bind("<Configure>", configure_scroll_region)
        
        def on_canvas_configure(event):
            settings_canvas.itemconfig(canvas_frame, width=event.width)
        
        settings_canvas.bind("<Configure>", on_canvas_configure)
        
        # تمكين التمرير بعجلة الماوس
        def on_mousewheel(event):
            settings_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        settings_window.bind_all("<MouseWheel>", on_mousewheel)
        settings_window.bind_all("<Button-4>", lambda e: settings_canvas.yview_scroll(-1, "units"))
        settings_window.bind_all("<Button-5>", lambda e: settings_canvas.yview_scroll(1, "units"))
        
        # رأس الصفحة
        header_frame = Frame(main_frame, bg=colors["bg"])
        header_frame.pack(fill="x", pady=(0, 20))
        
        Label(
            header_frame, 
            text="إعدادات التطبيق", 
            font=("Segoe UI", 18, "bold"), 
            fg=colors["text"], 
            bg=colors["bg"]
        ).pack(side="left")
        
        # قسم المظهر
        create_section_title(main_frame, "إعدادات المظهر")
        
        appearance_frame = create_glass_frame(main_frame)
        appearance_frame.pack(fill="x", pady=(0, 20))
        
        # اختيار السمة
        theme_row = Frame(appearance_frame, bg=colors["card_bg"], pady=8)
        theme_row.pack(fill="x")
        
        Label(
            theme_row, 
            text="سمة التطبيق:", 
            font=("Segoe UI", 10),
            fg=colors["text"], 
            bg=colors["card_bg"]
        ).pack(side="left")
        
        theme_var = StringVar(value=current_theme)
        
        theme_options_frame = Frame(theme_row, bg=colors["card_bg"])
        theme_options_frame.pack(side="right")
        
        ttk.Radiobutton(
            theme_options_frame,
            text="فاتح",
            variable=theme_var,
            value="light",
            command=lambda: change_theme("light", settings_window)
        ).pack(side="left", padx=5)
        
        ttk.Radiobutton(
            theme_options_frame,
            text="داكن",
            variable=theme_var,
            value="dark",
            command=lambda: change_theme("dark", settings_window)
        ).pack(side="left", padx=5)
        
        # تعديل الشفافية
        opacity_row = Frame(appearance_frame, bg=colors["card_bg"], pady=15)
        opacity_row.pack(fill="x")
        
        Label(
            opacity_row, 
            text="شفافية النافذة:", 
            font=("Segoe UI", 10),
            fg=colors["text"], 
            bg=colors["card_bg"]
        ).pack(side="left")
        
        opacity_var = tk.DoubleVar(value=config.get("opacity", 0.85))
        
        opacity_control = Frame(opacity_row, bg=colors["card_bg"])
        opacity_control.pack(side="right", fill="x", expand=True)
        
        opacity_scale = ttk.Scale(
            opacity_control, 
            from_=0.3, 
            to=1.0, 
            variable=opacity_var, 
            orient="horizontal",
            length=200
        )
        opacity_scale.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        opacity_label_text = tk.StringVar(value=f"{int(opacity_var.get()*100)}%")
        Label(
            opacity_control,
            textvariable=opacity_label_text,
            font=("Segoe UI", 9),
            fg=colors["text_secondary"],
            bg=colors["card_bg"],
            width=4
        ).pack(side="right")
        
        def update_opacity_label(*args):
            opacity_label_text.set(f"{int(opacity_var.get()*100)}%")
        
        opacity_var.trace("w", update_opacity_label)
        
        # حجم الخط
        font_row = Frame(appearance_frame, bg=colors["card_bg"], pady=15)
        font_row.pack(fill="x")
        
        Label(
            font_row, 
            text="حجم الخط:", 
            font=("Segoe UI", 10),
            fg=colors["text"], 
            bg=colors["card_bg"]
        ).pack(side="left")
        
        font_var = tk.IntVar(value=config.get("font_size", 18))
        
        font_control = Frame(font_row, bg=colors["card_bg"])
        font_control.pack(side="right", fill="x", expand=True)
        
        font_scale = ttk.Scale(
            font_control, 
            from_=12, 
            to=36,
            variable=font_var, 
            orient="horizontal",
            length=200
        )
        font_scale.pack(side="left", padx=(0, 10), fill="x", expand=True)
        
        font_label_text = tk.StringVar(value=f"{font_var.get()} pt")
        Label(
            font_control,
            textvariable=font_label_text,
            font=("Segoe UI", 9),
            fg=colors["text_secondary"],
            bg=colors["card_bg"],
            width=6
        ).pack(side="right")
        
        def update_font_label(*args):
            font_label_text.set(f"{font_var.get()} pt")
        
        font_var.trace("w", update_font_label)
        
        # إطار الأزرار
        buttons_frame = Frame(main_frame, bg=colors["bg"], pady=15)
        buttons_frame.pack(side="bottom", fill="x")
        
        # دالة حفظ الإعدادات
        def save_settings():
            config["opacity"] = opacity_var.get()
            config["font_size"] = font_var.get()
            config["theme"] = theme_var.get()
            save_config()
            settings_window.unbind_all("<MouseWheel>")
            settings_window.unbind_all("<Button-4>")
            settings_window.unbind_all("<Button-5>")
            apply_settings_to_main(parent_root)
            settings_window.destroy()
            settings_window_open = False
            settings_window = None
        
        # زر الحفظ
        Button(
            buttons_frame,
            text="حفظ",
            font=("Segoe UI", 10),
            bg=colors["accent"],
            fg="white",
            bd=0,
            padx=15,
            pady=8,
            command=save_settings
        ).pack(side="right", padx=5)
        
        # زر الإلغاء
        Button(
            buttons_frame,
            text="إلغاء",
            font=("Segoe UI", 10),
            bg=colors["surface"],
            fg=colors["text"],
            bd=0,
            padx=15,
            pady=8,
            command=lambda: [
                settings_window.unbind_all("<MouseWheel>"),
                settings_window.unbind_all("<Button-4>"),
                settings_window.unbind_all("<Button-5>"),
                settings_window.destroy(),
                globals().__setitem__('settings_window_open', False),
                globals().__setitem__('settings_window', None)
            ]
        ).pack(side="right", padx=5)
        
        # تعيين الحالة إلى مفتوحة
        settings_window_open = True
        
        # تنظيف عند إغلاق النافذة يدويًا
        def on_close():
            print("جاري إغلاق النافذة...")
            global settings_window_open, settings_window
            if settings_window is not None:
                try:
                    settings_window.destroy()
                    print("تم الإغلاق بنجاح")
                except Exception as e:
                    print(f"حدث خطأ: {e}")
            settings_window_open = False
            settings_window = None
        
        settings_window.protocol("WM_DELETE_WINDOW", on_close)
# تغيير السمة
def change_theme(theme, window=None):
    global current_theme, config
    current_theme = theme
    config["theme"] = theme
    save_config()
    
    if window:
        try:
            # إعادة تطبيق الألوان
            colors = THEME[current_theme]
            window.configure(bg=colors["bg"])
            
            # تحديث جميع العناصر في النافذة
            for widget in window.winfo_children():
                try:
                    if isinstance(widget, Frame):
                        widget.configure(bg=colors["bg"])
                    elif isinstance(widget, Label):
                        widget.configure(bg=colors["bg"], fg=colors["text"])
                except Exception:
                    pass  # تجاهل الأخطاء في حالة عدم وجود الخاصية
        except Exception:
            pass  # تجاهل الأخطاء العامة

# تطبيق الإعدادات المحدثة على النافذة الرئيسية
def apply_settings_to_main(root):
    colors = THEME[config["theme"]]
    
    # تحديث الشفافية
    root.attributes("-alpha", config["opacity"])
    
    # تحديث لون الخلفية
    root.configure(bg=colors["bg"])
    
    # تحديث جميع الإطارات والتسميات
    for widget in root.winfo_children():
        if isinstance(widget, Frame):
            widget.configure(bg=colors["bg"])
            for child in widget.winfo_children():
                update_widget_theme(child, colors)

# تحديث سمة عنصر واجهة
def update_widget_theme(widget, colors):
    try:
        if isinstance(widget, Frame):
            if "subtitle" in str(widget):
                widget.configure(bg=colors["card_bg"])
            else:
                widget.configure(bg=colors["bg"])
            
            for child in widget.winfo_children():
                update_widget_theme(child, colors)
        
        elif isinstance(widget, Label):
            if "subtitle" in str(widget):
                widget.configure(bg=colors["card_bg"], fg=colors["text"])
                # تطبيق حجم الخط إذا كان متاحًا في الإعدادات
                if "font_size" in config:
                    try:
                        widget.configure(font=("Segoe UI", config["font_size"], "bold"))
                    except:
                        pass
            elif "status" in str(widget):
                widget.configure(bg=colors["bg"], fg=colors["success"])
            else:
                # استخدام ألوان افتراضية للتصنيفات غير المعروفة
                if "card_bg" in str(widget.master):
                    widget.configure(bg=colors["card_bg"], fg=colors["text"])
                else:
                    widget.configure(bg=colors["bg"], fg=colors["text"])
        
        elif isinstance(widget, Button):
            widget.configure(bg=colors["surface"], fg=colors["text"])
    except:
        pass  # تجاهل أي أخطاء قد تحدث أثناء تحديث الألوان

# تنزيل موديل جديد
def download_new_model(parent, model_name):
    # إنشاء نافذة منبثقة للتحميل
    download_window = tk.Toplevel(parent)
    download_window.title("تحميل نموذج جديد")
    download_window.attributes("-topmost", True)
    
    # ضبط موقع وحجم النافذة
    width, height = 400, 250
    parent_x, parent_y = parent.winfo_x(), parent.winfo_y()
    download_window.geometry(f"{width}x{height}+{parent_x + 50}+{parent_y + 50}")
    
    # تطبيق ألوان السمة
    colors = THEME[current_theme]
    download_window.configure(bg=colors["bg"])
    
    # إنشاء Canvas مع شريط تمرير
    download_canvas = tk.Canvas(download_window, bg=colors["bg"], highlightthickness=0)
    download_canvas.pack(side="left", fill="both", expand=True)
    
    # إضافة شريط تمرير للcanvas
    scrollbar = ttk.Scrollbar(download_window, orient="vertical", command=download_canvas.yview)
    scrollbar.pack(side="right", fill="y")
    
    # ربط الـCanvas بشريط التمرير
    download_canvas.configure(yscrollcommand=scrollbar.set)
    
    # إنشاء الإطار الرئيسي داخل الـCanvas
    main_frame = Frame(download_canvas, bg=colors["bg"], padx=20, pady=15)
    
    # إضافة الإطار إلى Canvas
    canvas_frame = download_canvas.create_window((0, 0), window=main_frame, anchor="nw", width=width-20)
    
    # تحديث أبعاد المنطقة القابلة للتمرير عند تغير حجم الإطار
    def configure_scroll_region(event):
        download_canvas.configure(scrollregion=download_canvas.bbox("all"))
    
    main_frame.bind("<Configure>", configure_scroll_region)
    
    # تمكين التمرير بعجلة الماوس
    toggle_scrolling(download_window, download_canvas, True)
    
    # العنوان
    Label(
        main_frame,
        text=f"تحميل نموذج {model_name}",
        font=("Segoe UI", 12, "bold"),
        fg=colors["text"],
        bg=colors["bg"]
    ).pack(pady=(0, 10))
    
    # معلومات حول النموذج
    model_info = AVAILABLE_MODELS.get(model_name, {"size": "غير معروف", "description": "غير معروف"})
    Label(
        main_frame,
        text=f"حجم النموذج: {model_info['size']}\nالوصف: {model_info['description']}",
        fg=colors["text"],
        bg=colors["bg"],
        justify="right"
    ).pack(pady=(0, 15))
    
    # شريط التقدم
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        main_frame,
        variable=progress_var,
        orient="horizontal",
        length=100,
        mode="determinate"
    )
    progress_bar.pack(fill="x", pady=5)
    
    # حالة التحميل
    status_label = Label(
        main_frame,
        text="جاهز للتحميل...",
        fg=colors["text"],
        bg=colors["bg"]
    )
    status_label.pack(pady=5)
    
    # أزرار التحكم
    buttons_frame = Frame(main_frame, bg=colors["bg"], pady=10)
    buttons_frame.pack(fill="x")
    
    # وظيفة للإغلاق والتنظيف
    def close_window():
        toggle_scrolling(download_window, download_canvas, False)
        download_window.destroy()
    
    # زر بدء التحميل
    start_button = Button(
        buttons_frame,
        text="بدء التحميل",
        font=("Segoe UI", 10),
        bg=colors["accent"],
        fg="white",
        activebackground=colors["accent"],
        activeforeground="white",
        bd=0, padx=15, pady=5,
        command=lambda: threading.Thread(
            target=lambda: start_download(model_name, progress_var, status_label, start_button, download_window),
            daemon=True
        ).start()
    )
    start_button.pack(side="right", padx=5)
    
    # زر الإلغاء
    cancel_button = Button(
        buttons_frame,
        text="إلغاء",
        font=("Segoe UI", 10),
        bg=colors["card_bg"],
        fg=colors["text"],
        activebackground=colors["error"],
        activeforeground="white",
        bd=0, padx=15, pady=5,
        command=close_window
    )
    cancel_button.pack(side="right", padx=5)
    
    # التنظيف عند الإغلاق
    download_window.protocol("WM_DELETE_WINDOW", close_window)

# بدء تحميل النموذج
def start_download(model_name, progress_var, status_label, start_button, window):
    # تعطيل زر البدء
    start_button.configure(state="disabled")
    
    # بدء التحميل
    success = download_model(model_name, progress_var, status_label)
    
    if success:
        status_label.config(text=f"تم تحميل النموذج {model_name} بنجاح!")
        # إغلاق النافذة بعد فترة قصيرة
        window.after(2000, window.destroy)
    else:
        # تفعيل زر البدء مرة أخرى للمحاولة مرة أخرى
        start_button.configure(state="normal")

# تحديث واجهة المستخدم
def update_ui(root, subtitle_label, status_label, status_indicator):
    colors = THEME[current_theme]
    
    while not subtitle_queue.empty():
        try:
            data = subtitle_queue.get_nowait()
            
            if data[0] == "status":
                status = data[1]
                status_label.config(text=f"Status: {status}")
                
                # تغيير لون مؤشر الحالة حسب الوضع
                if "Listening" in status:
                    status_indicator.config(bg=colors["success"])
                elif "Processing" in status or "Transcribing" in status:
                    status_indicator.config(bg=colors["warning"])
                else:
                    status_indicator.config(bg=colors["accent"])
                    
            elif data[0] == "text":
                text, is_final = data[1], data[2]
                
                # تنسيق النص حسب نوعه (نهائي أو جزئي)
                if is_final:
                    subtitle_label.config(text=text, fg=colors["text"])
                else:
                    subtitle_label.config(text=text, fg=colors["text_secondary"])
                
                # تحديث حالة الاستماع
                status_label.config(text="Status: Listening", fg=colors["success"])
                status_indicator.config(bg=colors["success"])
        except:
            pass
    
    # جدولة التحديث القادم
    root.after(50, lambda: update_ui(root, subtitle_label, status_label, status_indicator))

# وظيفة بدء النسخ الرئيسية
def start_transcription():
    global MODEL, vad, config, current_theme
    
    # تحميل الإعدادات
    config = load_config()
    current_theme = config.get("theme", "dark")
    
    # تحميل المودل
    try:
        MODEL = whisper.load_model(config["model_size"], download_root=MODELS_DIR)
        print(f"تم تحميل نموذج Whisper {config['model_size']} على {DEVICE}")
    except Exception as e:
        print(f"خطأ في تحميل النموذج: {e}")
        messagebox.showerror("خطأ في التحميل", f"حدث خطأ أثناء تحميل نموذج {config['model_size']}.\n{str(e)}")
        create_model_selector()
        return
    
    # إعداد WebRTC VAD
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # استخدام وضع متوسط بدلاً من الوضع الأكثر تشدداً
    
    # البحث عن جهاز الصوت
    device_id = get_system_audio_device()
    if device_id is None:
        print("لم يتم العثور على جهاز صوت مناسب. يرجى التحقق من إعدادات الصوت.")
        messagebox.showerror("خطأ في جهاز الصوت", "لم يتم العثور على جهاز صوت مناسب.\nيرجى التحقق من إعدادات الصوت وأن الجهاز متصل بشكل صحيح.")
        return
    
    # إعداد واجهة المستخدم
    root, subtitle_label, status_label, status_indicator = setup_subtitles_ui()
    
    # الحصول على كائن subtitle_canvas
    subtitle_canvas = None
    for widget in root.winfo_children():
        if isinstance(widget, Frame):
            for child in widget.winfo_children():
                if isinstance(child, Frame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, tk.Canvas):
                            subtitle_canvas = grandchild
                            break
    
    print(f"جاري بدء العمل... (اضغط 'q' للخروج، 's' لإخفاء/إظهار الترجمة، 'o' للإعدادات)")
    print(f"النموذج: {config['model_size']}, اللغة: {config['language']}, الجهاز: {DEVICE}")
    
    # بدء مهمة النسخ في مؤشر ترابط منفصل
    transcription_thread = threading.Thread(target=transcribe_task, daemon=True)
    transcription_thread.start()
    
    # بدء تدفق الصوت
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=device_id,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        latency="low",
        callback=audio_callback
    )
    
    try:
        # بدء التدفق
        stream.start()
        
        # إظهار النافذة وبدء التحديث
        root.after(50, lambda: update_ui(root, subtitle_label, status_label, status_indicator))
        
        # إلغاء تسجيل الاختصارات السابقة إن وجدت
        keyboard.unhook_all()
        
        # تسجيل اختصارات لوحة المفاتيح بشكل صحيح
        keyboard.add_hotkey('s', lambda: toggle_translation(root, subtitle_canvas), suppress=True)
        keyboard.add_hotkey('o', lambda: show_settings(root), suppress=True)
        keyboard.add_hotkey('q', lambda: [toggle_scrolling(root, subtitle_canvas, False), root.destroy()], suppress=True)
                
        # تفعيل الاستجابة مباشرة للأزرار من واجهة Tkinter أيضًا
        root.bind("<KeyPress-s>", lambda event: toggle_translation(root, subtitle_canvas))
        root.bind("<KeyPress-o>", lambda event: [toggle_scrolling(root, subtitle_canvas, False), show_settings(root)])
        root.bind("<KeyPress-q>", lambda event: [toggle_scrolling(root, subtitle_canvas, False), root.destroy()])
        
        # حلقة الحدث الرئيسية
        root.mainloop()
        
    except Exception as e:
        print(f"خطأ: {e}")
    finally:
        # تنظيف
        if 'stream' in locals():
            stream.stop()
            stream.close()
        keyboard.unhook_all()
        
        try:
            root.destroy()
        except:
            pass
        
        # حفظ إعدادات الجهاز
        config["last_device_id"] = device_id
        save_config()

# الدالة الرئيسية
def main():
    try:
        # إعادة توجيه المخرجات للعمل في وضع النافذة
        log_file = redirect_stdout()
        print(f"بدء تشغيل التطبيق. السجلات محفوظة في: {log_file}")
        
        # تحميل الإعدادات في بداية البرنامج
        global config
        config = load_config()
        
        # التحقق مما إذا كان هذا أول استخدام
        first_run = not os.path.exists(CONFIG_FILE) or not check_model_downloaded(config.get("model_size", "tiny"))
        
        if first_run:
            print("👋 مرحباً بك في برنامج الترجمة الصوتية المباشرة")
            print("🔧 جاري إعداد البرنامج لأول مرة...")
            create_model_selector()
        else:
            start_transcription()
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البرنامج بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        messagebox.showerror("خطأ غير متوقع", f"حدث خطأ غير متوقع:\n{str(e)}")

def redirect_stdout():
    """إعادة توجيه المخرجات القياسية إلى ملف سجل"""
    try:
        import sys
        from datetime import datetime
        
        # الحصول على المسار الصحيح للسجلات
        if getattr(sys, 'frozen', False):
            # تشغيل كملف تنفيذي من PyInstaller
            app_dir = os.path.dirname(sys.executable)
        else:
            # تشغيل كسكريبت عادي
            app_dir = os.path.dirname(os.path.abspath(__file__))
        
        # إنشاء مجلد للسجلات إذا لم يكن موجودًا
        log_dir = os.path.join(app_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # إنشاء ملف سجل بتاريخ ووقت التشغيل
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"app_{timestamp}.log")
        
        # محاولة إعادة توجيه المخرجات القياسية
        try:
            sys.stdout = open(log_file, "w", encoding="utf-8")
            sys.stderr = sys.stdout
        except:
            # إذا فشل إعادة التوجيه، استخدم ملف بديل
            user_dir = os.path.expanduser("~")
            alt_log_file = os.path.join(user_dir, f"live_subtitles_log_{timestamp}.log")
            try:
                sys.stdout = open(alt_log_file, "w", encoding="utf-8")
                sys.stderr = sys.stdout
                return alt_log_file
            except:
                # إذا فشلت كل محاولات إعادة التوجيه، لا تفعل شيئًا
                pass
        
        return log_file
    except:
        return None

# دوال مساعدة لتصميم 2025
def create_modern_button(parent, text, command, icon=None, primary=False, **kwargs):
    """إنشاء زر بتصميم عصري بسيط"""
    colors = THEME[current_theme]
    
    # افتراضيات التنسيق
    padx = kwargs.pop('padx', 15)
    pady = kwargs.pop('pady', 8)
    
    if primary:
        button = Button(
            parent,
            text=text,
            font=("Segoe UI", 10, "normal"),
            bg=colors["accent"],
            fg="white",
            activebackground=colors["accent"],
            activeforeground="white",
            bd=0,
            padx=padx,
            pady=pady,
            command=command,
            **kwargs
        )
    else:
        button = Button(
            parent,
            text=text,
            font=("Segoe UI", 10, "normal"),
            bg=colors["surface"],
            fg=colors["text"],
            activebackground=colors["card_bg"],
            activeforeground=colors["accent"],
            bd=0,
            padx=padx,
            pady=pady,
            command=command,
            **kwargs
        )
    
    # إضافة تأثير بسيط عند التحويم
    def on_enter(e):
        if primary:
            button['bg'] = '#8D5FFF'  # لون أفتح قليلاً
        else:
            button['bg'] = colors["border"]
    
    def on_leave(e):
        if primary:
            button['bg'] = colors["accent"]
        else:
            button['bg'] = colors["surface"]
    
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    
    return button

def create_glass_frame(parent, **kwargs):
    """إنشاء إطار بتصميم بسيط"""
    colors = THEME[current_theme]
    padx = kwargs.pop('padx', 20)
    pady = kwargs.pop('pady', 20)
    
    frame = Frame(
        parent,
        bg=colors["card_bg"],
        bd=1,
        relief="solid",
        padx=padx,
        pady=pady,
        **kwargs
    )
    
    return frame

def create_section_title(parent, text):
    """إنشاء عنوان قسم بتصميم عصري"""
    colors = THEME[current_theme]
    
    title_frame = Frame(parent, bg=colors["card_bg"])
    title_frame.pack(fill="x", pady=(10, 5), anchor="w")
    
    accent_line = Frame(title_frame, bg=colors["accent"], width=3, height=18)
    accent_line.pack(side="left", padx=(0, 10))
    
    Label(
        title_frame,
        text=text,
        font=("Segoe UI", 11, "bold"),
        fg=colors["text"],
        bg=colors["card_bg"]
    ).pack(side="left")
    
    return title_frame

if __name__ == "__main__":
    main()
