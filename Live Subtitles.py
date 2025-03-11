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

#pyinstaller --onefile --windowed --add-data "C:\Users\karim\OneDrive\Desktop\Python\.venv\Lib\site-packages\whisper\assets;whisper\assets" LiveCaptions.py 

# الثوابت العامة

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
MAX_SEGMENT_DURATION = 2.5  # تقليل المدة القصوى للتجزئة للحصول على استجابة أسرع
MAX_QUEUE_SIZE = 5  # تقليل حجم الصف للذاكرة

# الألوان والمظهر العام
THEME = {
    "dark": {
        "bg": "#121212",
        "card_bg": "#1E1E1E",
        "text": "#FFFFFF",
        "text_secondary": "#AAAAAA",
        "accent": "#2196F3",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#F44336"
    },
    "light": {
        "bg": "#F5F5F5",
        "card_bg": "#FFFFFF",
        "text": "#212121",
        "text_secondary": "#757575",
        "accent": "#2196F3",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#F44336"
    }
}

# الخيارات المتاحة للمودل
AVAILABLE_MODELS = {
    "خفيف": {"size": "~75MB", "description": "أسرع، دقة منخفضة", "key": "tiny"},
    "متوسط": {"size": "~500MB", "description": "متوازن، دقة جيدة", "key": "small"},
    "دقيق": {"size": "~1.5GB", "description": "دقة عالية، أبطأ", "key": "medium"}
}

# اللغات المدعومة
SUPPORTED_LANGUAGES = {
    "ar": "العربية",
    "de": "الألمانية",
    "en": "الإنجليزية",
    "es": "الإسبانية",
    "fr": "الفرنسية",
    "ru": "الروسية",
    "zh": "الصينية",
    "ja": "اليابانية",
    "it": "الإيطالية"
}

# تهيئة المتغيرات العامة
audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)  # تحديد الحجم الأقصى لتجنب تسرب الذاكرة
processing_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
subtitle_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
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
                loaded_config = json.load(f)
                # دمج الإعدادات المحملة مع الإعدادات الافتراضية
                config = {**default_config, **loaded_config}
        else:
            config = default_config
    except Exception as e:
        print(f"⚠ خطأ في تحميل الإعدادات: {e}")
        config = default_config
    
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

# تنزيل المودل مع عرض شريط التقدم
def download_model(model_name, progress_var, status_label):
    try:
        # تحديث الحالة
        status_label.config(text=f"جاري تحميل نموذج {model_name}...")
        
        # تنزيل المودل باستخدام whisper (بدون progress)
        whisper.load_model(model_name, download_root=MODELS_DIR, 
                          in_memory=False)
        
        # تحديث الإعدادات
        if "models_downloaded" not in config:
            config["models_downloaded"] = []
        
        if model_name not in config["models_downloaded"]:
            config["models_downloaded"].append(model_name)
        
        config["model_size"] = model_name
        save_config()
        
        progress_var.set(100)
        status_label.config(text=f"تم تحميل نموذج {model_name} بنجاح!")
        return True
    except Exception as e:
        status_label.config(text=f"خطأ في تحميل النموذج: {str(e)}")
        print(f"⚠ خطأ في تحميل المودل: {e}")
        return False

# معالجة الصوت وإزالة الضوضاء
def process_audio(audio_data):
    # تجنب المعالجة إذا كانت البيانات فارغة
    if len(audio_data) == 0:
        return np.zeros(1, dtype=np.float32)
        
    # تطبيع مستوى الصوت - تبسيط الحسابات
    max_val = np.max(np.abs(audio_data))
    if max_val > 0.01:  # تجنب القسمة على أرقام صغيرة جداً
        normalized = audio_data / max_val * 0.9
    else:
        normalized = audio_data
    
    # تطبيق فلتر فقط إذا كانت هناك بيانات كافية
    if len(normalized) > 10:
        b, a = butter_highpass(cutoff=100, fs=SAMPLE_RATE, order=2)
        filtered = lfilter(b, a, normalized)
        return filtered.astype(np.float32)
    
    return normalized.astype(np.float32)

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
    
    # تجاهل البيانات الفارغة أو الخاطئة
    if status or np.max(np.abs(indata)) < 0.001:
        return
    
    try:
        # تحويل البيانات مباشرة
        audio_data = indata.flatten().astype(np.float32)
        
        # تحديد وجود كلام بطريقة مبسطة
        for i in range(0, len(audio_data) - FRAME_SIZE + 1, FRAME_SIZE):
            frame = audio_data[i:i + FRAME_SIZE]
            
            # تحديد ما إذا كان الإطار يحتوي على كلام
            speech_detected = is_speech(frame, vad)
            
            if speech_detected:
                if not is_speaking:
                    is_speaking = True
                    subtitle_queue.put(("status", "Listening..."))
                
                audio_buffer.append(frame)
                silence_counter = 0
                last_speech_time = time.time()
                
                # التجزئة المبكرة للاستجابة الأسرع
                if len(audio_buffer) >= 15 and processing_queue.qsize() < 2:
                    current_segment = np.concatenate(audio_buffer)
                    processing_queue.put((current_segment, False))
            else:
                # تحديث عداد الصمت
                silence_counter += 1
                
                # حساب مدة الصمت بطريقة مبسطة
                silence_duration = silence_counter * (FRAME_SIZE / SAMPLE_RATE)
                
                # إرسال المقطع عند اكتشاف صمت كافٍ
                if is_speaking and (silence_duration >= SILENCE_THRESHOLD or 
                                   time.time() - last_segment_time >= MAX_SEGMENT_DURATION) and len(audio_buffer) > 10:
                    
                    if not audio_queue.full():
                        full_segment = np.concatenate(audio_buffer)
                        is_speaking = False
                        audio_queue.put(full_segment, block=False)
                        processing_queue.put((full_segment, True), block=False)
                    
                    # إعادة تعيين المخزن المؤقت
                    audio_buffer.clear()
                    silence_counter = 0
                    last_segment_time = time.time()
                    subtitle_queue.put(("status", "Processing..."))
    
    except Exception as e:
        print(f"⚠ خطأ في معالجة الصوت: {e}")

# مهمة نسخ الصوت محسنة
def transcribe_task():
    global context_buffer
    
    # إعداد خيارات النسخ للأداء الأفضل
    transcribe_options = {
        "fp16": (torch.cuda.is_available()),
        "language": config.get("language", "de"),
        "task": "transcribe",
        "beam_size": 2,  # تقليل عدد الاحتمالات لتحسين السرعة
        "best_of": 1,
        "temperature": 0.0
    }
    
    while True:
        try:
            if not processing_queue.empty():
                audio_segment, is_final = processing_queue.get(timeout=0.1)
                
                # تجاهل المقاطع الصغيرة جداً
                if len(audio_segment) < SAMPLE_RATE * 0.25:  # أقل من 250 مللي ثانية
                    continue
                
                # معالجة الصوت
                audio_processed = process_audio(audio_segment)
                
                # تحديث حالة المعالجة
                subtitle_queue.put(("status", "Transcribing..."))
                
                # استخدام السياق فقط للمقاطع النهائية لتحسين الأداء
                if is_final and context_buffer:
                    transcribe_options["prompt"] = " ".join(context_buffer[-2:])
                
                # النسخ
                result = MODEL.transcribe(
                    audio_processed,
                    **transcribe_options
                )
                
                # معالجة النتيجة
                detected_text = result["text"].strip()
                
                if detected_text:
                    if is_final:
                        context_buffer.append(detected_text)
                        if len(context_buffer) > 3:  # تقليل حجم السياق لتوفير الذاكرة
                            context_buffer = context_buffer[-3:]
                    
                    subtitle_queue.put(("text", detected_text, is_final))
            else:
                time.sleep(0.05)  # راحة قصيرة لتقليل استهلاك وحدة المعالجة المركزية
                
        except queue.Empty:
            # تجاهل الأخطاء الناتجة عن فشل الحصول على البيانات من الصف
            pass
        except Exception as e:
            print(f"⚠ خطأ في النسخ: {e}")
            time.sleep(0.1)

# ----- واجهة المستخدم المحسنة -----

# إنشاء نافذة اختيار المودل واللغة
def create_model_selector():
    root = tk.Tk()
    root.title(f"{APP_NAME} - إعداد أول مرة")
    
    # ضبط حجم النافذة وموقعها
    width, height = 500, 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # تعيين الألوان من السمة الحالية
    colors = THEME[current_theme]
    root.configure(bg=colors["bg"])
    
    # إنشاء الإطار الرئيسي
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)
    
    # عنوان التطبيق
    title_label = ttk.Label(
        main_frame, 
        text=f"{APP_NAME} v{APP_VERSION}", 
        font=("Helvetica", 16, "bold")
    )
    title_label.pack(pady=(0, 20))
    
    # معلومات ترحيبية
    welcome_label = ttk.Label(
        main_frame,
        text="مرحباً بك! يبدو أنك تستخدم هذا التطبيق لأول مرة.\nالرجاء اختيار نموذج الذكاء الاصطناعي واللغة المفضلة لديك.",
        wraplength=width-50,
        justify="center"
    )
    welcome_label.pack(pady=(0, 20))
    
    # إطار اختيار المودل
    model_frame = ttk.LabelFrame(main_frame, text="اختر نموذج الترجمة")
    model_frame.pack(fill="x", pady=10, padx=10)
    
    # متغير لتخزين اختيار المودل
    model_var = StringVar(root)
    model_var.set(list(AVAILABLE_MODELS.keys())[0])  # القيمة الافتراضية
    
    # إنشاء اختيارات المودل المعدلة
    for i, (model_name, model_info) in enumerate(AVAILABLE_MODELS.items()):
        model_radio = ttk.Radiobutton(
            model_frame,
            text=f"{model_name} ({model_info['size']}): {model_info['description']}",
            variable=model_var,
            value=model_name
        )
        model_radio.pack(anchor="w", padx=10, pady=5)
    
    # إطار اختيار اللغة
    lang_frame = ttk.LabelFrame(main_frame, text="اختر اللغة المستهدفة")
    lang_frame.pack(fill="x", pady=10, padx=10)
    
    # متغير لتخزين اختيار اللغة
    lang_var = StringVar(root)
    lang_var.set(config["language"])  # القيمة الافتراضية
    
    # ترتيب اللغات في صفوف
    lang_grid_frame = ttk.Frame(lang_frame)
    lang_grid_frame.pack(fill="x", padx=10, pady=5)
    
    row, col = 0, 0
    for lang_code, lang_name in SUPPORTED_LANGUAGES.items():
        lang_radio = ttk.Radiobutton(
            lang_grid_frame,
            text=f"{lang_name} ({lang_code})",
            variable=lang_var,
            value=lang_code
        )
        lang_radio.grid(row=row, column=col, sticky="w", padx=5, pady=3)
        col += 1
        if col >= 2:  # عرض عمودين
            col = 0
            row += 1
    
    # شريط التقدم وحالة التحميل
    progress_frame = ttk.Frame(main_frame)
    progress_frame.pack(fill="x", pady=10, padx=10)
    
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        progress_frame, 
        variable=progress_var,
        orient="horizontal",
        length=100, 
        mode="determinate"
    )
    progress_bar.pack(fill="x", pady=5)
    
    status_label = ttk.Label(
        progress_frame,
        text="جاهز للتحميل...",
        anchor="center"
    )
    status_label.pack(pady=5)
    
    # العبارة الإسلامية (نبقيها فقط في صفحة الإعدادات)
    islamic_label = ttk.Label(
        main_frame,
        text="🤲 لا تنسونا من صالح دعائكم في هذه الأيام المفترجة 🌙",
        font=("Arial", 10, "italic"),
        foreground="#E91E63",  # لون وردي محمر مختلف
        justify="center"
    )
    islamic_label.pack(pady=10)
    
    # زر التأكيد
    def on_confirm():
        model_name = model_var.get()
        model_key = AVAILABLE_MODELS[model_name]["key"]  # استخدام المفتاح المقابل للنموذج
        lang_code = lang_var.get()
        
        # تحديث الإعدادات
        config["model_size"] = model_key  # نخزن المفتاح الحقيقي للنموذج
        config["language"] = lang_code
        save_config()
        
        # التحقق من وجود المودل وتحميله إذا لزم الأمر
        if not check_model_downloaded(model_key):
            threading.Thread(
                target=lambda: download_and_continue(model_key, progress_var, status_label, root),
                daemon=True
            ).start()
        else:
            root.destroy()
            start_transcription()
    
    confirm_button = ttk.Button(
        main_frame,
        text="تأكيد وبدء التشغيل",
        command=on_confirm
    )
    confirm_button.pack(pady=20)
    
    # تطبيق سمة ttk
    style = ttk.Style()
    if current_theme == "dark":
        style.theme_use("clam")
        style.configure("TFrame", background=colors["bg"])
        style.configure("TLabelframe", background=colors["bg"])
        style.configure("TLabelframe.Label", background=colors["bg"], foreground=colors["text"])
        style.configure("TLabel", background=colors["bg"], foreground=colors["text"])
        style.configure("TRadiobutton", background=colors["bg"], foreground=colors["text"])
        style.configure("TButton", background=colors["accent"], foreground=colors["text"])
    
    root.mainloop()

# تحميل المودل وبدء التطبيق
def download_and_continue(model_name, progress_var, status_label, root):
    success = download_model(model_name, progress_var, status_label)
    if success:
        # تحديث الحالة والانتظار لفترة قصيرة
        status_label.config(text=f"تم تحميل نموذج {model_name} بنجاح! جاري بدء التطبيق...")
        # جدولة الانتقال للخطوة التالية بعد ثانيتين لضمان ظهور الرسالة
        root.after(2000, lambda: proceed_after_download(root))
    else:
        messagebox.showerror("خطأ في التحميل", "حدث خطأ أثناء تحميل النموذج.\nالرجاء التحقق من اتصال الإنترنت والمحاولة مرة أخرى.")

# دالة جديدة للانتقال بعد التحميل
def proceed_after_download(root):
    root.destroy()
    start_transcription()

# إعداد واجهة الترجمة المباشرة المحسنة
def setup_subtitles_ui():
    root = tk.Tk()
    root.title(APP_NAME)
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", config.get("opacity", 0.7))  # زيادة الشفافية
    
    # تطبيق ألوان السمة
    colors = THEME[current_theme]
    
    modern_bg = "#181818" if current_theme == "dark" else "#F5F5F5"
    modern_card_bg = "#282828" if current_theme == "dark" else "#FFFFFF"
    modern_text = "#FFFFFF" if current_theme == "dark" else "#000000"
    
    root.configure(bg=modern_bg)
    
    # ضبط حجم النافذة
    screen = screeninfo.get_monitors()[0]
    window_width = 750
    window_height = 100
    x = (screen.width - window_width) // 2
    y = screen.height - window_height - 50
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # إطار الترجمة الرئيسي - مبسط
    main_frame = Frame(root, bg=modern_bg, padx=5, pady=5)
    main_frame.pack(fill="both", expand=True)
    
    subtitle_frame = Frame(main_frame, bg=modern_card_bg, padx=15, pady=8)
    subtitle_frame.pack(fill="both", expand=True)
    
    subtitle_frame.config(highlightbackground="#444444" if current_theme == "dark" else "#DDDDDD", 
                        highlightthickness=1)
    
    # نص الترجمة
    font_size = config.get("font_size", 18)
    subtitle_label = Label(
        subtitle_frame, 
        text="جاهز للاستماع...", 
        font=("Segoe UI", font_size, "bold"), 
        fg=modern_text, 
        bg=modern_card_bg, 
        wraplength=window_width-40, 
        justify="center"
    )
    subtitle_label.pack(fill="both", expand=True)
    
    # شريط التحكم المخفي
    control_frame = Frame(main_frame, bg=modern_bg, pady=2)
    control_frame.pack(fill="x")
    control_frame.pack_forget()
    
    # حالة العمل
    status_indicator = Frame(control_frame, width=8, height=8, bg=colors["success"])
    status_indicator.pack(side="left", padx=5)
    
    status_label = Label(
        control_frame, 
        text="Ready", 
        font=("Segoe UI", 8), 
        fg=colors["text_secondary"], 
        bg=modern_bg
    )
    status_label.pack(side="left")
    
    # اختصارات
    shortcuts_label = Label(
        control_frame,
        text="[S] إخفاء | [O] إعدادات | [Q] خروج",
        font=("Segoe UI", 8),
        fg=colors["text_secondary"],
        bg=modern_bg
    )
    shortcuts_label.pack(side="right", padx=5)
    
    # إضافة دعم السحب
    add_drag_support(root, subtitle_frame)
    
    # إظهار وإخفاء شريط التحكم
    def show_controls(event):
        control_frame.pack(fill="x", before=subtitle_frame)
        
    def hide_controls(event):
        control_frame.pack_forget()
    
    root.bind("<Enter>", show_controls)
    root.bind("<Leave>", hide_controls)
    
    # متغير تتبع حالة النافذة
    root.window_visible = True
    
    def toggle_visibility():
        if root.window_visible:
            root.withdraw()
            root.window_visible = False
        else:
            root.deiconify()
            root.focus_force()
            root.window_visible = True
    
    # اختصارات المفاتيح الرئيسية فقط
    keyboard.unhook_all()  # إلغاء أي اختصارات سابقة
    keyboard.add_hotkey('s', toggle_visibility)
    keyboard.add_hotkey('o', lambda: show_settings(root) if root.window_visible else None)
    keyboard.add_hotkey('q', root.destroy)
    
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

# عرض نافذة الإعدادات
def show_settings(parent_root):
    # حفظ الموقع الحالي للنافذة الرئيسية
    root_x, root_y = parent_root.winfo_x(), parent_root.winfo_y()
    
    settings = tk.Toplevel(parent_root)
    settings.title(f"{APP_NAME} - الإعدادات")
    settings.attributes("-topmost", True)
    
    # تطبيق ألوان السمة
    colors = THEME[current_theme]
    settings.configure(bg=colors["bg"])
    
    # تعيين حجم وموقع النافذة
    width, height = 400, 450
    settings.geometry(f"{width}x{height}+{root_x + 50}+{root_y - height // 2}")
    
    # الإطار الرئيسي
    main_frame = Frame(settings, bg=colors["bg"], padx=20, pady=15)
    main_frame.pack(fill="both", expand=True)
    
    # العنوان
    Label(
        main_frame, 
        text="إعدادات التطبيق", 
        font=("Segoe UI", 14, "bold"), 
        fg=colors["text"], 
        bg=colors["bg"]
    ).pack(pady=(0, 15))
    
    # إطار إعدادات المظهر
    appearance_frame = Frame(main_frame, bg=colors["card_bg"], padx=15, pady=10)
    appearance_frame.pack(fill="x", pady=5)
    
    Label(
        appearance_frame, 
        text="المظهر", 
        font=("Segoe UI", 10, "bold"), 
        fg=colors["text"], 
        bg=colors["card_bg"]
    ).pack(anchor="w")
    
    # اختيار السمة
    theme_frame = Frame(appearance_frame, bg=colors["card_bg"])
    theme_frame.pack(fill="x", pady=5)
    
    theme_var = StringVar(value=current_theme)
    
    Label(
        theme_frame, 
        text="السمة:", 
        fg=colors["text"], 
        bg=colors["card_bg"]
    ).pack(side="left")
    
    theme_menu = OptionMenu(
        theme_frame, 
        theme_var, 
        *THEME.keys(), 
        command=lambda v: change_theme(v, settings)
    )
    theme_menu.configure(bg=colors["card_bg"], fg=colors["text"], activebackground=colors["accent"])
    theme_menu.pack(side="right")
    
    # تعديل الشفافية
    opacity_frame = Frame(appearance_frame, bg=colors["card_bg"], pady=5)
    opacity_frame.pack(fill="x")
    
    Label(
        opacity_frame, 
        text="الشفافية:", 
        fg=colors["text"], 
        bg=colors["card_bg"]
    ).pack(side="left")
    
    opacity_var = tk.DoubleVar(value=config.get("opacity", 0.85))
    opacity_scale = ttk.Scale(
        opacity_frame, 
        from_=0.3, 
        to=1.0, 
        variable=opacity_var, 
        orient="horizontal",
        length=200
    )
    opacity_scale.pack(side="right")
    
    # حجم الخط
    font_frame = Frame(appearance_frame, bg=colors["card_bg"], pady=5)
    font_frame.pack(fill="x")
    
    Label(
        font_frame, 
        text="حجم الخط:", 
        fg=colors["text"], 
        bg=colors["card_bg"]
    ).pack(side="left")
    
    font_var = tk.IntVar(value=config.get("font_size", 18))
    font_scale = ttk.Scale(
        font_frame, 
        from_=12, 
        to=36, 
        variable=font_var, 
        orient="horizontal",
        length=200
    )
    font_scale.pack(side="right")
    
    # إطار إعدادات اللغة والنموذج
    model_frame = Frame(main_frame, bg=colors["card_bg"], padx=15, pady=10)
    model_frame.pack(fill="x", pady=10)
    
    Label(
        model_frame, 
        text="إعدادات النموذج واللغة", 
        font=("Segoe UI", 10, "bold"), 
        fg=colors["text"], 
        bg=colors["card_bg"]
    ).pack(anchor="w")
    
    # اختيار اللغة
    lang_frame = Frame(model_frame, bg=colors["card_bg"], pady=5)
    lang_frame.pack(fill="x")
    
    Label(
        lang_frame, 
        text="اللغة المستهدفة:", 
        fg=colors["text"], 
        bg=colors["card_bg"]
    ).pack(side="left")
    
    lang_var = StringVar(value=config["language"])
    lang_menu = OptionMenu(
        lang_frame, 
        lang_var, 
        *SUPPORTED_LANGUAGES.keys(),
    )
    lang_menu.configure(bg=colors["card_bg"], fg=colors["text"], activebackground=colors["accent"])
    lang_menu.pack(side="right")
    
    # اختيار المودل
    model_label_frame = Frame(model_frame, bg=colors["card_bg"], pady=5)
    model_label_frame.pack(fill="x")
    
    Label(
        model_label_frame, 
        text="تغيير النموذج:", 
        fg=colors["text"], 
        bg=colors["card_bg"]
    ).pack(side="left")
    
    model_var = StringVar(value=config["model_size"])
    model_menu = OptionMenu(
        model_label_frame, 
        model_var, 
        *AVAILABLE_MODELS.keys()
    )
    model_menu.configure(bg=colors["card_bg"], fg=colors["text"], activebackground=colors["accent"])
    model_menu.pack(side="right")
    
    # زر لتحميل موديل جديد
    new_model_frame = Frame(model_frame, bg=colors["card_bg"], pady=10)
    new_model_frame.pack(fill="x")
    
    download_button = Button(
        new_model_frame,
        text="تحميل نموذج جديد",
        font=("Segoe UI", 9),
        bg=colors["accent"],
        fg="white",
        activebackground=colors["accent"],
        activeforeground="white",
        bd=0, padx=10, pady=5,
        command=lambda: download_new_model(settings, model_var.get())
    )
    download_button.pack(side="right", padx=5)
    
    # أزرار الحفظ والإلغاء
    buttons_frame = Frame(main_frame, bg=colors["bg"], pady=15)
    buttons_frame.pack(fill="x")
    
    # دالة حفظ الإعدادات
    def save_settings():
        # تحديث الإعدادات الجديدة
        config["theme"] = theme_var.get()
        config["opacity"] = opacity_var.get()
        config["font_size"] = font_var.get()
        config["language"] = lang_var.get()
        if config["model_size"] != model_var.get() and check_model_downloaded(model_var.get()):
            config["model_size"] = model_var.get()
            # إعادة تحميل المودل
            global MODEL
            MODEL = whisper.load_model(config["model_size"], download_root=MODELS_DIR)
        
        # حفظ الإعدادات
        save_config()
        
        # تطبيق التغييرات على النافذة الرئيسية
        apply_settings_to_main(parent_root)
        
        # إغلاق نافذة الإعدادات
        settings.destroy()
    
    # زر الحفظ
    save_button = Button(
        buttons_frame,
        text="حفظ التغييرات",
        font=("Segoe UI", 10),
        bg=colors["success"],
        fg="white",
        activebackground=colors["success"],
        activeforeground="white",
        bd=0, padx=15, pady=5,
        command=save_settings
    )
    save_button.pack(side="right", padx=5)
    
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
        command=settings.destroy
    )
    cancel_button.pack(side="right", padx=5)

# تغيير السمة
def change_theme(theme, window=None):
    global current_theme
    current_theme = theme
    config["theme"] = theme
    save_config()
    
    if window:
        # إعادة تطبيق الألوان
        colors = THEME[current_theme]
        window.configure(bg=colors["bg"])
        
        # تحديث جميع العناصر في النافذة
        for widget in window.winfo_children():
            if isinstance(widget, Frame):
                widget.configure(bg=colors["bg"])
            elif isinstance(widget, Label):
                widget.configure(bg=colors["bg"], fg=colors["text"])

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
    if isinstance(widget, Frame):
        if widget.winfo_name() in ["!subtitle_frame"]:
            widget.configure(bg=colors["card_bg"])
        else:
            widget.configure(bg=colors["bg"])
        
        for child in widget.winfo_children():
            update_widget_theme(child, colors)
    
    elif isinstance(widget, Label):
        parent_name = widget.winfo_parent().split(".")[-1]
        if "subtitle" in widget.winfo_name():
            widget.configure(bg=colors["card_bg"], fg=colors["text"], font=("Segoe UI", config["font_size"], "bold"))
        elif "status" in widget.winfo_name():
            widget.configure(bg=colors["bg"], fg=colors["success"])
        elif "title" in widget.winfo_name():
            widget.configure(bg=colors["bg"], fg=colors["text_secondary"])
        else:
            # للعناصر في إطارات أخرى
            if parent_name in ["!subtitle_frame", "card_bg"]:
                widget.configure(bg=colors["card_bg"], fg=colors["text"])
            else:
                widget.configure(bg=colors["bg"], fg=colors["text"])
    
    elif isinstance(widget, Button):
        if "exit" in widget.cget("text").lower() or "خروج" in widget.cget("text"):
            widget.configure(bg=colors["card_bg"], fg=colors["text"], activebackground=colors["error"])
        else:
            widget.configure(bg=colors["card_bg"], fg=colors["text"], activebackground=colors["accent"])

# تنزيل موديل جديد
def download_new_model(parent, model_name):
    # إنشاء نافذة منبثقة للتحميل
    download_window = tk.Toplevel(parent)
    download_window.title("تحميل نموذج جديد")
    download_window.attributes("-topmost", True)
    
    # ضبط موقع وحجم النافذة
    width, height = 400, 200
    parent_x, parent_y = parent.winfo_x(), parent.winfo_y()
    download_window.geometry(f"{width}x{height}+{parent_x + 50}+{parent_y + 50}")
    
    # تطبيق ألوان السمة
    colors = THEME[current_theme]
    download_window.configure(bg=colors["bg"])
    
    # إنشاء الإطار الرئيسي
    main_frame = Frame(download_window, bg=colors["bg"], padx=20, pady=15)
    main_frame.pack(fill="both", expand=True)
    
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
    Button(
        buttons_frame,
        text="إلغاء",
        font=("Segoe UI", 10),
        bg=colors["card_bg"],
        fg=colors["text"],
        activebackground=colors["error"],
        activeforeground="white",
        bd=0, padx=15, pady=5,
        command=download_window.destroy
    ).pack(side="right", padx=5)

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
    
    # معالجة كل البيانات المتاحة في مرة واحدة
    updates = []
    while not subtitle_queue.empty():
        try:
            updates.append(subtitle_queue.get_nowait())
        except:
            break
    
    # تطبيق التحديثات المهمة فقط
    status_updated = False
    text_updated = False
    
    for data in updates:
        if data[0] == "status" and not status_updated:
            status = data[1]
            status_label.config(text=f"Status: {status}")
            
            if "Listening" in status:
                status_indicator.config(bg=colors["success"])
            elif "Processing" in status or "Transcribing" in status:
                status_indicator.config(bg=colors["warning"])
            else:
                status_indicator.config(bg=colors["accent"])
            
            status_updated = True
                
        elif data[0] == "text":
            text, is_final = data[1], data[2]
            
            if is_final or not text_updated:
                subtitle_label.config(text=text, fg=colors["text"] if is_final else colors["text_secondary"])
                text_updated = True
    
    # جدولة التحديث التالي بوتيرة أبطأ قليلاً لتقليل استهلاك المعالج
    root.after(100, lambda: update_ui(root, subtitle_label, status_label, status_indicator))

# وظيفة بدء النسخ الرئيسية
def start_transcription():
    global MODEL, vad, config, current_theme
    
    # تحميل الإعدادات
    config = load_config()
    current_theme = config.get("theme", "dark")
    
    # تحميل المودل - استخدام محرك GPU إذا كان متاحًا للسرعة
    try:
        # ضبط استخدام CUDA للأداء السريع
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # تفريغ الذاكرة المؤقتة لتحسين الأداء
            torch.backends.cudnn.benchmark = True  # تمكين معايير الأداء
        
        MODEL = whisper.load_model(config["model_size"], download_root=MODELS_DIR)
        print(f"🔄 تم تحميل نموذج Whisper {config['model_size']} على {DEVICE}")
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")
        messagebox.showerror("خطأ في التحميل", f"حدث خطأ أثناء تحميل نموذج {config['model_size']}.\n{str(e)}")
        create_model_selector()
        return
    
    # إعداد VAD
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    
    # البحث عن جهاز الصوت
    device_id = get_system_audio_device()
    if device_id is None:
        print("❌ لم يتم العثور على جهاز صوت مناسب. يرجى التحقق من إعدادات الصوت.")
        messagebox.showerror("خطأ في جهاز الصوت", "لم يتم العثور على جهاز صوت مناسب.\nيرجى التحقق من إعدادات الصوت وأن الجهاز متصل بشكل صحيح.")
        return
    
    # تنظيف المتغيرات العامة قبل البدء
    global audio_buffer, silence_counter, is_speaking
    audio_buffer = []
    silence_counter = 0
    is_speaking = False
    
    # تفريغ قوائم الانتظار
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: pass
    
    while not processing_queue.empty():
        try: processing_queue.get_nowait()
        except: pass
    
    while not subtitle_queue.empty():
        try: subtitle_queue.get_nowait()
        except: pass
    
    # إعداد واجهة المستخدم
    root, subtitle_label, status_label, status_indicator = setup_subtitles_ui()
    
    # بدء مهمة النسخ
    transcription_thread = threading.Thread(target=transcribe_task, daemon=True)
    transcription_thread.start()
    
    # بدء تدفق الصوت مع إعدادات محسنة
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        device=device_id,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        latency="low",  # استخدام وضع اللاتنسي المنخفض للاستجابة الأسرع
        callback=audio_callback
    )
    
    try:
        stream.start()
        root.after(100, lambda: update_ui(root, subtitle_label, status_label, status_indicator))
        root.mainloop()
        
    except Exception as e:
        print(f"⚠ خطأ: {e}")
    finally:
        # تنظيف الموارد بشكل صحيح
        if 'stream' in locals():
            stream.stop()
            stream.close()
        
        keyboard.unhook_all()
        
        try:
            root.destroy()
        except:
            pass
        
        # حفظ الإعدادات
        config["last_device_id"] = device_id
        save_config()

# الدالة الرئيسية
def main():
    try:
        # تحميل الإعدادات
        global config
        config = load_config()
        
        # التحقق من الاستخدام الأول
        first_run = not os.path.exists(CONFIG_FILE) or not check_model_downloaded(config.get("model_size", "tiny"))
        
        if first_run:
            print("👋 مرحباً بك في برنامج الترجمة الصوتية المباشرة")
            create_model_selector()
        else:
            start_transcription()
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البرنامج بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        messagebox.showerror("خطأ غير متوقع", f"حدث خطأ غير متوقع:\n{str(e)}")

if __name__ == "__main__":
    main()