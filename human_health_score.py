import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import cv2
import numpy as np
import datetime
import csv
import tkinter as tk
import threading
import math
import time
import tensorflow as tf

# ─────────────────────────────────────────────
# GLOBAL VARIABLES
# ─────────────────────────────────────────────

fatigue_model = None
fatigue_labels = None
stress_model = None
stress_labels = None
models_loaded = False
cap = None

# ─────────────────────────────────────────────
# LOADING SCREEN WITH SPINNER
# ─────────────────────────────────────────────

def load_models_thread():
    global fatigue_model, fatigue_labels, stress_model, stress_labels, models_loaded, cap
    from tf_keras.models import load_model

    fatigue_model = load_model('models/fatigue_model/keras_model.h5', compile=False)
    fatigue_labels = open('models/fatigue_model/labels.txt').read().splitlines()

    stress_model = load_model('models/stress_model/keras_model.h5', compile=False)
    stress_labels = open('models/stress_model/labels.txt').read().splitlines()

    # Warmup
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    fatigue_model.predict(dummy, verbose=0)
    stress_model.predict(dummy, verbose=0)

    # Open camera in background
    cap = cv2.VideoCapture(0)

    models_loaded = True

def show_loading_window():
    root = tk.Tk()
    root.title("Human Health Score")
    root.configure(bg="#111111")
    root.geometry("600x400")
    root.resizable(False, False)

    tk.Label(root, text="HUMAN HEALTH SCORE",
             bg="#111111", fg="white",
             font=("Arial", 22, "bold")).pack(pady=(50, 5))

    tk.Label(root, text="AI-powered employee health assessment",
             bg="#111111", fg="#666666",
             font=("Arial", 11)).pack(pady=(0, 30))

    canvas = tk.Canvas(root, width=80, height=80,
                       bg="#111111", highlightthickness=0)
    canvas.pack()

    messages = [
        "Initializing system...",
        "Calibrating facial recognition... <3",
        "Loading wellness protocols...",
        "We care about you. Please stand by...",
    ]
    msg_label = tk.Label(root, text=messages[0],
                         bg="#111111", fg="#aaaaaa",
                         font=("Arial", 12))
    msg_label.pack(pady=20)

    angle = [0]
    msg_index = [0]
    msg_timer = [0]

    def animate():
        if models_loaded:
            msg_label.config(text="Please look into the camera.")
            root.after(1500, root.destroy)
            return

        canvas.delete("all")
        a = angle[0]
        for i in range(8):
            theta = math.radians(a + i * 45)
            x1 = 40 + 28 * math.cos(theta)
            y1 = 40 + 28 * math.sin(theta)
            x2 = 40 + 36 * math.cos(theta)
            y2 = 40 + 36 * math.sin(theta)
            alpha = int(255 * (i + 1) / 8)
            color = f"#{alpha:02x}{alpha:02x}{alpha:02x}"
            canvas.create_line(x1, y1, x2, y2,
                               fill=color, width=3, capstyle="round")

        angle[0] = (angle[0] + 10) % 360

        msg_timer[0] += 1
        if msg_timer[0] >= 150 and msg_index[0] < len(messages) - 1:
            msg_index[0] += 1
            msg_label.config(text=messages[msg_index[0]])
            msg_timer[0] = 0

        root.after(30, animate)

    animate()
    root.mainloop()

# Start model loading in background
thread = threading.Thread(target=load_models_thread, daemon=True)
thread.start()

# Show loading window (runs in main thread)
show_loading_window()

print(f"Models loaded: {models_loaded}")

# Wait until models are fully loaded
while not models_loaded:
    time.sleep(0.1)

# ─────────────────────────────────────────────
# tf.function
# ─────────────────────────────────────────────

@tf.function
def predict_fatigue(img):
    return fatigue_model(img)

@tf.function
def predict_stress(img):
    return stress_model(img)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

WORK_START_HOUR = 8
LOG_FILE = 'scan_log.csv'

scanned = False

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def check_punctuality():
    now = datetime.datetime.now()
    if now.hour >= WORK_START_HOUR:
        return 1, f"Late (scanned at {now.strftime('%H:%M')})"
    return 0, f"On time (scanned at {now.strftime('%H:%M')})"

def get_week_number(date):
    return date.isocalendar()[1]

def log_scan(score):
    now = datetime.datetime.now()
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now.strftime('%Y-%m-%d %H:%M'), score])

def count_scores_this_week():
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    now = datetime.datetime.now()
    current_week = get_week_number(now)
    current_year = now.year
    if not os.path.exists(LOG_FILE):
        return counts
    with open(LOG_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                entry_date = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M')
                if (get_week_number(entry_date) == current_week and
                        entry_date.year == current_year):
                    s = row[1].strip()
                    if s in counts:
                        counts[s] += 1
            except:
                continue
    return counts

def get_warning_text(score, counts):
    if score == 'C':
        return f"This week: {counts['C']}x Score C  (action triggered at 3x)"
    elif score == 'D':
        return f"This week: {counts['D']}x Score D  (action triggered at 2x)"
    elif score == 'E':
        return f"This week: {counts['E']}x Score E  (action triggered at 1x)"
    return ""

# ─────────────────────────────────────────────
# TKINTER RESULT WINDOW
# ─────────────────────────────────────────────

SCORE_COLORS = {
    "A": "#1a8c3a",
    "B": "#4db84d",
    "C": "#e6b800",
    "D": "#e67300",
    "E": "#cc1a1a",
}

SCORE_DESCRIPTIONS = {
    "A": "Fully operational - no action required.",
    "B": "Minor indicators detected - self-monitoring advised.",
    "C": "Reduced capacity identified - formal warning issued.",
    "D": "Significant strain detected - Stage 1 activated.",
    "E": "Immediate intervention required - Stage 2 activated.",
}

def show_result(score, label, fatigue_class, stress_class, punctuality_label, counts):
    color = SCORE_COLORS[score]
    warning_text = get_warning_text(score, counts)

    root = tk.Tk()
    root.title("Human Health Score - Result")
    root.configure(bg="#111111")
    root.geometry("720x920")
    root.resizable(False, False)

    # Header
    header = tk.Frame(root, bg="#222222", height=70)
    header.pack(fill="x")
    tk.Label(header, text="HUMAN HEALTH SCORE",
             bg="#222222", fg="white",
             font=("Arial", 20, "bold")).pack(pady=18)

    # Score box
    score_frame = tk.Frame(root, bg="#111111")
    score_frame.pack(fill="x", padx=30, pady=(20, 0))

    tk.Label(score_frame, text=score,
             bg=color, fg="white",
             font=("Arial", 72, "bold"),
             width=2, height=1).pack(side="left")

    label_frame = tk.Frame(score_frame, bg="#111111")
    label_frame.pack(side="left", padx=20)

    tk.Label(label_frame, text=label,
             bg="#111111", fg=color,
             font=("Arial", 22, "bold")).pack(anchor="w")
    tk.Label(label_frame, text=SCORE_DESCRIPTIONS[score],
             bg="#111111", fg="#999999",
             font=("Arial", 11)).pack(anchor="w", pady=(6, 0))

    # Divider
    tk.Frame(root, bg="#333333", height=1).pack(fill="x", padx=30, pady=15)

    # AI result
    info_frame = tk.Frame(root, bg="#111111")
    info_frame.pack(fill="x", padx=30)

    tk.Label(info_frame,
             text=f"Fatigue: {fatigue_class}     |     Stress: {stress_class}",
             bg="#111111", fg="#888888",
             font=("Arial", 11)).pack(anchor="w")
    tk.Label(info_frame,
             text=f"Punctuality: {punctuality_label}",
             bg="#111111", fg="#888888",
             font=("Arial", 11)).pack(anchor="w", pady=(4, 0))

    if warning_text:
        tk.Label(info_frame, text=warning_text,
                 bg="#111111", fg="#e6b800",
                 font=("Arial", 10)).pack(anchor="w", pady=(4, 0))

    # Divider
    tk.Frame(root, bg="#333333", height=1).pack(fill="x", padx=30, pady=15)

    # We care about you
    tk.Label(root, text="<3  We care about you.",
             bg="#111111", fg=color,
             font=("Arial", 16, "bold")).pack(anchor="w", padx=30, pady=(0, 15))

    # Content
    content_frame = tk.Frame(root, bg="#111111")
    content_frame.pack(fill="x", padx=30)

    def add_line(text, size=12, fg="white", bold=False):
        style = "bold" if bold else "normal"
        tk.Label(content_frame, text=text,
                 bg="#111111", fg=fg,
                 font=("Arial", size, style),
                 justify="left", anchor="w",
                 wraplength=640).pack(anchor="w", pady=2)

    def add_spacer():
        tk.Label(content_frame, text="", bg="#111111").pack()

    if score == "A":
        add_line("Your health indicators are within optimal range.")
        add_line("No action required at this time.")
        add_spacer()
        add_line("Keep it up!", fg=color, bold=True)

    elif score == "B":
        add_line("Minor indicators have been detected.")
        add_line("We recommend self-monitoring over the next few days.")
        add_spacer()
        add_line("No action required at this time.", fg="#888888")

    elif score == "C":
        add_line("We have noticed some concerning indicators.")
        add_line("You will be enrolled in our Wellness Support Program:")
        add_spacer()
        add_line("-  Written notification by HR department")
        add_spacer()
        add_line("You will receive an email from your manager about the details.", fg="#888888")

    elif score == "D":
        add_line("You will be enrolled in our Performance Optimization Program:")
        add_spacer()
        add_line("-  Stress management seminar (2x per month)")
        add_line("-  Nutrition coaching (monthly, individual session)")
        add_spacer()
        add_line("You will receive an email from your manager about the details.", fg="#888888")
        add_spacer()
        add_line("Please note: All program costs will be deducted from your salary.",
                 size=10, fg="#666666")

    elif score == "E":
        add_line("You will be immediately enrolled in our Intensive Performance Program:")
        add_spacer()
        add_line("-  Stress management seminar (weekly, all-day Saturdays)")
        add_line("-  Nutrition coaching (weekly, individual session)")
        add_line("-  Individual coaching with occupational health advisor")
        add_line("-  Weekly reporting to HR and direct supervisor")
        add_spacer()
        add_line("You will receive an email from your manager about the details.", fg="#888888")
        add_spacer()
        add_line("Please note: All program costs will be deducted from your salary.",
                 size=10, fg="#666666")

    # Footer
    footer = tk.Frame(root, bg="#222222", height=50)
    footer.pack(fill="x", side="bottom")
    tk.Label(footer, text="Scan complete. Results logged.",
             bg="#222222", fg="#666666",
             font=("Arial", 10)).pack(pady=15)

    root.mainloop()


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Human Health Score", frame)
    cv2.waitKey(30)

    if len(faces) > 0 and not scanned:
        (x, y, w, h) = faces[0]

        face_img = frame[y:y+h, x:x+w]
        img = cv2.resize(face_img, (224, 224))
        img = np.array(img, dtype=np.float32) / 127.5 - 1
        img = np.expand_dims(img, axis=0)

        fatigue_pred = predict_fatigue(img).numpy()
        fatigue_class = fatigue_labels[np.argmax(fatigue_pred)].split(' ', 1)[1]
        fatigue_confidence = float(np.max(fatigue_pred))

        stress_pred = predict_stress(img).numpy()
        stress_class = stress_labels[np.argmax(stress_pred)].split(' ', 1)[1]
        stress_confidence = float(np.max(stress_pred))

        punctuality_points, punctuality_label = check_punctuality()

        # Score calculation
        fatigue_points = 2 if ('Fatigue' in fatigue_class and fatigue_confidence > 0.65) else 0
        stress_points = 1 if ('stress' in stress_class and 'no' not in stress_class and stress_confidence > 0.65) else 0
        total = fatigue_points + stress_points + punctuality_points

        print(f"RAW fatigue: '{fatigue_class}' ({fatigue_confidence:.0%}) | RAW stress: '{stress_class}' ({stress_confidence:.0%}) | Punctuality: '{punctuality_label}'")
        print(f"Points: fatigue={fatigue_points} | stress={stress_points} | punctuality={punctuality_points} | total={total}")

        if total == 0:
            score, label = "A", "PEAK CONDITION"
        elif total == 1:
            score, label = "B", "PERFORMING"
        elif total == 2:
            score, label = "C", "IMPAIRED"
        elif total == 3:
            score, label = "D", "AT RISK"
        else:
            score, label = "E", "CRITICAL"

        print(f"Score: {score} - {label}")

        log_scan(score)
        counts = count_scores_this_week()
        print(f"Weekly counts: {counts}")

        # Show scan complete
        scan_frame = frame.copy()
        cv2.putText(scan_frame, "SCAN COMPLETE", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Human Health Score", scan_frame)
        cv2.waitKey(1)

        for _ in range(30):
            cv2.waitKey(100)

        cv2.destroyWindow("Human Health Score")
        cv2.waitKey(1)

        show_result(score, label, fatigue_class, stress_class, punctuality_label, counts)

        scanned = True
        break

cap.release()
cv2.destroyAllWindows()