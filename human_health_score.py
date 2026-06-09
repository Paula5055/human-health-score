import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import cv2
import numpy as np
import datetime
import csv
from tf_keras.models import load_model

# Modelle laden
fatigue_model = load_model('models/fatigue_model/keras_model.h5', compile=False)
fatigue_labels = open('models/fatigue_model/labels.txt').read().splitlines()
stress_model = load_model('models/stress_model/keras_model.h5', compile=False)
stress_labels = open('models/stress_model/labels.txt').read().splitlines()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

WORK_START_HOUR = 8
LOG_FILE = 'scan_log.csv'

cap = cv2.VideoCapture(1)
scanned = False

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
        return f"This week: {counts['C']}x Score C (action triggered at 3x)"
    elif score == 'D':
        return f"This week: {counts['D']}x Score D (action triggered at 2x)"
    elif score == 'E':
        return f"This week: {counts['E']}x Score E (action triggered at 1x)"
    return ""

def show_result(score, label, fatigue_class, stress_class, punctuality_label, counts):
    panel = np.zeros((900, 700, 3), dtype=np.uint8)

    colors = {
        "A": (34, 139, 34),
        "B": (80, 200, 80),
        "C": (0, 200, 230),
        "D": (0, 140, 255),
        "E": (0, 0, 210)
    }
    color = colors[score]
    gray_col = (180, 180, 180)
    white = (255, 255, 255)
    yellow = (0, 215, 255)

    # Header
    cv2.rectangle(panel, (0, 0), (700, 80), (30, 30, 30), -1)
    cv2.putText(panel, "HUMAN HEALTH SCORE", (30, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, white, 2)

    # Score Buchstabe
    cv2.rectangle(panel, (30, 100), (160, 230), color, -1)
    cv2.putText(panel, score, (62, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 4.0, white, 5)

    # Label
    cv2.putText(panel, label, (185, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    descriptions = {
        "A": "Fully operational - no action required.",
        "B": "Minor indicators detected - self-monitoring advised.",
        "C": "Reduced capacity identified - formal warning issued.",
        "D": "Significant strain detected - Stufe 1 activated.",
        "E": "Immediate intervention required - Stufe 2 activated."
    }
    cv2.putText(panel, descriptions[score], (185, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)

    # Trennlinie
    cv2.line(panel, (30, 250), (670, 250), (60, 60, 60), 1)

    # KI-Ergebnis
    cv2.putText(panel, f"Fatigue: {fatigue_class}   |   Stress: {stress_class}", (30, 278),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)
    cv2.putText(panel, f"Punctuality: {punctuality_label}", (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)

    # Wochenzähler
    warning_text = get_warning_text(score, counts)
    if warning_text:
        cv2.putText(panel, warning_text, (30, 318),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, yellow, 1)

    # Trennlinie
    cv2.line(panel, (30, 330), (670, 330), (60, 60, 60), 1)

    y = 360

    # <3 We care about you
    cv2.putText(panel, "<3  We care about you.", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    y += 45

    if score == "A":
        cv2.putText(panel, "Your health indicators are within optimal range.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 32
        cv2.putText(panel, "No action required at this time.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 42
        cv2.putText(panel, "Keep it up!", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    elif score == "B":
        cv2.putText(panel, "Minor indicators have been detected.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 32
        cv2.putText(panel, "We recommend self-monitoring over the next few days.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 42
        cv2.putText(panel, "No action required at this time.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, gray_col, 1)

    elif score == "C":
        cv2.putText(panel, "We have noticed some concerning indicators.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 32
        cv2.putText(panel, "You will be enrolled in our Wellness Support Program:", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 38
        cv2.putText(panel, "- Written notification by HR department", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1)
        y += 38
        cv2.putText(panel, "You will receive an email from your manager", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)
        y += 28
        cv2.putText(panel, "about the details.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)

    elif score == "D":
        cv2.putText(panel, "You will be enrolled in our", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 30
        cv2.putText(panel, "Performance Optimization Program:", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 38
        items = [
            "- Stress management seminar (2x per month)",
            "- Nutrition coaching (monthly, individual session)",
        ]
        for item in items:
            cv2.putText(panel, item, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, white, 1)
            y += 30
        y += 10
        cv2.putText(panel, "You will receive an email from your manager", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)
        y += 28
        cv2.putText(panel, "about the details.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)
        y += 35
        cv2.putText(panel, "Please note: All program costs will be deducted", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)
        y += 24
        cv2.putText(panel, "from your salary.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)

    elif score == "E":
        cv2.putText(panel, "You will be immediately enrolled in our", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 30
        cv2.putText(panel, "Intensive Performance Program:", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
        y += 38
        items = [
            "- Stress management seminar (weekly, all-day Saturdays)",
            "- Nutrition coaching (weekly, individual session)",
            "- Individual coaching with occupational health advisor",
            "- Weekly reporting to HR and direct supervisor",
        ]
        for item in items:
            cv2.putText(panel, item, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, white, 1)
            y += 28
        y += 10
        cv2.putText(panel, "You will receive an email from your manager", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)
        y += 28
        cv2.putText(panel, "about the details.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, gray_col, 1)
        y += 35
        cv2.putText(panel, "Please note: All program costs will be deducted", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)
        y += 24
        cv2.putText(panel, "from your salary.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, white, 1)

    # Footer
    cv2.rectangle(panel, (0, 840), (700, 900), (30, 30, 30), -1)
    cv2.putText(panel, "Scan complete. Results logged.", (30, 878),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, gray_col, 1)

    cv2.imshow("Human Health Score - Result", panel)
    cv2.waitKey(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Human Health Score", frame)
    cv2.waitKey(1)

    if len(faces) > 0 and not scanned:
        (x, y, w, h) = faces[0]

        face_img = frame[y:y+h, x:x+w]
        img = cv2.resize(face_img, (224, 224))
        img = np.array(img, dtype=np.float32) / 127.5 - 1
        img = np.expand_dims(img, axis=0)

        fatigue_pred = fatigue_model.predict(img, verbose=0)
        fatigue_class = fatigue_labels[np.argmax(fatigue_pred)].split(' ', 1)[1]

        stress_pred = stress_model.predict(img, verbose=0)
        stress_class = stress_labels[np.argmax(stress_pred)].split(' ', 1)[1]

        punctuality_points, punctuality_label = check_punctuality()

        fatigue_points = 2 if 'Fatigue' in fatigue_class else 0
        stress_points = 1 if 'stress' in stress_class and 'no' not in stress_class else 0
        total = fatigue_points + stress_points + punctuality_points

        print(f"RAW fatigue: '{fatigue_class}' | RAW stress: '{stress_class}' | Punctuality: '{punctuality_label}'")
        print(f"Punkte: fatigue={fatigue_points} | stress={stress_points} | punctuality={punctuality_points} | total={total}")

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

        # Scan loggen
        log_scan(score)
        counts = count_scores_this_week()
        print(f"Wochenzähler: {counts}")

        # Scan Complete anzeigen
        scan_frame = frame.copy()
        cv2.putText(scan_frame, "SCAN COMPLETE", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Human Health Score", scan_frame)
        cv2.waitKey(1)  # kurz rendern lassen

        # 3 Sekunden warten
        for _ in range(30):
            cv2.waitKey(100)

        cv2.destroyWindow("Human Health Score")
        cv2.waitKey(1)  # sicherstellen dass Fenster wirklich zu ist

        # Ergebnis-Fenster öffnen
        show_result(score, label, fatigue_class, stress_class, punctuality_label, counts)

        scanned = True
        break

cap.release()
cv2.destroyAllWindows()