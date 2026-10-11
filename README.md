# Human Health Score

> AI-powered employee health assessment – a critical design object

## About

The Human Health Score is a satirical AI scanner that assesses employees
via facial recognition and assigns a wellness score from A to E –
inspired by the Nutri-Score. It is presented as a harmless desk object:
a plush plant with a visible webcam between its leaves.

Developed as part of the AI seminar at Leuphana University Lüneburg.

---

## How it works

1. USB camera detects the face
2. Two AI models analyze fatigue and stress
3. A rule checks punctuality
4. Points are converted into a score A–E
5. Result is shown in a Tkinter dashboard

| Points | Score | Label |
|---|---|---|
| 0 | A | PEAK CONDITION |
| 1 | B | PERFORMING |
| 2 | C | IMPAIRED |
| 3 | D | AT RISK |
| 4 | E | CRITICAL |

Fatigue +2 · Stress +1 · Late scan (from 8:00) +1

---

## Consequences

| Trigger | Program |
|---|---|
| 3x C per week | Wellness Support Program (notification by HR) |
| 2x D per week | Performance Optimization Program (costs deducted from salary) |
| 1x E per week | Intensive Performance Program (costs deducted from salary) |

---

## Installation & Usage

```bash
pip install opencv-python tensorflow tf-keras numpy
python human_health_score.py
```

If the camera is not found, change `cv2.VideoCapture(0)` to `1` or `2`.

---

## Build the Object

**You need:** a plush plant, a USB webcam and a laptop. No 3D-printed parts.

1. Fix the webcam between the leaves at face height – lens clearly visible
2. Route the cable behind the plant to the laptop
3. Place the laptop next to the plant to show the dashboard

**Before a demo:** plug in the webcam before starting, avoid backlight,
and delete `scan_log.csv` to reset the weekly counts.

---

## Models & Tech

Models trained with Google Teachable Machine:

| Model | Classes | Dataset |
|---|---|---|
| Fatigue | awake / fatigued | Kaggle: Fatigue Dataset (rihabkaci99) |
| Stress | no_stress / stress | Kaggle: FANE Dataset + FER2013 |

Tech: Python, OpenCV, TensorFlow / tf-keras, Tkinter, NumPy

---

## Critical Reflection

The score looks objective – just a letter, calculated by an algorithm.
But every decision was made by humans: we decided that fatigue counts
double and that 8:00 is the cutoff, and the training data decided what
a "fatigued" face looks like.

A manager saying "you look tired today" feels subjective.
A machine giving you a D feels like a fact.

This project makes that illusion visible.

---

**Paula Heide & Marco Andres Ruge Rebolledo**
Creative Space für Künstliche Intelligenz · Leuphana Universität Lüneburg · 2026 