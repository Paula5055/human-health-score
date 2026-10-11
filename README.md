# Human Health Score

> AI-based health assessment in the workplace – an Object for Debate

## About

The Human Health Score is an AI-based assessment system that evaluates the
health of employees based on external facial features. A camera captures the
face, two AI models analyse visible signs of fatigue and stress, and the time
of the scan is included as punctuality. The result is a score from A to E,
modelled on the Nutri-Score for food.

The system is based on the assumption that productivity is closely linked to
the physical condition of employees. Poor scores lead to automatic enrolment
in company programmes. The system communicates this as care – "We care about
you" – while the costs are borne by the employees themselves.

All texts in the system are in English so that it resembles real corporate
software.

The physical object is a plush plant with a visible webcam between its leaves.
The analysis runs on a laptop next to it.

---

## How It Works

1. The webcam captures the face, OpenCV detects it
2. Two AI models trained with Google Teachable Machine detect fatigue and stress
3. A rule checks punctuality: scans from 8:00 count as late
4. The points are converted into a score from A to E
5. The result is displayed in a dashboard and logged with the time of the scan

| Component | Points |
|---|---|
| Fatigue | +2 |
| Stress | +1 |
| Late scan (from 8:00) | +1 |

| Points | Score | Label |
|---|---|---|
| 0 | A | PEAK CONDITION |
| 1 | B | PERFORMING |
| 2 | C | IMPAIRED |
| 3 | D | AT RISK |
| 4 | E | CRITICAL |

---

## Consequences

Scores are counted per week. Depending on the result, employees are
automatically enrolled in the following programmes:

| Trigger | Programme | Content |
|---|---|---|
| 3× C per week | Wellness Support Program | Notification by HR |
| 2× D per week | Performance Optimization Program | Stress management seminar twice a month, monthly nutrition coaching |
| 1× E per week | Intensive Performance Program | Weekly Saturday seminar, weekly nutrition and individual coaching, reporting to HR and supervisor |

The costs of all programmes are deducted from the employee's salary.

---

## Installation and Usage

```bash
pip install opencv-python tensorflow tf-keras numpy
python human_health_score.py
```

If the camera is not detected, change `cv2.VideoCapture(0)` to `1` or `2`.

---

## Building the Object

**Materials:** plush plant, USB webcam, laptop. No 3D-printed parts are required.

1. Attach the webcam between the leaves at face height, with the lens visible
2. Route the cable behind the plant to the laptop
3. Place the laptop next to the plant to display the dashboard

**Before a demonstration:** connect the webcam before starting the script,
avoid backlighting and delete `scan_log.csv` to reset the weekly count.

---

## AI Models and Technologies

| AI Model | Dataset | Classes |
|---|---|---|
| Fatigue | Kaggle: Fatigue Dataset (rihabkaci99) | Non Fatigue / Fatigue – 500 images each |
| Stress | Kaggle: FANE Dataset (furcifer) | angry + disgust → stress, neutral → no stress – 500 images each |

Technologies: Python, OpenCV, TensorFlow / tf-keras, Tkinter, Google Teachable Machine

Data protection: no images are stored – only the score and the time of the scan.

---

## Critical Reflection

The Human Health Score shows how a wellness technology can be marketed as a
caring measure while creating new forms of control and of standardising bodies.

The AI models do not measure health. They measure deviation from a "normal"
state defined by the training data. What the AI system reads as "exhausted"
may be genetic, cultural or simply individual. The weighting of the points and
the 8:00 cutoff are also human decisions, even though the score appears
objective.

The project is intended to prompt discussion about how quickly technologies
that appear caring can turn into control, social pressure and personal
consequences – and at what point the private lives of employees become the
employer's concern.

---

**Paula Heide & Marco Andres Ruge Rebolledo**
Creative Space für Künstliche Intelligenz · Leuphana Universität Lüneburg · 2026