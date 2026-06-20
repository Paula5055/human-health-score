# human-health-score

# Human Health Score

> AI-powered workplace health assessment – a critical design object

## About

The Human Health Score is a satirical AI scanner that assesses employees
via facial recognition and assigns a wellness score from A to E –
inspired by the Nutri-Score system. The project was developed as part
of the AI seminar at Leuphana University Lüneburg and critically
examines algorithmic surveillance in the workplace.

**Developed by:** Paula Heide & Marco Andres Ruge Rebolledo

---

## How it works

1. USB camera captures the face
2. Two AI models analyze fatigue and stress
3. A rule-based component checks punctuality
4. Python calculates a weighted score A–E
5. Result is displayed in a custom score panel

---

## Score Logic

| Points | Score | Label |
|---|---|---|
| 0 | A | PEAK CONDITION |
| 1 | B | PERFORMING |
| 2 | C | IMPAIRED |
| 3 | D | AT RISK |
| 4 | E | CRITICAL |

**Point system:**
- Fatigue detected: +2 points
- Stress detected: +1 point
- Late scan (after 8:00): +1 point

---

## Consequences

| Score | Consequence |
|---|---|
| A / B | No action required |
| 3x C per week | Formal warning by HR |
| 2x D per week | Performance Optimization Program (costs deducted from salary) |
| 1x E per week | Intensive Performance Program (costs deducted from salary) |

---

## Installation

```bash
pip install opencv-python tensorflow tf-keras numpy
```

## Usage

```bash
python human_health_score.py
```

> **Note:** If the camera is not found, change `cv2.VideoCapture(1)` 
> to `cv2.VideoCapture(0)` in the script.

---

## Models

Trained with Google Teachable Machine:

| Model | Classes | Dataset |
|---|---|---|
| Fatigue | awake / fatigued | Kaggle: Fatigue Dataset (rihabkaci99) |
| Stress | no_stress / stress | Kaggle: FANE Dataset |

---

## Tech Stack

- Python
- OpenCV – camera, face detection
- TensorFlow / tf-keras – model loading
- Google Teachable Machine – model training
- NumPy – image processing

---

## Project Context

This project questions:
- Who defines what "healthy" or "productive" means?
- How quickly can caring technology turn into control?
- How do AI models reproduce bias through training data?

## Critical Reflection

The score looks objective – just a letter, A to E, calculated by an algorithm. No human judgment, no bias, just numbers. But in reality, every decision in this system was made by humans: we decided that fatigue counts double, that 8:00 is the punctuality cutoff, and the training data decided what a "fatigued" face looks like – based on specific skin tones, lighting conditions and facial structures.

A human manager saying "you look tired today" feels subjective. A machine giving you a D feels like a fact. The apparent objectivity of algorithmic systems is itself a form of power.