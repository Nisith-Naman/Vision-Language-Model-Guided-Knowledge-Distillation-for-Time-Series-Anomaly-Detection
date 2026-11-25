"""
IE643 Viva Preparation Guide  --  PDF Generator
Comprehensive Q&A covering all viva questions in plain language.
Run: python report/generate_viva_pdf.py
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus.flowables import Flowable

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY      = HexColor('#0D1B2A')
BLUE      = HexColor('#1565C0')
LIGHT_BLUE = HexColor('#E3F2FD')
TEAL      = HexColor('#00695C')
TEAL_LIGHT = HexColor('#E0F2F1')
ORANGE    = HexColor('#E65100')
ORANGE_LIGHT = HexColor('#FFF3E0')
GRAY      = HexColor('#37474F')
LIGHT_GRAY = HexColor('#F5F5F5')
MID_GRAY  = HexColor('#78909C')
DARK      = HexColor('#1A1A1A')
WHITE     = white
GREEN     = HexColor('#2E7D32')
GREEN_LIGHT = HexColor('#E8F5E9')
RED       = HexColor('#B71C1C')
RED_LIGHT  = HexColor('#FFEBEE')
PURPLE    = HexColor('#4A148C')
PURPLE_LIGHT = HexColor('#F3E5F5')

PAGE_W, PAGE_H = A4
LMAR = RMAR = 1.8 * cm
TMAR = BMAR = 2.0 * cm


class SideBar(Flowable):
    """Colored left sidebar for Q&A boxes."""
    def __init__(self, color, width, height):
        super().__init__()
        self.bar_color = color
        self.bar_width = width
        self.bar_height = height

    def wrap(self, aw, ah):
        return self.bar_width, self.bar_height

    def draw(self):
        self.canv.setFillColor(self.bar_color)
        self.canv.rect(0, 0, self.bar_width, self.bar_height, fill=1, stroke=0)


def on_page(canvas, doc):
    w, h = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 1.1*cm, w, 1.1*cm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont('Helvetica-Bold', 8)
    canvas.drawString(LMAR, h - 0.72*cm, 'IE643 | VLM-KD Time-Series Anomaly Detection | Viva Preparation Guide')
    canvas.setFont('Helvetica', 7.5)
    canvas.drawRightString(w - RMAR, h - 0.72*cm, f'Page {doc.page}')
    canvas.setFillColor(BLUE)
    canvas.rect(0, 0, w, 0.5*cm, fill=1, stroke=0)
    canvas.restoreState()


def build_styles():
    def s(name, **kw):
        d = dict(fontName='Helvetica', fontSize=10, leading=15,
                 textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=4)
        d.update(kw)
        return ParagraphStyle(name=name, **d)

    return {
        'cover_title': s('ct', fontName='Helvetica-Bold', fontSize=22, leading=28,
                         alignment=TA_CENTER, textColor=NAVY, spaceAfter=6),
        'cover_sub': s('cs', fontName='Helvetica', fontSize=13, leading=18,
                       alignment=TA_CENTER, textColor=BLUE, spaceAfter=4),
        'cover_info': s('ci', fontName='Helvetica', fontSize=10,
                        alignment=TA_CENTER, textColor=MID_GRAY, spaceAfter=3),
        'chapter': s('ch', fontName='Helvetica-Bold', fontSize=16, leading=22,
                     textColor=WHITE, alignment=TA_LEFT, spaceAfter=0,
                     leftIndent=0.4*cm),
        'h1': s('h1', fontName='Helvetica-Bold', fontSize=13, leading=18,
                textColor=NAVY, spaceBefore=10, spaceAfter=4,
                borderPad=0.2*cm),
        'h2': s('h2', fontName='Helvetica-Bold', fontSize=11, leading=15,
                textColor=BLUE, spaceBefore=6, spaceAfter=3),
        'q': s('q', fontName='Helvetica-Bold', fontSize=10, leading=14,
               textColor=NAVY, spaceAfter=2, leftIndent=0.3*cm),
        'a': s('a', fontName='Helvetica', fontSize=10, leading=15,
               textColor=DARK, alignment=TA_JUSTIFY, spaceAfter=3,
               leftIndent=0.3*cm),
        'code': s('code', fontName='Courier', fontSize=8.5, leading=12,
                  textColor=TEAL, leftIndent=0.5*cm, spaceAfter=3),
        'formula': s('formula', fontName='Courier', fontSize=9, leading=13,
                     textColor=PURPLE, leftIndent=0.5*cm, alignment=TA_CENTER,
                     spaceAfter=3),
        'note': s('note', fontName='Helvetica-Oblique', fontSize=9.5, leading=14,
                  textColor=ORANGE, leftIndent=0.3*cm, spaceAfter=4),
        'bullet': s('bullet', fontName='Helvetica', fontSize=10, leading=14,
                    leftIndent=0.7*cm, spaceAfter=2),
        'imp': s('imp', fontName='Helvetica-Bold', fontSize=10, leading=14,
                 textColor=GREEN, leftIndent=0.3*cm, spaceAfter=3),
        'warn': s('warn', fontName='Helvetica-Bold', fontSize=10, leading=14,
                  textColor=RED, leftIndent=0.3*cm, spaceAfter=3),
        'body': s('body'),
        'caption': s('cap', fontName='Helvetica-Oblique', fontSize=8.5,
                     textColor=MID_GRAY, alignment=TA_CENTER, spaceAfter=4),
        'toc': s('toc', fontName='Helvetica', fontSize=10, leading=16,
                 leftIndent=0.5*cm, spaceAfter=2),
        'toc_h': s('toch', fontName='Helvetica-Bold', fontSize=11, leading=18,
                   leftIndent=0, spaceAfter=2, textColor=NAVY),
    }


ST = build_styles()


def sp(n=1): return Spacer(1, n * 0.3 * cm)
def P(t, style): return Paragraph(t, ST[style])
def HR(color=BLUE, t=0.8): return HRFlowable(width='100%', thickness=t, color=color, spaceAfter=4)


def chapter_header(letter, title, color=NAVY):
    data = [[Paragraph(f'<b>Section {letter}: {title}</b>', ST['chapter'])]]
    t = Table(data, colWidths=[PAGE_W - LMAR - RMAR])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), color),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [4,4,4,4]),
    ]))
    return t


def qa(question, *answers, q_num='', tip=None, code=None, formula=None, warn=None, imp=None):
    """Build a question-answer block."""
    items = []
    q_text = f'Q{q_num}: {question}' if q_num else f'Q: {question}'
    items.append(P(q_text, 'q'))
    for ans in answers:
        items.append(P(ans, 'a'))
    if formula:
        items.append(P(formula, 'formula'))
    if code:
        items.append(P(code, 'code'))
    if tip:
        items.append(P(f'<i>Tip: {tip}</i>', 'note'))
    if imp:
        items.append(P(f'Key Point: {imp}', 'imp'))
    if warn:
        items.append(P(f'Watch Out: {warn}', 'warn'))
    items.append(sp(0.5))
    return items


def build_table(data, col_widths, head_color=NAVY):
    rows = []
    for i, row in enumerate(data):
        cells = []
        for j, cell in enumerate(row):
            if i == 0:
                cells.append(P(f'<b>{cell}</b>',
                    'a' if True else 'a'))
            else:
                cells.append(P(str(cell), 'a'))
        rows.append(cells)
    ts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), head_color),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
        ('GRID', (0,0), (-1,-1), 0.3, MID_GRAY),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ])
    t = Table(rows, colWidths=col_widths)
    t.setStyle(ts)
    return t


# ═══════════════════════════════════════════════════════════════════════════════
# CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

def build_cover(story):
    story.append(sp(5))
    data = [[P('<b>IE643  --  Deep Learning for Computer Vision</b>', 'cover_sub')]]
    t = Table(data, colWidths=[PAGE_W - LMAR - RMAR])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), LIGHT_BLUE),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(t)
    story.append(sp(2))
    story.append(P('VLM Teacher - RNN Student Knowledge Distillation<br/>for Time-Series Anomaly Detection', 'cover_title'))
    story.append(sp(1))
    story.append(HR(BLUE, 2))
    story.append(sp(1))
    story.append(P('COMPLETE VIVA PREPARATION GUIDE', 'cover_sub'))
    story.append(P('Every Question Answered  --  From Basic Concepts to Mathematical Details', 'cover_info'))
    story.append(sp(2))

    info_data = [
        ['Course', 'IE 643  --  Deep Learning for Computer Vision'],
        ['Project', 'VLM-Guided Knowledge Distillation for Time-Series Anomaly Detection'],
        ['Framework', 'Frozen CLIP Teacher  +  GRU Student  +  Knowledge Distillation'],
        ['Interface', '7-Page Interactive Streamlit Dashboard'],
        ['Tests', 'Unit-test suite present  --  run before claiming pass status'],
        ['Code', 'Fully Implemented in PyTorch + Streamlit'],
    ]
    t2 = Table([[P(f'<b>{r}</b>', 'a'), P(v, 'a')] for r, v in info_data],
               colWidths=[4.5*cm, 12*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), LIGHT_BLUE),
        ('GRID', (0,0), (-1,-1), 0.3, BLUE),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t2)
    story.append(sp(3))
    story.append(P('<i>This guide answers every viva question in plain language with actual code references.<br/>'
                   'Read it as if a senior is asking you  --  you should be able to explain every detail.</i>', 'cover_info'))
    story.append(PageBreak())


def build_quick_reference(story):
    story.append(chapter_header('0', 'Quick Reference  --  The Whole Project in One Page', TEAL))
    story.append(sp())
    story.append(P('<b>The Big Picture in Simple English:</b>', 'h2'))
    story.append(P(
        'You have sensor data (like a heart-rate monitor, oil pressure gauge, or credit card transaction log) '
        'that mostly behaves normally but sometimes goes wrong (fraud, equipment failure, medical event). '
        'Your job is to <b>automatically find those wrong moments</b>. '
        'The challenge is that wrong moments are rare (maybe 1 in 500 readings), '
        'and you don\'t always have someone to tell you exactly when they occurred.', 'a'))
    story.append(sp(0.5))
    story.append(P('<b>The Solution  --  Three-Step Approach:</b>', 'h2'))
    steps = [
        ('<b>Step 1  --  The Smart Teacher (CLIP VLM):</b>',
         'Take each chunk (window) of sensor data, draw it as a picture (heatmap), '
         'show it to a powerful AI called CLIP (which understands images and text), '
         'and ask it: "Does this look normal or anomalous?" CLIP gives a score. '
         'This costs a lot of computing time and cannot run on small devices.'),
        ('<b>Step 2  --  The Fast Student (GRU):</b>',
         'Train a small, fast AI called a GRU (a type of recurrent neural network) '
         'to mimic the teacher\'s scores while also learning from actual known anomaly labels. '
         'The student learns: (a) which windows are anomalous, (b) which exact time steps within each window are anomalous.'),
        ('<b>Step 3  --  Inference:</b>',
         'At test time, throw away the teacher entirely. '
         'Use only the tiny, fast student GRU to score new sensor data in real time. '
         'It applies a threshold and marks anomalous intervals.'),
    ]
    for title, body in steps:
        story.append(P(title, 'q'))
        story.append(P(body, 'a'))
    story.append(sp())
    story.append(HR())

    ref_data = [
        ['Component', 'What it is', 'File', 'Key Function'],
        ['Data Loader', 'Reads CSV, validates schema', 'src/data/loader.py', 'load_csv()'],
        ['Preprocessor', 'Sort, impute, scale, window', 'src/data/preprocessing.py', 'make_windows()'],
        ['Image Generator', 'Window -> 224x224 heatmap', 'src/teacher/image_gen.py', 'window_to_image()'],
        ['CLIP Teacher', 'Scores heatmap vs text prompts', 'src/teacher/clip_teacher.py', 'score_batch()'],
        ['GRU Student', 'Dual-head RNN model', 'src/student/model.py', 'GRUStudent.forward()'],
        ['Loss Functions', 'KD + Localization + Window CE', 'src/student/losses.py', 'TotalLoss.forward()'],
        ['Trainer', 'Training loop, early stopping', 'src/training/trainer.py', 'Trainer.fit()'],
        ['Metrics', 'AUPRC, F1, ROC-AUC, IoU', 'src/evaluation/metrics.py', 'compute_window_metrics()'],
        ['Threshold Tuner', 'Picks best val threshold', 'src/evaluation/threshold.py', 'tune_threshold()'],
        ['Localizer', 'IoU, Event-F1, Point-Adj', 'src/evaluation/localization.py', 'mean_iou(), event_f1()'],
        ['Dashboard', '7-page Streamlit app', 'app.py + src/ui/', 'page*.show()'],
    ]
    story.append(build_table(ref_data, [3*cm, 4*cm, 5*cm, 4.3*cm]))
    story.append(P('Implementation Map  --  Every component, file, and entry-point function.', 'caption'))
    story.append(PageBreak())


def section_A(story):
    story.append(chapter_header('A', 'Understanding the Project', NAVY))
    story.append(sp())

    items = []
    items += qa(
        'What is the exact problem this project is trying to solve?',
        'We are solving <b>multivariate time-series anomaly detection and localization</b>. '
        'Given a CSV file of sensor readings (e.g. accelerometers, pressure gauges, transaction amounts) '
        'recorded over time, the system must: (1) predict which time windows contain anomalies, '
        'and (2) pinpoint which exact time steps within those windows are anomalous. '
        'The key challenge is doing this with very few labeled examples and handling extreme class imbalance.',
        imp='The system outputs BOTH window-level alerts AND timestep-level anomaly masks  --  two tasks at once.',
        q_num='A1')

    items += qa(
        'What is time-series anomaly detection?',
        'A time series is a sequence of values measured at regular time intervals  --  like your heartbeat '
        'recorded every second, or a factory sensor sampled every minute. Anomaly detection means '
        'automatically identifying time points where the behavior is unusual  --  a spike, a sudden shift, '
        'a long dropout, or a gradual drift away from normal behavior.',
        'Think of it like a security camera alarm: it watches normally and only beeps when something strange happens.',
        q_num='A2')

    items += qa(
        'What is multivariate time-series data?',
        'Instead of ONE measurement per time step, you have MULTIPLE measurements simultaneously. '
        'For example, a Parkinson\'s patient wears sensors on their ankle, thigh, and trunk  --  '
        'that\'s 9 channels at once. The matrix is X with shape (T, F) where T = number of time steps '
        'and F = number of features (sensors). Relationships between sensors often carry anomaly information '
        'that no single sensor alone would reveal.',
        formula='X  in  R^(T x F)   where T = time steps, F = features (sensors)',
        q_num='A3')

    items += qa(
        'What is an anomaly, and how is it different from an ordinary outlier?',
        'An <b>outlier</b> is simply a data point far from the statistical average (e.g., someone very tall). '
        'An <b>anomaly</b> is a data point that is unusual GIVEN ITS CONTEXT  --  it may not be statistically '
        'extreme but is wrong for the situation. For example, a healthy heart rate of 120 bpm during exercise '
        'is normal, but the same 120 bpm at rest while sleeping is an anomaly. '
        'Context (the surrounding time series, the domain, the other sensors) is what makes something anomalous.',
        q_num='A4')

    items += qa(
        'What are point, contextual, and collective anomalies?',
        '<b>Point anomaly:</b> A single time step has an extreme value (e.g., a sudden pressure spike at t=500). '
        'Our system handles these via point-spike injection.',
        '<b>Contextual anomaly:</b> A value is normal in absolute terms but wrong in context '
        '(e.g., 20 degreesC outside temperature is fine in summer but very wrong if the data says it is January).',
        '<b>Collective anomaly:</b> Individual points look fine, but a GROUP of consecutive points forms an '
        'unusual pattern (e.g., a freezing-of-gait episode in Daphnet where each individual accelerometer '
        'reading is within range, but the PATTERN of reduced movement is the anomaly). '
        'Our window-based approach naturally captures collective anomalies.',
        imp='Our dual-head student detects all three types: step head for point anomalies, window head for collective.',
        q_num='A5')

    items += qa(
        'What are some real-world examples of anomalies in our selected datasets?',
        '<b>Credit Card Fraud:</b> A transaction of $5,000 in a foreign country immediately after a $50 '
        'grocery purchase at home. The "fraud" label (Class=1) is only 0.172% of 284,807 transactions.',
        '<b>Daphnet FoG:</b> A Parkinson\'s patient suddenly "freezes" mid-walk  --  their ankle acceleration '
        'drops close to zero while trunk/thigh keep moving. Annotation=2 marks these freeze events.',
        '<b>Air Quality:</b> A sensor returns -200 for hours (broken sensor) or suddenly spikes due to a '
        'traffic jam. No original labels  --  we inject synthetic anomalies for testing.',
        '<b>3W Oil Well:</b> A sudden pressure drop in the production tube (P-PDG sensor) indicating a '
        'slug flow or valve failure. Class code 1 - 8 indicates different failure modes.',
        '<b>MetroPT-3:</b> An air compressor on a metro train has a bearing failure on April 18, 2020. '
        'The Motor_current and Oil_temperature channels spike hours before the catastrophic failure.',
        q_num='A6')

    items += qa(
        'Why is anomaly detection difficult?',
        '<b>1. Extreme class imbalance:</b> Only 0.17% of credit card transactions are fraud. '
        'A model that always says "normal" gets 99.83% accuracy but is useless.',
        '<b>2. No labels available:</b> Labeling thousands of hours of sensor data is expensive. '
        'The Air Quality dataset has NO labels  --  we must inject synthetic ones.',
        '<b>3. Anomaly diversity:</b> Fraud looks different from FoG which looks different from compressor failure. '
        'A model trained on one domain rarely transfers directly.',
        '<b>4. Temporal context:</b> You cannot evaluate one time step in isolation  --  you need surrounding history.',
        '<b>5. False alarm costs:</b> In medical or industrial settings, too many false alarms cause alarm fatigue '
        'and operators start ignoring alerts, defeating the purpose.',
        q_num='A7')

    items += qa(
        'Why are anomalies usually a class-imbalance problem?',
        'By definition, anomalies are RARE events. If they happened frequently, they would become "normal." '
        'Credit card fraud is 0.17% of transactions. Equipment failures happen maybe once every few months '
        'of continuous operation. This means the training data has 99+ examples of "normal" for every 1 '
        'example of "anomaly." Standard models learn to always predict "normal" and still get high accuracy. '
        'Our solution: use <b>pos_weight=10.0</b> in the BCE loss  --  anomalous time steps count 10x more '
        'toward the gradient than normal time steps, forcing the model to take them seriously.',
        code='# In src/student/losses.py - LocalizationLoss\nF.binary_cross_entropy_with_logits(\n    step_logits, y_true, pos_weight=torch.tensor([10.0])\n)',
        q_num='A8')

    items += qa(
        'What is the difference between anomaly detection and anomaly localization?',
        '<b>Detection:</b> Answering "Is there an anomaly somewhere in this 128-step window?" '
        '-> window-level binary classification. Output: YES/NO.',
        '<b>Localization:</b> Answering "WHICH exact time steps within this window are anomalous?" '
        '-> step-level binary mask. Output: [0,0,0,1,1,1,0,0,...] one label per time step.',
        'Our GRU student has TWO output heads: the window head for detection, the step head for localization. '
        'This is what makes our system useful in practice  --  an operator can see exactly when the anomaly started.',
        imp='The student is the ONLY model with step-level localization ability. The teacher only scores whole windows.',
        q_num='A9')

    items += qa(
        'What is the expected input and final output of our complete system?',
        '<b>Input:</b> A CSV file containing sensor readings over time, with a timestamp column, '
        'one or more numeric feature columns, and optionally a binary label column.',
        '<b>Final output after inference:</b> '
        '(1) A timeline plot of anomaly scores for every time step, '
        '(2) A list of detected anomaly intervals with start/end timestamps, '
        '(3) Evaluation metrics (F1, PR-AUC, IoU) if labels are available, '
        '(4) Downloadable CSV of predictions, '
        '(5) Optional PDF summary report.',
        code='# Page 7 of Streamlit dashboard: src/ui/page7_inference.py',
        q_num='A10')

    items += qa(
        'What is the central research question of our project?',
        'Can a FROZEN, general-purpose vision-language model (CLIP)  --  which was never trained on sensor '
        'data  --  provide useful training signals (soft anomaly priors) that improve a lightweight recurrent '
        'neural network student beyond what hard ground-truth labels alone can achieve?',
        'In other words: <b>Does knowledge from CLIP\'s world model of "what looks normal vs. anomalous" '
        'transfer usefully to the sensor heatmap domain through knowledge distillation?</b>',
        imp='This is the core hypothesis. The ablation (student_kd vs student_baseline) directly tests it.',
        q_num='A11')

    items += qa(
        'What is the main motivation for combining a VLM with an RNN?',
        '<b>CLIP\'s strength:</b> Semantic, zero-shot reasoning about visual patterns. '
        'It can "understand" that a spikey, irregular pattern looks anomalous without being trained on it.',
        '<b>CLIP\'s weakness:</b> Cannot process time series directly, has no temporal memory, '
        'and returns a window-level score rather than locating anomalous time steps. Its runtime and size advantage have not been measured here.',
        '<b>GRU\'s strength:</b> Designed to be lightweight, captures temporal dynamics, '
        'learns exact step-level patterns from labeled data, produces fine-grained localization.',
        '<b>GRU\'s weakness:</b> Needs labeled training data; without it may miss semantic pattern understanding.',
        '<b>Combined:</b> CLIP teaches the GRU what "anomalous patterns" look like semantically, '
        'while labeled data teaches the GRU what exact sensor signatures correspond to each anomaly type.',
        imp='The teacher provides pattern intuition; the labels provide precise ground truth. Together they beat either alone.',
        q_num='A12')

    items += qa(
        'What are the project\'s assumptions and limitations?',
        '<b>Assumption 1:</b> Time-series windows rendered as heatmaps contain enough visual information '
        'for CLIP to distinguish anomalous from normal patterns  --  this is UNVERIFIED for all domains.',
        '<b>Assumption 2:</b> A single pair of text prompts (normal vs. anomaly) captures the relevant semantic distinction  --  '
        'this is domain-dependent and fragile.',
        '<b>Assumption 3:</b> Anomalies in val/test have similar visual character to CLIP\'s understanding  --  '
        'this is the core domain-gap risk.',
        '<b>Limitation 1:</b> No actual experimental results exist yet  --  reported numbers in the report are ESTIMATES. '
        'Real results need actual dataset downloads and training runs.',
        '<b>Limitation 2:</b> Air Quality labels are SYNTHETIC (injected by us)  --  results on this dataset do not '
        'demonstrate real anomaly detection, only ability to recover our own injected patterns.',
        '<b>Limitation 3:</b> The student is retrained separately for each dataset  --  it does not generalize '
        'across datasets with different feature counts without architectural changes.',
        warn='Do not present any performance score as measured until it comes from a saved held-out run.',
        q_num='A13')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_B(story):
    story.append(chapter_header('B', 'Dataset Collection and Understanding', TEAL))
    story.append(sp())
    items = []

    items += qa(
        'Which datasets are actually implemented in the codebase?',
        'Five dataset loaders are implemented as separate Python modules:',
        '1. <b>Credit Card Fraud</b>  --  src/data/datasets/creditcard.py',
        '2. <b>Daphnet Freezing of Gait</b>  --  src/data/datasets/daphnet.py',
        '3. <b>UCI Air Quality</b>  --  src/data/datasets/airquality.py',
        '4. <b>3W Oil Well Dataset</b>  --  src/data/datasets/oil3w.py',
        '5. <b>MetroPT-3 Air Compressor</b>  --  src/data/datasets/metropt.py',
        'All five are registered in src/data/loader.py via the load_dataset() dispatch function.',
        'In addition, SYNTHETIC test datasets are generated by scripts/generate_synthetic.py '
        '(univariate, multivariate, no-labels, with-missing variants).',
        q_num='B1')

    items += qa(
        'What is the source and purpose of each dataset?',
        '<b>Credit Card Fraud (Kaggle):</b> 284,807 European credit card transactions over 2 days in 2013. '
        'Features are anonymized PCA projections (V1-V28) + Amount. Purpose: binary fraud vs. normal.',
        '<b>Daphnet FoG (UCI #245):</b> 10 Parkinson\'s patients with tri-axial accelerometers (ankle, thigh, trunk) '
        'at 64 Hz. Purpose: detect Freezing of Gait episodes.',
        '<b>UCI Air Quality (UCI #360):</b> 9,358 hours of chemical sensor measurements in an Italian city. '
        'Purpose: environmental anomaly detection (NO ORIGINAL LABELS  --  we inject synthetic ones).',
        '<b>3W Oil Well (Petrobras GitHub):</b> ~1,984 instances of 8 event types in oil well operations. '
        'Features: P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, T-JUS-CKGL, QGL. '
        'Purpose: detect operational failures and transients.',
        '<b>MetroPT-3 (Zenodo):</b> ~6 months of metro compressor data with 15 channels (7 analog + 8 digital). '
        'Purpose: predict/detect 4 real compressor failure events.',
        q_num='B2')

    items += qa(
        'Which datasets contain ground-truth anomaly labels?',
        '<b>Real labels:</b> Credit Card (Class column), Daphnet (annotation=2), '
        '3W Oil Well (class != 0), MetroPT-3 (failure_windows defined in config).',
        '<b>NO original labels  --  synthetic injection required:</b> UCI Air Quality.',
        '<b>Event-level only (not timestep-level):</b> MetroPT-3 uses time windows '
        '(e.g., April 18, 2020 00:00 to 23:59) as anomaly regions  --  these are event-level boundaries, '
        'not individual timestep labels. We convert them to timestep labels by marking all steps inside '
        'the window as anomalous.',
        warn='Air Quality results are evaluated against OUR OWN injected patterns, not real anomalies. '
             'This must be disclosed clearly in the viva.',
        q_num='B3')

    items += qa(
        'Are different datasets trained independently or jointly?',
        'INDEPENDENTLY. Each dataset has its own GRU student model with an input_size matching '
        'that dataset\'s feature count (F). For Credit Card: F=29, for Daphnet: F=9, for MetroPT-3: F=15. '
        'You cannot use the Credit Card-trained GRU directly on Daphnet data because the first linear '
        'layer of the GRU has a different size.',
        'Joint/transfer learning across datasets would require architectural changes '
        '(e.g., a shared embedding layer) and is listed as a future novelty direction.',
        code='# In src/student/model.py - GRUStudent.__init__\nself.gru = nn.GRU(input_size=input_size, ...)  # input_size = F of THAT dataset',
        q_num='B4')

    items += qa(
        'What happens if a dataset has no label column?',
        'Three possibilities in the codebase:',
        '1. <b>Semi-synthetic injection:</b> inject_anomalies() adds synthetic point spikes, '
        'level shifts, drift, and noise bursts independently into train/val/test after splitting. '
        'Used for Air Quality (label_strategy="semi_synthetic").',
        '2. <b>Dropout labeling:</b> dropout_labels() marks long runs of sensor missing-code (-200) '
        'as anomalous. Used when sensors physically fail.',
        '3. <b>Teacher-only mode:</b> Run CLIP scores as anomaly predictions without any student training. '
        'No ground-truth evaluation is possible but inference still works.',
        'In the Streamlit dashboard (page 2), if no label column is selected, '
        'evaluation pages are disabled and only inference/visualization is shown.',
        q_num='B5')

    items += qa(
        'What dataset-specific preprocessing or assumptions are required?',
        '<b>Credit Card:</b> Time column is a cumulative elapsed seconds counter, not a real timestamp. '
        'Features V1-V28 are already PCA-normalized by Kaggle. Amount is NOT normalized by default '
        '(we apply StandardScaler on top). Window L=64, stride=32.',
        '<b>Daphnet:</b> annotation=0 rows (not part of experiment) are DROPPED and treated as gaps. '
        'The series is split at these gaps  --  windows never cross gap indices. '
        'Subject-based split prevents cross-patient data leakage. Window L=256 (4 seconds at 64Hz).',
        '<b>Air Quality:</b> Semicolon-separated CSV with decimal COMMAS (1,23 not 1.23). '
        'Missing values coded as -200. Columns with >50% missing are dropped entirely. '
        'Date and Time columns merged into a single timestamp.',
        '<b>3W Oil Well:</b> Transient classes (101-108) counted as anomalous '
        '(config: transient_is_anomaly=True). Forward-fill for short gaps. '
        'All-NaN columns get a boolean flag column added.',
        '<b>MetroPT-3:</b> Analog channels resampled to 1-minute median; digital channels max. '
        'Failure windows (4 events) converted to timestep labels.',
        q_num='B6')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_C(story):
    story.append(chapter_header('C', 'CSV Processing and Data Preparation', BLUE))
    story.append(sp())
    items = []

    items += qa(
        'How does the application load an uploaded CSV?',
        'The load_csv() function in src/data/loader.py is the entry point. It:',
        '1. Auto-detects the separator (comma, semicolon, or tab) by trying each.',
        '2. Reads the file with pandas pd.read_csv().',
        '3. Computes a DataQualityReport: counts rows, columns, missing values per column, '
        'duplicate rows, non-numeric columns, and shows the first 5 rows as a sample.',
        '4. Returns (DataFrame, DataQualityReport) so the UI can show the user a quality summary.',
        code='# src/data/loader.py\ndef load_csv(path: str) -> tuple[pd.DataFrame, DataQualityReport]:\n    # tries sep=",", sep=";", sep="\\t"\n    ...',
        q_num='C1')

    items += qa(
        'How is a time series represented mathematically?',
        'As a 2D matrix X of shape (T, F):',
        '- T = number of time steps (rows in your CSV)',
        '- F = number of features/sensors (numeric columns you selected)',
        'Each ROW is one time step (one moment in time).',
        'Each COLUMN is one sensor or feature measured at all time steps.',
        formula='X = [[x11, x12, ..., x1F],   <- time step 1\n     [x21, x22, ..., x2F],   <- time step 2\n     ...\n     [xT1, xT2, ..., xTF]]  <- time step T',
        q_num='C2')

    items += qa(
        'What is a sliding window, and why do we use it?',
        'A sliding window cuts the long time series into short, overlapping chunks of fixed length L. '
        'We move along the series by S steps at a time (stride S), creating many windows.',
        '<b>Why:</b> The GRU student needs a fixed-size input. Also, anomalies are local events '
        '(a freeze lasts 2 seconds, a pressure spike lasts 5 minutes). Windows let us analyze '
        'local segments without losing temporal context from neighboring steps.',
        '<b>Config values:</b> Default L=128 time steps, S=32 steps stride. '
        'Daphnet uses L=256 (4 seconds at 64Hz). Air Quality uses L=48 (48 hours).',
        formula='Window i starts at position: i * stride\nWindow i covers steps: [i*S, i*S + L)\nShape of windows array: (N, L, F) where N = number of windows',
        q_num='C3')

    items += qa(
        'How are window-level labels calculated?',
        'The "any" rule (config default): if ANY time step within the window has y_step=1, '
        'the entire window gets y_window=1.',
        'The "majority" rule (alternative): if MORE THAN 50% of time steps are anomalous, y_window=1.',
        'These window-level labels (y_window) are used for the window classification loss.',
        'The step-level labels (y_step, one per time step in each window) are used for the localization loss.',
        code='# src/data/preprocessing.py - make_windows()\nif anomaly_rule == "any":\n    y_win = (y_step_win.sum(axis=-1) > 0).astype(np.int8)',
        q_num='C4')

    items += qa(
        'How does the code prevent windows from crossing split boundaries?',
        'The make_windows() function accepts a gap_indices parameter  --  a sorted list of indices '
        'where gaps in the data exist (split boundaries, dropped annotation=0 rows in Daphnet, etc.).',
        'Before creating each window [start, start+L), the code checks whether any gap_index '
        'falls within [start, start+L). If yes, that window is SKIPPED.',
        'This is implemented using Python\'s bisect module for efficient range checks.',
        'This is the key leakage prevention mechanism  --  no window ever sees both training and '
        'validation data simultaneously.',
        code='# src/data/preprocessing.py\n# Uses bisect to check if any gap falls in [start, start+L)\nif bisect_left(gaps, start) != bisect_left(gaps, start + L):\n    continue  # skip this window',
        q_num='C5')

    items += qa(
        'How is normalization performed?',
        'StandardScaler (zero mean, unit variance) is the default (config: scaler="standard").',
        '<b>CRITICAL RULE:</b> The scaler is fitted on TRAINING windows only, '
        'then applied to validation and test windows using the SAME parameters.',
        'This prevents the model from "peeking" at future statistics.',
        'The scaler stats (mean, std per feature) are saved in the checkpoint and '
        'must be applied to any new data during inference.',
        formula='scaled_value = (original_value - train_mean) / train_std\n\nFor each feature f:\n  mean_f = average of X_train[:, f]\n  std_f  = std deviation of X_train[:, f]',
        imp='If you accidentally fit the scaler on all data (including test), your metrics are LIES.',
        q_num='C6')

    items += qa(
        'How are missing values handled?',
        'The fill_missing() function in preprocessing.py handles this.',
        'Default strategy: "interpolate"  --  linear interpolation between known values, '
        'then backward-fill for any remaining NaNs at the beginning.',
        'Other options: "ffill" (carry last value forward), "bfill" (copy next value backward), "mean" (fill with column mean).',
        'Columns where more than 50% of values are NaN/missing are DROPPED entirely (configurable threshold).',
        'Air Quality dataset specifically replaces -200 with NaN first, then interpolates.',
        warn='Interpolation creates artificial in-between values. These are NOT real sensor readings  --  label them as such.',
        q_num='C7')

    items += qa(
        'How are chronological train/validation/test splits created?',
        'chronological_split() in preprocessing.py does this. For "chronological" split:',
        '- First 70% of time steps -> training set',
        '- Next 15% -> validation set',
        '- Last 15% -> test set',
        'For subject-based split (Daphnet): subjects are divided by group_id, not time.',
        'The key guarantee: NO future data enters training. If the test set has timestamps after '
        'the training set, this is automatically satisfied.',
        'check_leakage() asserts zero timestamp overlap between all three splits.',
        formula='split_idx_1 = int(T * 0.70)  # end of train\nsplit_idx_2 = int(T * 0.85)  # end of val\ntrain = X[:split_idx_1], val = X[split_idx_1:split_idx_2], test = X[split_idx_2:]',
        q_num='C8')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_D(story):
    story.append(chapter_header('D', 'The Teacher Model  --  CLIP', HexColor('#4A0070')))
    story.append(sp())
    items = []

    items += qa(
        'What is a Vision-Language Model (VLM)?',
        'A VLM is an AI model trained to understand BOTH images and text at the same time. '
        'It learns to match images with their descriptions. For example, it learns that '
        'a photo of a dog matches the text "a dog" but NOT "a cat."',
        'This is done by showing the model millions of (image, caption) pairs from the internet '
        'and training it to push matching pairs close together in a shared embedding space.',
        q_num='D1')

    items += qa(
        'What is CLIP, and what does its name stand for?',
        'CLIP stands for <b>Contrastive Language-Image Pre-training</b>. '
        'It was created by OpenAI in 2021.',
        'It was trained on 400 MILLION image-text pairs scraped from the internet.',
        'The "contrastive" part means: given a batch of N images and N captions, '
        'CLIP learns to match each image to its correct caption (N correct pairs) '
        'and push all wrong N^2-N combinations apart.',
        'After training, CLIP can classify any image into any category just by comparing '
        'image similarity to text descriptions  --  no task-specific training needed (zero-shot).',
        imp='CLIP was NEVER trained on sensor data or time-series heatmaps. Its domain is natural photos.',
        q_num='D2')

    items += qa(
        'Which exact pretrained CLIP model is used in our implementation?',
        'Model: <b>openai/clip-vit-base-patch32</b> from Hugging Face Transformers.',
        'This is CLIP with a Vision Transformer (ViT) image encoder that divides the image '
        'into 32x32 patches before processing.',
        'Parameters: ~151 million total (vision encoder + text encoder).',
        code='# src/teacher/clip_teacher.py\nfrom transformers import CLIPModel, CLIPProcessor\nself.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")\nself.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")',
        q_num='D3')

    items += qa(
        'What does freezing the teacher model mean, and how is it done?',
        'Freezing means making all the teacher\'s parameters (weights) UNTRAINABLE. '
        'They will not change during student training. The teacher is a fixed function.',
        'This is done by setting requires_grad=False on every parameter in the model.',
        'After loading CLIP, the code calls model.eval() (puts it in inference mode, '
        'disables dropout) and loops through all parameters.',
        'A hard assertion verifies no parameter has requires_grad=True after freezing.',
        code='# src/teacher/clip_teacher.py\nfor p in self.model.parameters():\n    p.requires_grad = False\n# Verification:\nassert not any(p.requires_grad for p in self.model.parameters())',
        imp='If the teacher were not frozen, it would start changing during student training and produce inconsistent scores.',
        q_num='D4')

    items += qa(
        'Can CLIP directly process a numerical time-series matrix?',
        'NO. CLIP expects a PIL Image of size 224x224 pixels as its vision input, '
        'and a text string (up to 77 tokens) as its language input.',
        'A numerical matrix like [[1.2, 0.5, -0.3], [1.3, 0.6, -0.2], ...] '
        'cannot be fed to CLIP directly.',
        'This is WHY we convert windows to heatmap images first.',
        q_num='D5')

    items += qa(
        'How exactly is a heatmap generated from a window?',
        'window_to_image() in src/teacher/image_gen.py does this:',
        '1. Takes window X_w of shape (L, F)  --  L time steps, F features.',
        '2. Creates a matplotlib figure with Agg backend (non-interactive, no GUI).',
        '3. Draws the matrix as a seaborn/matplotlib heatmap: time on X-axis, '
        'features (sensors) on Y-axis, value shown as color (Viridis colormap).',
        '4. NO axis labels containing timestamps or feature names that hint at ground truth.',
        '5. Renders to a PIL Image at 100 DPI, resizes to exactly 224x224 pixels (LANCZOS filter).',
        '6. Closes the figure immediately to free memory.',
        formula='X_w shape (L=128, F=9) -> heatmap -> PIL Image (224, 224, 3)',
        code='# src/teacher/image_gen.py\nfig, ax = plt.subplots(figsize=(2.24, 2.24))\nax.imshow(window.T, aspect="auto", cmap="viridis")\nbuf = io.BytesIO()\nfig.savefig(buf, format="png", dpi=100)\nplt.close(fig)\nimg = PIL.Image.open(buf).resize((224, 224), PIL.Image.LANCZOS)',
        q_num='D6')

    items += qa(
        'What are the exact normal and anomaly text prompts?',
        'From config.yaml:',
        '<b>Normal:</b> "A normal regular time series signal without anomalies"',
        '<b>Anomaly:</b> "An anomalous irregular time series signal with spikes or distortions"',
        'These prompts were designed to be clearly contrastive and domain-agnostic.',
        'They describe VISUAL properties (spikes, irregularity) that CLIP might recognize in a heatmap.',
        warn='These prompts are NOT optimized. Different prompts may give very different teacher scores. '
             'Prompt engineering is listed as a future novelty direction (CoOp-style learnable prompts).',
        q_num='D7')

    items += qa(
        'How does CLIP compare an image with the two text prompts?',
        '1. The image is encoded by CLIP\'s vision encoder into a 512-dimensional vector: image_embedding.',
        '2. Each text prompt is encoded by CLIP\'s text encoder into a 512-dimensional vector: text_embedding.',
        '3. Cosine similarity is computed between the image embedding and EACH text embedding.',
        '4. The two cosine similarity scores are passed through softmax to get a probability distribution '
        'summing to 1: [p_normal, p_anomaly].',
        formula='similarity_normal = cos(image_emb, text_emb_normal)\nsimilarity_anomaly = cos(image_emb, text_emb_anomaly)\n\n[p_normal, p_anomaly] = softmax([similarity_normal, similarity_anomaly])',
        code='# src/teacher/clip_teacher.py - score_window()\nwith torch.no_grad():\n    outputs = self.model(**inputs)\n    logits = outputs.logits_per_image  # shape (1, 2)\n    p_T = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # [p_normal, p_anomaly]',
        q_num='D8')

    items += qa(
        'Are these teacher probabilities calibrated real anomaly probabilities?',
        'NO. This is one of the most important points to understand.',
        'CLIP was trained to match images to captions on natural photographs. '
        'When it sees a heatmap of sensor data, it compares it to its learned visual vocabulary '
        'of "normal" and "anomalous" looking patterns  --  based on what "normal" and "anomalous" '
        'mean in NATURAL IMAGES, not sensor data.',
        'The resulting p_anomaly score is a proxy or heuristic signal, NOT a true Bayesian '
        'posterior probability P(anomaly | sensor_data).',
        'This is why we call it a "soft prior" and regularize the student with it rather than '
        'using it as hard labels.',
        imp='The code contains this warning: "CLIP was not trained for time-series anomaly detection. '
            'Teacher scores are candidate signals, not calibrated anomaly probabilities."',
        q_num='D9')

    items += qa(
        'Why are teacher outputs cached?',
        'CLIP inference scores rendered images against prompts. Runtime depends on device and batch size; it has not been measured here. '
        'For 5,000 windows, that\'s 600 seconds = 10 minutes just for teacher scoring.',
        'Since the teacher is FROZEN (never changes), its output for a given window never changes.',
        'So we run teacher scoring ONCE and save results to disk as a .pkl file (pickle).',
        'Subsequent training runs load from cache instead of re-running CLIP.',
        'Cache invalidation happens if the windowing config changes (different L or stride would '
        'produce different windows with different heatmaps).',
        code='# src/teacher/clip_teacher.py - cache_and_score()\nif cache_path.exists():\n    scores, meta = load_image_cache(cache_path)  # load from disk\nelse:\n    images = batch_to_images(windows, ...)\n    scores = self.score_batch(images, ...)\n    save_image_cache(scores, cache_path, meta)  # save to disk',
        q_num='D10')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_E(story):
    story.append(chapter_header('E', 'The Student Model  --  GRU Architecture', HexColor('#1B5E20')))
    story.append(sp())
    items = []

    items += qa(
        'What is an RNN, and what problem does it solve?',
        'RNN = Recurrent Neural Network. It is a neural network that processes sequences '
        'by maintaining a "memory" called a hidden state.',
        'In a regular neural network, each input is processed independently. '
        'In an RNN, when processing step t, the network also sees its own output '
        'from step t-1 (the hidden state h_{t-1}).',
        'This allows it to "remember" recent history and model temporal patterns.',
        formula='h_t = f(W_x * x_t + W_h * h_{t-1} + b)\n\nwhere x_t = current input, h_{t-1} = previous hidden state\nh_t = new hidden state (the "memory")',
        q_num='E1')

    items += qa(
        'What is a GRU, and how is it different from a basic RNN?',
        'GRU = Gated Recurrent Unit. It fixes the "vanishing gradient" problem of basic RNNs.',
        'Problem with basic RNN: when you backpropagate through hundreds of time steps, '
        'gradients shrink exponentially  --  the model forgets long-range patterns.',
        '<b>GRU solution:</b> Two gates control how much old memory to keep and how much new '
        'information to write:',
        '<b>Update gate z_t:</b> How much of the OLD hidden state to keep (vs. replace). '
        'Range [0,1]. 0 = forget everything, 1 = keep everything.',
        '<b>Reset gate r_t:</b> How much of the old state to USE when computing new content.',
        formula='Reset gate:    r_t = sigmoid(W_r * [h_{t-1}, x_t])\nUpdate gate:   z_t = sigmoid(W_z * [h_{t-1}, x_t])\nNew content:   n_t = tanh(W_n * [r_t * h_{t-1}, x_t])\nNew state:     h_t = (1 - z_t) * h_{t-1} + z_t * n_t',
        q_num='E2')

    items += qa(
        'How is a GRU different from an LSTM?',
        'LSTM has THREE gates (input, forget, output) and TWO states (hidden + cell). '
        'GRU has TWO gates (update, reset) and ONE state (hidden).',
        'GRU has ~25% fewer parameters than LSTM for the same hidden size, '
        'trains faster, and performs comparably on most tasks.',
        'We chose GRU for the student because: fewer parameters = faster inference = more suitable for edge deployment.',
        code='# Our GRU: 2 layers, H=64, F input features\n# Fewer params than LSTM:\n# GRU params per layer ~ 3 * (F + H) * H + 3H\n# LSTM params per layer ~ 4 * (F + H) * H + 4H',
        q_num='E3')

    items += qa(
        'What is the exact architecture of our GRU student model?',
        'Defined in src/student/model.py  --  class GRUStudent.',
        '<b>Layer 1  --  GRU Backbone:</b> 2 stacked GRU layers, hidden size H=64, dropout=0.3 '
        'between layers, unidirectional (processes left-to-right only), batch_first=True. '
        'Input shape: (B, L, F). Output shape: (B, L, 64) + hidden states.',
        '<b>Layer 2  --  Mean Pooling:</b> Average the L hidden vectors to get one vector per window. '
        'Shape: (B, 64).',
        '<b>Layer 3  --  Window Head:</b> Linear(64->64) -> ReLU -> Dropout(0.3) -> Linear(64->2). '
        'Outputs window logits of shape (B, 2).',
        '<b>Layer 4  --  Step Head:</b> Linear(64->32) -> ReLU -> Linear(32->1) -> squeeze. '
        'Applied to EACH time step\'s hidden vector. Outputs step logits of shape (B, L).',
        formula='Input:         (B, L, F)\nGRU output:    (B, L, 64)\nMean pool:     (B, 64)       <- for window head\nWindow logits: (B, 2)        <- is this window anomalous?\nStep logits:   (B, L)        <- is each time step anomalous?',
        tip='F is different for each dataset. The GRU is re-created for each dataset with the correct input_size.',
        q_num='E4')

    items += qa(
        'How many trainable parameters does the student actually have?',
        'The student parameter count depends on input feature count and the selected GRU configuration.',
        'The app displays the exact trainable count for the instantiated student. Measure the selected CLIP model count too before reporting a compression ratio.',
        code='# src/student/model.py - parameter_count()\ndef parameter_count(self) -> int:\n    return sum(p.numel() for p in self.parameters() if p.requires_grad)',
        imp='Do not claim a size ratio or comparable performance until both values are measured for the selected configuration.',
        q_num='E5')

    items += qa(
        'What information does a hidden state h_t contain?',
        'h_t is a 64-dimensional vector (for hidden_size=64). '
        'Think of it as a compressed "summary" of everything the GRU has seen from step 1 up to step t.',
        'It encodes: recent trends (is the value going up or down?), '
        'recent patterns (did we just see a spike?), long-range context (how unusual is the current '
        'value given the history?).',
        'The step head reads h_t and predicts: "Given what I have seen up to time t, '
        'how likely is time step t to be anomalous?"',
        'The window head reads the MEAN of all h_t values and predicts: '
        '"Given the entire window history, how likely is this window to contain an anomaly?"',
        q_num='E6')

    items += qa(
        'Does the student directly receive the original numerical data or the heatmap image?',
        '<b>The student receives the original numerical data  --  NOT the heatmap image.</b>',
        'Input to GRU: X_windows of shape (B, L, F)  --  raw normalized floating-point sensor values.',
        'The heatmap image is only created for the TEACHER (CLIP).',
        'During student training and inference, no image is created. '
        'The student operates entirely in the numerical domain.',
        'This is a key architectural choice: the student\'s speed advantage comes partly from '
        'not needing expensive image rendering.',
        imp='Teacher sees heatmap images. Student sees raw numbers. They talk through the loss function.',
        q_num='E7')

    items += qa(
        'Does the student require the VLM during inference?',
        '<b>NO  --  the student is completely independent of CLIP at inference time.</b>',
        'The distillation happens ONLY during training: CLIP provides soft targets that shape '
        'how the student learns. Once training is done, the student checkpoint contains '
        'only GRU weights  --  no reference to CLIP.',
        'During inference: load checkpoint -> create GRU -> forward pass on raw windows -> '
        'apply threshold -> output anomaly timeline.',
        'This is the key practical value: deploy a 500KB GRU on an edge sensor device '
        'instead of a 600MB CLIP model.',
        imp='This is why knowledge distillation is powerful: transfer knowledge at training time, '
            'discard teacher at inference time.',
        q_num='E8')

    for item in items:
        story.append(item)
    # Let the next chapter flow onto the next page without risking an empty page.


def section_FG(story):
    story.append(chapter_header('F+G', 'Data Flow and Knowledge Distillation Mathematics', HexColor('#BF360C')))
    story.append(sp())
    items = []

    items += qa(
        'Can you trace one sample from CSV to final prediction?',
        '<b>Step 1  --  CSV Load:</b> load_csv("sensor_data.csv") reads the file, detects separator, '
        'returns DataFrame of shape (9358, 14) for Air Quality.',
        '<b>Step 2  --  Preprocessing:</b> Sort chronologically -> drop duplicates -> split 70/15/15 -> '
        'impute NaNs -> fit StandardScaler on train -> transform all splits.',
        '<b>Step 3  --  Windowing:</b> make_windows(X_train, y_step_train, L=128, S=32) creates '
        'N_train windows of shape (N_train, 128, F).',
        '<b>Step 4  --  Teacher scoring:</b> For each window, render heatmap (224x224 image) -> '
        'CLIP encodes image + 2 prompts -> softmax similarity -> p_T = [p_normal, p_anomaly].',
        '<b>Step 5  --  Student training:</b> For each batch of 32 windows, forward pass through '
        'GRU -> window_logits (B,2) + step_logits (B,L) -> compute 3-term loss -> backprop -> '
        'update GRU weights.',
        '<b>Step 6  --  Threshold selection:</b> On val set, sweep 100 thresholds, pick one '
        'maximizing F1.',
        '<b>Step 7  --  Inference on test:</b> Forward pass -> sigmoid(step_logits) -> apply '
        'threshold -> anomaly_mask[t] in {0,1} for every time step.',
        '<b>Step 8  --  Localization:</b> Cluster consecutive 1s into intervals -> '
        '[(t_start1, t_end1), (t_start2, t_end2), ...].',
        '<b>Step 9  --  Evaluation:</b> Compare against y_true -> compute F1, AUPRC, IoU, Event-F1.',
        q_num='F1')

    items += qa(
        'What is knowledge distillation, and why do we use it?',
        'Knowledge distillation is a technique where a large, powerful (but slow) model called '
        'the TEACHER trains a small, fast model called the STUDENT.',
        'Instead of training the student only on binary labels (0=normal, 1=anomaly), '
        'the student also learns to mimic the teacher\'s probability distributions.',
        'Analogy: Imagine learning from both a textbook (hard labels: right/wrong answer) '
        'AND a wise professor (soft guidance: this is probably right, that is possibly wrong, '
        'these two cases are surprisingly similar). The professor\'s nuanced guidance helps '
        'the student generalize better than the textbook alone.',
        imp='The student trained with KD uses the teacher\'s "world model" as an additional learning signal.',
        q_num='G1')

    items += qa(
        'What are hard labels and soft targets?',
        '<b>Hard labels:</b> Binary ground truth  --  y=0 (normal) or y=1 (anomaly). '
        'No in-between. Tells the model ONLY whether each window is anomalous, nothing more.',
        '<b>Soft targets:</b> The teacher\'s probability distribution  --  e.g., [p_normal=0.3, p_anomaly=0.7]. '
        'This says "I am 70% sure this is anomalous" rather than just "anomalous."',
        'The soft target for a borderline window might be [0.48, 0.52]  --  almost ambiguous. '
        'The student learns that this window is uncertain, which is itself useful information.',
        'This extra information helps the student build better decision boundaries.',
        q_num='G2')

    items += qa(
        'What is KL divergence, and why is it used for distillation?',
        'KL divergence (Kullback-Leibler divergence) measures how different two probability '
        'distributions are. It\'s zero when they are identical, positive when they differ.',
        'We use KL divergence to measure how far the student\'s probability distribution '
        'is from the teacher\'s distribution  --  then minimize this difference.',
        'This forces the student to produce probabilities that look like the teacher\'s, '
        'not just to classify correctly.',
        formula='KL(p_T || p_S) = sum over classes c:  p_T(c) * log(p_T(c) / p_S(c))\n\nwhere p_T = teacher probs (target/reference)\n      p_S = student log-probs (what we are training)',
        warn='In PyTorch KLDivLoss: first argument is log(student), second is teacher. Not the other way around!',
        q_num='G3')

    items += qa(
        'Why do we multiply the KL loss by T^2?',
        'When you use temperature T to soften distributions, the softmax output values become smaller '
        '(closer to uniform). The gradient magnitudes also scale down by factor 1/T^2.',
        'Example: a base teacher distribution [0.9, 0.1] becomes about [0.75, 0.25] at T=2. '
        'This keeps the first class preferred while giving the second class more weight.',
        'Multiplying by T^2 restores the original gradient magnitude, ensuring that the '
        'KD loss has similar influence on training regardless of what temperature is used.',
        'This is Hinton\'s original trick from the 2015 paper.',
        formula='L_KD = T^2 * KL(softmax(log(p_T)/T) || softmax(z_S/T))\n\nT=2 means T^2=4, so we multiply KL by 4. The teacher and student distributions are both softened.',
        code='# src/student/losses.py - KnowledgeDistillationLoss.forward()\nteacher_log_p = torch.log(t_probs.clamp_min(eps))\nteacher_soft_p = torch.softmax(teacher_log_p / T, dim=-1)\nstudent_log_p = F.log_softmax(student_logits / T, dim=-1)\nkl = self.kl_div(student_log_p, teacher_soft_p)\nreturn (T * T) * kl  # T=2, so multiply by 4',
        q_num='G4')

    items += qa(
        'What is the localization loss, and why is weighted BCE used?',
        'The localization loss trains the STEP HEAD  --  it teaches the model to predict '
        'which individual time steps within a window are anomalous.',
        '<b>Binary Cross-Entropy (BCE):</b> For each time step, the step head outputs a logit. '
        'BCE loss compares sigmoid(logit) against the binary label (0 or 1).',
        '<b>Why weighted:</b> Most time steps are NORMAL. Without weighting, the model learns '
        'to always predict 0 (normal) and gets a low loss just by ignoring anomalies. '
        'The positive weight w_+=10 makes each anomalous time step count 10x more toward the loss.',
        formula='L_loc = -(1/B*L) * SUM_{b,t} [\n  w+ * y_{bt} * log(sigmoid(logit_{bt}))\n  + (1-y_{bt}) * log(1 - sigmoid(logit_{bt}))\n]\n\nw+ = 10.0  (anomalous steps count 10x more)',
        code='# src/student/losses.py - LocalizationLoss\nF.binary_cross_entropy_with_logits(\n    step_logits, y_true, pos_weight=torch.tensor([10.0])\n)',
        q_num='G5')

    items += qa(
        'What is the total loss function? What do beta and lambda_KD control?',
        'Three loss terms are combined:',
        '<b>L_loc:</b> Weighted BCE on step-level predictions (teaches WHERE in the window anomalies are).',
        '<b>L_win:</b> Cross-entropy on window-level predictions (teaches WHETHER the window is anomalous). '
        'beta=1.0 gives this term a coefficient of 1.0, matching L_loc\'s coefficient; the actual loss magnitudes can differ.',
        '<b>L_KD:</b> Temperature-scaled KL divergence between teacher and student distributions. '
        'lambda_KD=0.5 multiplies the KD loss by 0.5; its actual contribution depends on the KD loss magnitude.',
        'If lambda_KD=0: student trained on hard labels only (student_baseline mode). '
        'If lambda_KD=too large: teacher noise dominates, ground-truth labels have little effect.',
        formula='L_total = L_loc + beta * L_win + lambda_KD * L_KD\n\nWith our config values:\nL_total = L_loc + 1.0 * L_win + 0.5 * L_KD',
        imp='The loss has three jobs: localize anomalies (step BCE), classify windows (window CE), and mimic teacher (KD KL).',
        q_num='G6')

    items += qa(
        'Why is the teacher target detached from gradient computation?',
        'torch.Tensor.detach() creates a version of the tensor that is excluded from the computational graph.',
        'We detach p_T because: (1) the teacher is FROZEN  --  we never want gradients to flow '
        'into CLIP\'s parameters even by accident, and (2) p_T is treated as a constant target, '
        'not a variable to optimize.',
        'If we forgot to detach, PyTorch would try to compute gradients through the teacher, '
        'which would fail (parameters are requires_grad=False) or waste computation.',
        code='# src/student/losses.py - KnowledgeDistillationLoss.forward()\nt_probs = teacher_probs.detach().to(device)  # <-- KEY: detach from graph',
        q_num='G7')

    items += qa(
        'How does backpropagation update the student?',
        '<b>Step 1:</b> Forward pass: X -> GRU -> window_logits + step_logits.',
        '<b>Step 2:</b> Compute L_total from logits, ground-truth labels, and detached teacher probs.',
        '<b>Step 3:</b> loss.backward()  --  PyTorch computes dL/dW for every GRU parameter W '
        'by automatic differentiation through the computational graph.',
        '<b>Step 4:</b> Gradient clipping: clip gradients to max norm 1.0 to prevent '
        'exploding gradients (a common RNN problem).',
        '<b>Step 5:</b> optimizer.step()  --  Adam updates each parameter: W = W - lr * adjusted_gradient.',
        code='# src/training/trainer.py - Trainer.train_epoch()\nself.optimizer.zero_grad()\noutputs = self.model(X)\nloss, loc, win, kd = self.loss_fn(window_logits, step_logits, y_window, y_step, t_probs)\nloss.backward()\nnn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)\nself.optimizer.step()',
        q_num='G8')

    for item in items:
        story.append(item)
    # Avoid a forced break: this section can already end at the top of a fresh page.


def section_H(story):
    story.append(chapter_header('H', 'Training Procedure and Hyperparameters', HexColor('#1A237E')))
    story.append(sp())
    items = []

    items += qa(
        'What optimizer is used, and why was Adam selected?',
        'Optimizer: <b>Adam</b> (Adaptive Moment Estimation), configured in config.yaml.',
        'Also supports: AdamW (Adam with decoupled weight decay) and SGD with momentum.',
        '<b>Why Adam:</b> Adam adapts the learning rate for each parameter individually based on '
        'running estimates of first and second moments of gradients. This makes it much faster '
        'to converge than standard SGD on deep networks. It\'s the standard choice for most '
        'deep learning tasks.',
        formula='Adam update:\nm_t = beta1 * m_{t-1} + (1-beta1) * g_t       (mean)\nv_t = beta2 * v_{t-1} + (1-beta2) * g_t^2    (variance)\nW = W - lr * m_t / (sqrt(v_t) + epsilon)      (update)',
        code='# src/training/trainer.py\nself.optimizer = torch.optim.Adam(\n    self.model.parameters(),\n    lr=0.001,\n    weight_decay=1e-4\n)',
        q_num='H1')

    hparam_data = [
        ['Hyperparameter', 'Value', 'Where Set', 'What it does'],
        ['Learning rate', '0.001 (1e-3)', 'config.yaml: training.learning_rate',
         'Step size for parameter updates. Too high -> unstable. Too low -> slow.'],
        ['Batch size', '32', 'config.yaml: training.batch_size',
         'Windows processed per gradient update.'],
        ['Max epochs', '30', 'config.yaml: training.epochs',
         'Maximum training iterations. Early stopping usually stops before this.'],
        ['Early stopping patience', '5', 'config.yaml: training.early_stopping_patience',
         'Stop if val loss does not improve for 5 consecutive epochs.'],
        ['Weight decay', '1e-4', 'config.yaml: training.weight_decay',
         'L2 regularization  --  penalizes large weights to prevent overfitting.'],
        ['Gradient clipping', '1.0', 'config.yaml: training.gradient_clip',
         'Clips gradient norm to max 1.0  --  prevents RNN gradient explosions.'],
        ['LR Scheduler', 'ReduceLROnPlateau', 'trainer.py line 254',
         'Halves LR if val loss does not improve for patience//2=2 epochs.'],
        ['KD Temperature T', '2.0', 'config.yaml: distillation.temperature',
         'Softens distributions. Higher T -> softer, more information.'],
        ['KD weight lambda', '0.5', 'config.yaml: distillation.lambda_kd',
         'KD loss contribution = 0.5 x L_KD.'],
        ['Window weight beta', '1.0', 'config.yaml: distillation.beta',
         'Window loss contribution = 1.0 x L_win.'],
        ['Positive weight w+', '10.0', 'config.yaml: distillation.pos_weight',
         'Anomalous steps count 10x more in BCE loss.'],
        ['Random seed', '42', 'config.yaml: seed', 'Reproducibility.'],
    ]
    items.append(build_table(hparam_data, [3*cm, 2.5*cm, 4*cm, 6.8*cm]))
    items.append(P('All Hyperparameters  --  What They Are and Where They Come From', 'caption'))
    items.append(sp())

    items += qa(
        'What is early stopping?',
        'Early stopping monitors the validation loss at the end of each epoch. '
        'If the validation loss does not IMPROVE (decrease by at least 1e-5) for 5 consecutive epochs, '
        'training stops  --  even if the maximum epoch count has not been reached.',
        '<b>Why:</b> After a certain point, further training on the training set only makes the model '
        'memorize training data better (overfitting) while validation performance degrades.',
        '<b>Best weights restore:</b> When training stops, the code loads back the model weights '
        'from the BEST epoch (lowest validation loss), not from the final epoch.',
        code='# src/training/trainer.py\nif val_loss < (best_val_loss - 1e-5):\n    best_val_loss = val_loss\n    best_state = copy.deepcopy(self.model.state_dict())\nelse:\n    patience_counter += 1\nif patience_counter >= self.patience:\n    break  # early stopping',
        q_num='H2')

    items += qa(
        'Does teacher inference run at every epoch?',
        'NO  --  this is a crucial efficiency point.',
        'Teacher inference (heatmap rendering + CLIP forward pass) runs ONCE before training starts, '
        'and the results are cached in memory (and optionally to disk).',
        'During each training epoch, the cached teacher probability array p_T is indexed '
        'to get teacher probabilities for each batch, without re-running CLIP.',
        'This is why the 4th element of each DataLoader batch can be teacher_probs  --  '
        'they are pre-computed and passed directly.',
        imp='Without caching, teacher inference would need to run for EVERY BATCH OF EVERY EPOCH, '
            'making training 100x slower.',
        q_num='H3')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_IJ(story):
    story.append(chapter_header('I+J', 'Anomaly Scoring, Localization, and Evaluation', HexColor('#004D40')))
    story.append(sp())
    items = []

    items += qa(
        'How does the student produce a score for every time step?',
        '<b>Step 1  --  Forward pass:</b> A batch of windows X (shape BxLxF) goes through the GRU.',
        '<b>Step 2  --  Step logits:</b> The step head applies a linear layer to each time step\'s '
        'hidden vector h_t, producing step_logits of shape (B, L)  --  one raw score per time step.',
        '<b>Step 3  --  Sigmoid activation:</b> step_scores = sigmoid(step_logits). '
        'Sigmoid maps any real number to [0, 1]. A score of 0.8 means "80% likely anomalous."',
        '<b>Step 4  --  Aggregation:</b> Since windows overlap, each time step appears in multiple windows. '
        'Scores are averaged across all windows that contain that step.',
        formula='sigmoid(x) = 1 / (1 + e^(-x))\nRange: (0, 1)\n\nFor step t in window w:\n  score[t] += step_scores[w, t]\n  count[t] += 1\nfinal_score[t] = score[t] / count[t]',
        code='# src/evaluation/localization.py - aggregate_step_scores()\nfor w_idx, start_t in enumerate(window_starts):\n    for local_t in range(window_length):\n        global_t = start_t + local_t\n        sum_scores[global_t] += step_scores[w_idx, local_t]\n        counts[global_t] += 1\nfinal = sum_scores / counts',
        q_num='I1')

    items += qa(
        'How is the anomaly threshold selected?',
        'tune_threshold() in src/evaluation/threshold.py sweeps 100 threshold values from 0.01 to 0.99.',
        'For each candidate threshold theta, it computes: y_pred[t] = 1 if score[t] >= theta, else 0. '
        'Then computes F1 score against validation labels.',
        'The threshold that gives the HIGHEST F1 on the VALIDATION SET is selected.',
        'This threshold is then applied to the TEST SET without any further adjustment.',
        '<b>Why validation only:</b> If we picked the threshold on the test set, we would be '
        '"cheating"  --  tuning to the specific test data means the test results are overly optimistic.',
        formula='For each candidate threshold theta in [0.01, 0.99]:\n  y_pred = (score >= theta)\n  f1 = F1(y_val, y_pred)\nbest_theta = argmax(f1)',
        code='# src/evaluation/threshold.py - tune_threshold()\nbest_threshold, info = select_threshold(y_val, scores_val, objective="f1", n_thresholds=100)',
        q_num='I2')

    items += qa(
        'What is PR-AUC, and why is it the primary metric for imbalanced data?',
        '<b>Precision:</b> Of all the windows I predicted as anomalous, what fraction actually are?  '
        'TP / (TP + FP). High precision = few false alarms.',
        '<b>Recall:</b> Of all truly anomalous windows, what fraction did I correctly detect?  '
        'TP / (TP + FN). High recall = few missed anomalies.',
        '<b>PR Curve:</b> Plot (Precision, Recall) for all possible thresholds.',
        '<b>PR-AUC (= Average Precision):</b> Area under this curve. Ranges from 0 to 1.',
        '<b>Why preferred over ROC-AUC for imbalanced data:</b> ROC-AUC can look good (e.g., 0.9) '
        'even when the model produces many false positives, because it uses true negative rate (TN/(TN+FP)) '
        'in the denominator, and TN is huge when negatives dominate. '
        'PR-AUC directly measures performance on the positive (minority, anomaly) class.',
        imp='A random classifier on 0.17% fraud data gets ROC-AUC approx. 0.5 but PR-AUC approx. 0.0017. '
            'PR-AUC shows you the real difficulty of the task.',
        q_num='J1')

    items += qa(
        'What is Point-Adjusted F1, and why is it problematic?',
        '<b>Point Adjustment (PA) heuristic:</b> If ANY predicted anomaly timestep falls within '
        'a true anomaly interval, ALL timesteps in that interval are marked as "detected" (recall=1).',
        '<b>Intuition:</b> If you detect the anomaly 2 hours late, you still get credit for finding it.',
        '<b>Problem (Kim et al. 2022):</b> A model that randomly guesses 1% positive predictions '
        'will eventually hit inside a LONG anomaly interval. With PA, this inflates recall to 1.0 '
        'and F1 to artificially high values  --  masking that the model is essentially random.',
        '<b>Our implementation:</b> We implement PA-F1 (point_adjustment() in localization.py) '
        'but ALSO report raw F1 and IoU, and include a unit test demonstrating the inflation flaw.',
        code='# src/evaluation/localization.py - point_adjustment()\n# If any pred=1 falls inside a true anomaly segment,\n# set ALL steps of that segment to pred=1\ny_adj = y_pred.copy()\nfor seg_start, seg_end in true_segments:\n    if y_pred[seg_start:seg_end].any():\n        y_adj[seg_start:seg_end] = 1',
        q_num='J2')

    items += qa(
        'What are Event-F1 and Mean-IoU?',
        '<b>Event-F1:</b> Instead of counting individual timesteps, count EVENTS (contiguous anomaly intervals).',
        'A true event is "detected" if ANY predicted interval overlaps with it (at least 1 timestep).',
        'Event Precision = detected_true_events / total_predicted_events',
        'Event Recall = detected_true_events / total_true_events',
        'Event F1 = harmonic mean of event precision and recall.',
        '<b>Mean-IoU:</b> For each true anomaly interval, find all overlapping predicted intervals, '
        'union them, and compute: |intersection| / |union|.',
        'Mean-IoU averages this ratio over all true events.',
        formula='IoU for one event:\n  intersection = overlap length of true and predicted intervals\n  union = total length covered by either interval\n  IoU = intersection / union\n\nMean-IoU = average IoU across true events',
        imp='IoU of 1.0 means perfect boundaries. IoU of 0.5 means you predicted an interval twice as long as the truth.',
        q_num='J3')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_K(story):
    story.append(chapter_header('K', 'Streamlit Dashboard  --  7 Pages Explained', HexColor('#37474F')))
    story.append(sp())
    items = []

    pages_data = [
        ['Page', 'Name', 'File', 'What the User Does'],
        ['1', 'Overview & Architecture', 'src/ui/page1_overview.py',
         'Read the pipeline description, see the Mermaid flowchart, understand dataset catalog'],
        ['2', 'Upload & Inspect', 'src/ui/page2_upload.py',
         'Upload CSV, select timestamp/feature/label columns, see DataQualityReport, preview data'],
        ['3', 'Preprocessing', 'src/ui/page3_preprocessing.py',
         'Configure imputation strategy, scaler type, window length, stride, split ratios. Apply preprocessing.'],
        ['4', 'Teacher Model (CLIP)', 'src/ui/page4_teacher.py',
         'Enter normal/anomaly prompts, select image mode (heatmap/lineplot), run CLIP scoring. Preview heatmaps.'],
        ['5', 'Student Training (GRU)', 'src/ui/page5_training.py',
         'Configure GRU hyperparameters and training mode. Click Train. Watch live loss curves.'],
        ['6', 'Evaluation & Threshold', 'src/ui/page6_evaluation.py',
         'Compare student KD with the supervised baseline. See PR/ROC curves and threshold plots.'],
        ['7', 'Inference & Timeline', 'src/ui/page7_inference.py',
         'Run inference on test set. See anomaly timeline plot. Download predictions CSV and PDF report.'],
    ]
    items.append(build_table(pages_data, [1*cm, 3.5*cm, 4*cm, 7.8*cm], TEAL))
    items.append(P('The 7 Streamlit Dashboard Pages', 'caption'))
    items.append(sp())

    items += qa(
        'What tests have actually been executed?',
        'The existing test suite was run before the latest KD temperature correction: <b>70 passed</b>.',
        'The temperature correction and its updated equality assertion have not yet been rerun. Re-run the suite before presenting this code snapshot as verified.',
        code='python -m pytest tests/ -v',
        q_num='K1')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_L(story):
    story.append(chapter_header('L', 'Critical Review and Debugging', HexColor('#B71C1C')))
    story.append(sp())
    items = []

    items += qa(
        'What are the biggest limitations of the current implementation?',
        '<b>1. No actual benchmark results:</b> The benchmark files are not included in this workspace and no real dataset runs have been saved.',
        '<b>2. Air Quality "ground truth" is fake:</b> We injected the anomalies ourselves. '
        'The model\'s success on Air Quality only proves it can recover OUR injected patterns, '
        'not that it detects real sensor anomalies.',
        '<b>3. Domain gap for CLIP:</b> CLIP has no concept of sensor readings. '
        'A "normal" heatmap from pressure sensors may look visually similar to an "anomalous" '
        'one if the values are numerically close. The teacher\'s signal quality is untested.',
        '<b>4. Fixed architecture per dataset:</b> Must retrain a new GRU for each dataset. '
        'No cross-dataset transfer.',
        '<b>5. No time for hyperparameter search:</b> All hyperparameters are defaults. '
        'Tuning LR, hidden size, or KD temperature might significantly improve results.',
        warn='No public repository URL is included. Refer to the accompanying project folder for the source code.',
        q_num='L1')

    items += qa(
        'Have the tests been verified for this code snapshot?',
        'The earlier code snapshot had 70 passing tests. This follow-up changed teacher temperature scaling, so the current snapshot still needs a fresh run.',
        code='python -m pytest tests/ -v',
        q_num='L2')

    items += qa(
        'Which report claims have not been independently verified?',
        '<b>NOT MEASURED:</b> No held-out test metrics are available in this workspace.',
        '<b>NOT MEASURED:</b> Teacher/student latency and model-size comparison.',
        '<b>NOT VERIFIED AFTER THE LATEST KD FIX:</b> Test-suite pass count. The previous snapshot had 70 passing tests; full application launch and benchmark training runs are also not verified.',
        warn='Be honest with the TA: say "results are expected based on architecture design and literature, '
             'not from completed experiment runs on real datasets."',
        q_num='L3')

    items += qa(
        'What would you change if knowledge distillation does not improve performance?',
        '<b>Diagnose first:</b> Is the teacher providing useful signal? Compute correlation between '
        'teacher p_anomaly and ground-truth labels on the validation set. If near 0, teacher is noise.',
        '<b>Option 1  --  Better prompts:</b> Try more specific prompts for each dataset domain. '
        'For oil wells: "pressure sensor showing normal production" vs. "pressure sensor showing slug flow failure."',
        '<b>Option 2  --  Learnable prompts (CoOp):</b> Add a few learnable token vectors before the prompt '
        'and optimize them on labeled val data. This adapts CLIP to the specific domain.',
        '<b>Option 3  --  Different image representation:</b> Try line plots instead of heatmaps, '
        'or Gramian Angular Fields, or Short-Time Fourier Transform spectrograms.',
        '<b>Option 4  --  Reduce lambda_KD:</b> If teacher is noisy, set lambda_KD very small (0.1) '
        'so teacher provides only a light regularization signal rather than dominating training.',
        '<b>Option 5  --  Ablate completely:</b> Set lambda_KD=0 and use student_baseline mode. '
        'If this matches or beats student_kd, the teacher is not helping for this dataset.',
        q_num='L4')

    items += qa(
        'What is the first experiment you would run to verify the whole pipeline?',
        '<b>End-to-end smoke test with synthetic data:</b>',
        '1. Generate synthetic CSV: python scripts/generate_synthetic.py',
        '2. Upload synthetic_multivariate.csv to Streamlit (it has known anomalies at specific time steps).',
        '3. Run full pipeline: preprocess -> teacher score -> train student_kd -> evaluate.',
        '4. Check: (a) Does the model achieve >0.5 F1 on the synthetic data? '
        '(b) Do the detected intervals match the injected anomaly positions? '
        '(c) Is the KD mode better than student_baseline on this controlled data?',
        'This smoke test verifies the complete data flow without needing real dataset downloads.',
        code='# Quick smoke test command sequence:\npython scripts/generate_synthetic.py\nstreamlit run app.py\n# Then in browser: Upload data/synthetic/synthetic_multivariate.csv',
        q_num='L5')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def section_M(story):
    story.append(chapter_header('M', 'Final Understanding  --  The Full Picture', HexColor('#1B5E20')))
    story.append(sp())
    items = []

    items += qa(
        'Can you explain the complete project in one minute?',
        'We have sensor data  --  time series of numbers from machines, bodies, or transactions. '
        'Sometimes these sensors show unusual readings: a credit card fraud, a patient freezing mid-walk, '
        'a compressor about to fail. We want to automatically detect these moments.',
        'Our system uses two AI models working together. The first is CLIP  --  a massive AI trained on '
        'hundreds of millions of photos and captions. We convert short chunks of sensor data into '
        'colorful pictures (heatmaps) and show them to CLIP, asking "normal or anomalous?" '
        'CLIP cannot run in real time  --  it\'s too big and slow.',
        'So we use CLIP to TEACH a second, tiny AI called a GRU. The GRU learns: '
        '(1) what anomalous sensor patterns look like, from CLIP\'s visual intuition; '
        '(2) exactly WHEN in each chunk the anomaly happens, from our labeled training data.',
        'Once trained, the GRU can run without CLIP. It is designed to be smaller and faster, '
        'but the size and latency advantage must be measured on the same hardware before making a claim. '
        'It outputs a score for each time step so detected spans can be shown on the signal.',
        q_num='M1')

    items += qa(
        'Can you explain it without technical jargon?',
        'Imagine you are training a young employee to identify defective products on a factory line. '
        'You could show them examples of defective items all day  --  but good defective items are rare, '
        'so you don\'t have many to show.',
        'Instead, you pair them with a very experienced quality inspector (the teacher) who has seen '
        'thousands of products and can say "this looks suspicious" vs "this looks fine" '
        'with nuance  --  not just yes/no but "I\'m 70% sure this is bad."',
        'The young employee (student) watches the expert\'s assessments AND sees the actual defect labels. '
        'Over time, they develop both the expert\'s intuition AND precise knowledge of specific defect patterns.',
        'Now the expert can retire. The young employee works alone on the factory floor, checking products '
        'thousands of times per second  --  something the expert could never do at scale.',
        'In our project: Expert = CLIP (powerful, slow, expensive). Young employee = GRU (small, fast, cheap). '
        'Products = chunks of sensor data. Defects = anomalies.',
        q_num='M2')

    items += qa(
        'What is the most important scientific hypothesis in this project?',
        'The hypothesis: <b>"A foundation VLM\'s visual understanding of "normal vs. anomalous" '
        'patterns, when transferred to sensor heatmaps via knowledge distillation, improves '
        'a recurrent student\'s anomaly detection beyond what ground-truth supervision alone achieves."</b>',
        'The experimental test: compare student_kd (with CLIP teaching) vs. student_baseline '
        '(same GRU, same labels, no CLIP teaching). If student_kd consistently scores higher across '
        'multiple datasets, the hypothesis is supported.',
        imp='The hypothesis can only be confirmed/denied by running actual experiments on real datasets. '
            'We have implemented the framework; the experiments remain to be executed.',
        q_num='M3')

    items += qa(
        'What are the three most important contributions of the project?',
        '<b>1. End-to-End Pipeline:</b> An implemented and interactive system '
        'covering data ingestion, preprocessing, VLM teacher scoring, GRU student distillation, '
        'evaluation, and visualization.',
        '<b>2. Multi-Task Loss for Distillation + Localization:</b> A novel combination of Hinton '
        'KD loss (window-level teacher knowledge), cross-entropy (window classification), and '
        'weighted BCE (timestep localization) in a single joint objective that trains the student '
        'to simultaneously detect AND localize anomalies from a window-only teacher.',
        '<b>3. Leakage-Aware Evaluation:</b> Time/group splits, train-only scaler fitting, '
        'validation-based thresholds, and a test suite are implemented. Run the suite and inspect '
        'the split outputs before claiming verification.',
        q_num='M4')

    items += qa(
        'What is the single biggest unresolved technical risk?',
        'The domain gap between CLIP\'s natural-photo pretraining and sensor heatmaps.',
        'If CLIP\'s p_anomaly scores correlate poorly with actual anomalies in sensor heatmaps '
        '(e.g., a spiky anomalous signal produces a heatmap that CLIP sees as perfectly normal), '
        'then the KD signal is noise  --  or worse, anti-correlated signal.',
        'In that case, student_kd would perform WORSE than student_baseline, and the core '
        'hypothesis of the project would be falsified.',
        'Mitigation being planned: learnable prompt tuning (CoOp) and multi-prompt ensemble.',
        warn='This must be honestly disclosed in the viva. We have not yet measured CLIP\'s accuracy '
             'on our heatmaps. It could be 50% (random) which would break the whole approach.',
        q_num='M5')

    items += qa(
        'If you had to demonstrate only one experiment to the TA, which would you choose?',
        '<b>End-to-end on synthetic data with ablation comparison.</b>',
        'Why this experiment:',
        '  >  Does not require downloading any real dataset (can run right now).',
        '  >  Demonstrates the COMPLETE pipeline: CSV -> preprocessing -> teacher -> student -> evaluation.',
        '  >  The synthetic anomalies are known exactly (we injected them with seed=42), '
        'so ground truth is perfectly reliable.',
        '  >  Compare the two current training modes (student_baseline and student_kd) and inspect teacher validation diagnostics.',
        '  >  Runtime depends on CLIP model download, device, and dataset size.',
        code='# Commands to run this demo:\npython scripts/generate_synthetic.py\nstreamlit run app.py\n# Upload: data/synthetic/synthetic_multivariate.csv\n# Select feature columns, label column\n# Run student_baseline and student_kd, compare on Page 6',
        q_num='M6')

    for item in items:
        story.append(item)
    story.append(PageBreak())


def build_impl_map(story):
    story.append(chapter_header('MAP', 'Implementation Map  --  Every File and Function', NAVY))
    story.append(sp())
    story.append(P(
        'This map tells you exactly WHERE in the code each part of the pipeline is implemented. '
        'For every viva question about implementation details, you can point to the specific file and function.',
        'body'))
    story.append(sp())

    impl_data = [
        ['Pipeline Stage', 'File', 'Key Function/Class', 'What It Does'],
        ['CSV loading', 'src/data/loader.py', 'load_csv(path)',
         'Detects separator, reads CSV, returns (DataFrame, DataQualityReport)'],
        ['Data quality report', 'src/data/loader.py', 'DataQualityReport dataclass',
         'Missing counts, dtypes, duplicates, sample rows'],
        ['Dataset dispatch', 'src/data/loader.py', 'load_dataset(name, config)',
         'Calls creditcard.py/daphnet.py/etc. based on dataset name'],
        ['Chronological sort', 'src/data/preprocessing.py', 'sort_chronologically(df, col)',
         'Sorts by timestamp, attempts datetime parsing'],
        ['Duplicate removal', 'src/data/preprocessing.py', 'handle_duplicates(df, col, policy)',
         'Policies: first/last/mean on duplicate timestamps'],
        ['Missing value fill', 'src/data/preprocessing.py', 'fill_missing(arr, strategy)',
         'interpolate/ffill/bfill/mean, returns filled float32 array'],
        ['Scaler fit', 'src/data/preprocessing.py', 'fit_scaler(X_train, type)',
         'Returns fitted sklearn StandardScaler/MinMaxScaler'],
        ['Scaler apply', 'src/data/preprocessing.py', 'transform_scaler(X, scaler)',
         'Applies fitted scaler to X'],
        ['Train/Val/Test split', 'src/data/preprocessing.py', 'chronological_split(series, ratios)',
         '70/15/15 chronological or subject-based split'],
        ['Sliding windows', 'src/data/preprocessing.py', 'make_windows(values, y_step, ...)',
         'Creates (N,L,F) X_windows, (N,) y_window, (N,L) y_step_windows with gap checks'],
        ['Leakage check', 'src/data/preprocessing.py', 'check_leakage(ts_train, ts_val, ts_test)',
         'Asserts zero timestamp overlap between splits'],
        ['Synthetic anomalies', 'src/data/synthetic_labels.py', 'inject_anomalies(values, cfg, seed)',
         '4 anomaly types injected independently after split (train/val/test)'],
        ['Dropout labels', 'src/data/synthetic_labels.py', 'dropout_labels(values, missing_code)',
         'Labels long runs of missing_code as anomalous'],
        ['Credit Card loader', 'src/data/datasets/creditcard.py', 'load(config)',
         'Loads CSV, V1-V28+Amount features, Class label'],
        ['Daphnet loader', 'src/data/datasets/daphnet.py', 'load(config)',
         'Loads .txt files, splits at annotation=0 gaps, annotation=2 is anomaly'],
        ['Air Quality loader', 'src/data/datasets/airquality.py', 'load(config)',
         'Parses semicolons+decimal commas, injects synthetic labels'],
        ['3W Oil loader', 'src/data/datasets/oil3w.py', 'load(config)',
         'Walks CSV files, class!=0 is anomaly, handles all-NaN columns'],
        ['MetroPT loader', 'src/data/datasets/metropt.py', 'load(config)',
         'Resamples to 1-min, converts failure_windows to step labels'],
        ['Heatmap rendering', 'src/teacher/image_gen.py', 'window_to_image(window, mode)',
         '(L,F) array -> 224x224 PIL Image (heatmap or lineplot)'],
        ['Batch image render', 'src/teacher/image_gen.py', 'batch_to_images(windows, ...)',
         'Applies window_to_image to all windows with progress bar'],
        ['Score caching', 'src/teacher/image_gen.py', 'save/load_image_cache()',
         'Pickles scores array to/from disk'],
        ['CLIP teacher load', 'src/teacher/clip_teacher.py', 'CLIPTeacher.load_model()',
         'Loads clip-vit-base-patch32, freezes all params, verifies frozen'],
        ['Single window score', 'src/teacher/clip_teacher.py', 'CLIPTeacher.score_window(img, p1, p2)',
         'Returns [p_normal, p_anomaly] via cosine similarity + softmax'],
        ['Batch scoring', 'src/teacher/clip_teacher.py', 'CLIPTeacher.score_batch(imgs, p1, p2)',
         'Returns (N,2) array of soft probs'],
        ['Cache + score', 'src/teacher/clip_teacher.py', 'CLIPTeacher.cache_and_score(...)',
         'Check cache, else score, save. Returns (N,2) teacher probs'],
        ['GRU student model', 'src/student/model.py', 'GRUStudent class',
         '2-layer GRU + window_head + step_head. Forward returns (window_logits, step_logits)'],
        ['Model config', 'src/student/model.py', 'ModelConfig dataclass',
         'input_size, hidden_size=64, num_layers=2, dropout=0.3, seq_len=128'],
        ['Model predict', 'src/student/model.py', 'GRUStudent.predict_proba(x)',
         'Returns (window_proba, step_scores) as numpy arrays, no gradient'],
        ['Localization loss', 'src/student/losses.py', 'LocalizationLoss(pos_weight=10)',
         'Weighted BCE on step_logits vs y_step'],
        ['Window CE loss', 'src/student/losses.py', 'WindowClassificationLoss()',
         'Cross-entropy on window_logits vs y_window'],
        ['KD loss', 'src/student/losses.py', 'KnowledgeDistillationLoss(T=2)',
         'T^2 x KL(p_T(T) || p_S(T)), teacher detached'],
        ['Total loss', 'src/student/losses.py', 'TotalLoss.forward(...)',
         'L_loc + betaxL_win + lambda_KDxL_KD. Returns total + dict breakdown'],
        ['Training loop', 'src/training/trainer.py', 'Trainer.fit(...)',
         'Full training loop: DataLoader, train_epoch, validate, early stop, LR sched, best restore'],
        ['One epoch', 'src/training/trainer.py', 'Trainer.train_epoch(loader)',
         'Forward pass, loss, backward, clip grad, optimizer step. Returns avg loss dict'],
        ['Validation', 'src/training/trainer.py', 'Trainer.validate(loader)',
         'No grad, forward pass only. Returns avg loss dict'],
        ['Get predictions', 'src/training/trainer.py', 'Trainer.get_predictions(X)',
         'Returns (window_proba, step_scores) for all test windows'],
        ['Save checkpoint', 'src/training/checkpoint.py', 'save_checkpoint(model, config, ...)',
         'Saves model weights + config + scaler + training result to .pt file'],
        ['Load checkpoint', 'src/training/checkpoint.py', 'load_checkpoint(path, model)',
         'Loads and validates checkpoint schema'],
        ['Window metrics', 'src/evaluation/metrics.py', 'compute_window_metrics(y_true, scores)',
         'Precision, Recall, F1, PR-AUC, ROC-AUC at given threshold'],
        ['Step metrics', 'src/evaluation/metrics.py', 'compute_step_metrics(y_true, scores)',
         'Same metrics at timestep level'],
        ['Threshold tuning', 'src/evaluation/threshold.py', 'tune_threshold(y_val, scores)',
         '100-point sweep [0.01, 0.99], maximize F1. Returns best_theta and info dict'],
        ['Score aggregation', 'src/evaluation/localization.py', 'aggregate_step_scores(scores, starts)',
         'Averages overlapping window step scores into per-timestep scores'],
        ['IoU per event', 'src/evaluation/localization.py', 'iou_per_anomaly(y_true, y_pred)',
         'Returns IoU for each true anomaly segment'],
        ['Mean IoU', 'src/evaluation/localization.py', 'mean_iou(y_true, y_pred)',
         'Average IoU over all true events'],
        ['Point Adjustment', 'src/evaluation/localization.py', 'point_adjustment(y_true, y_pred)',
         'If any pred in true segment, fill whole segment. Returns adjusted y_pred'],
        ['Event F1', 'src/evaluation/localization.py', 'event_f1(y_true, y_pred)',
         'Overlap-based event-level F1 score'],
        ['Plots', 'src/visualization/plots.py', 'plot_anomaly_timeline(), plot_pr_curve()',
         'Interactive Plotly charts for dashboard'],
        ['Report generation', 'src/visualization/report.py', 'generate_report()',
         'Compiles PDF/HTML summary report'],
        ['Dashboard entry', 'app.py', 'main script',
         'Streamlit config, CSS, session state init, sidebar nav, page routing'],
        ['Page 2 (Upload)', 'src/ui/page2_upload.py', 'show()',
         'File uploader, column selector, DataQualityReport display, data preview'],
        ['Page 3 (Preproc)', 'src/ui/page3_preprocessing.py', 'show()',
         'Preprocessing config, run pipeline, window stats, leakage check result'],
        ['Page 4 (Teacher)', 'src/ui/page4_teacher.py', 'show()',
         'Prompt input, run CLIP scoring, heatmap preview, teacher prob histogram'],
        ['Page 5 (Training)', 'src/ui/page5_training.py', 'show()',
         'Hyperparameter inputs, run training, live loss curves, training result summary'],
        ['Page 6 (Eval)', 'src/ui/page6_evaluation.py', 'show()',
         'Three-mode comparison, PR/ROC curves, confusion matrix, threshold sweep'],
        ['Page 7 (Inference)', 'src/ui/page7_inference.py', 'show()',
         'Run test inference, anomaly timeline, interval table, download CSV/PDF'],
    ]
    story.append(build_table(impl_data, [3.5*cm, 4*cm, 4.5*cm, 4.3*cm]))
    story.append(P('Complete Implementation Map  --  All files, functions, and their roles.', 'caption'))
    story.append(sp())
    story.append(HR(RED, 1.5))
    story.append(P('<b>Questions that CANNOT be answered from current code (data not available):</b>', 'warn'))
    cannot = [
        'Actual F1/AUPRC scores on real datasets (no dataset files downloaded yet)',
        'Whether CLIP teacher signal is useful on sensor heatmaps (not empirically tested)',
        'Exact training wall-clock time on real datasets (no saved benchmark runs)',
        'Whether student_kd beats student_baseline in practice (hypothesis not yet verified)',
        'CLIP inference latency on specific hardware (not measured)',
    ]
    for c in cannot:
        story.append(P(f'  X  {c}', 'bullet'))
    story.append(sp())
    story.append(P(
        '<b>Honest summary for the viva:</b> The implementation and test suite are present; the previous snapshot had 70 tests pass. '
        'Rerun after the final KD temperature correction and run an end-to-end benchmark before claiming the current pipeline has been verified. '
        'The dataset choices are TA-approved; download those files and complete the real dataset experiments.',
        'imp'))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BUILD
# ═══════════════════════════════════════════════════════════════════════════════

def build_viva_pdf(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=LMAR, rightMargin=RMAR,
        topMargin=TMAR + 0.8*cm, bottomMargin=BMAR + 0.5*cm,
        title='IE643 VLM-KD  --  Viva Preparation Guide',
        author='IE643 Course Project',
    )

    story = []
    build_cover(story)
    build_quick_reference(story)
    section_A(story)
    section_B(story)
    section_C(story)
    section_D(story)
    section_E(story)
    section_FG(story)
    section_H(story)
    section_IJ(story)
    section_K(story)
    section_L(story)
    section_M(story)
    build_impl_map(story)

    doc.build(
        story,
        onFirstPage=on_page,
        onLaterPages=on_page,
    )
    print(f'[OK] Viva guide saved to: {output_path}')
    return output_path


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(__file__), 'IE643_Viva_Preparation_Guide.pdf')
    build_viva_pdf(out)

