"""
IE643 Course Project Report Generator
Produces a full academic PDF using ReportLab.
Run: python report/generate_pdf.py
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
import os

# ── Color palette ─────────────────────────────────────────────────────────────
IITB_BLUE   = HexColor('#002147')
ACCENT      = HexColor('#1A6BAA')
LIGHT_BLUE  = HexColor('#D6E8F8')
LIGHT_GRAY  = HexColor('#F5F5F5')
MID_GRAY    = HexColor('#888888')
DARK_GRAY   = HexColor('#333333')
TABLE_HEAD  = HexColor('#1A6BAA')
TABLE_ALT   = HexColor('#EEF5FB')

PAGE_W, PAGE_H = A4
L_MARGIN = R_MARGIN = 2.0 * cm
T_MARGIN = B_MARGIN = 2.0 * cm
COL_GAP  = 0.5 * cm
COL_W    = (PAGE_W - L_MARGIN - R_MARGIN - COL_GAP) / 2


# ── Custom horizontal rule ─────────────────────────────────────────────────────
class ColorHR(Flowable):
    def __init__(self, color=ACCENT, width=None, thickness=1.5):
        super().__init__()
        self.color = color
        self.line_width = width
        self.thickness = thickness

    def wrap(self, avail_w, avail_h):
        self.line_width = self.line_width or avail_w
        return self.line_width, self.thickness + 2

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.line_width, 0)


# ── Styles ─────────────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    def s(name, **kw):
        d = dict(fontName='Times-Roman', fontSize=10, leading=14,
                 textColor=DARK_GRAY, alignment=TA_JUSTIFY, spaceAfter=4)
        d.update(kw)
        return ParagraphStyle(name=name, **d)

    styles = {
        'title': s('title',
                   fontName='Times-Bold', fontSize=18, leading=22,
                   alignment=TA_CENTER, textColor=IITB_BLUE, spaceAfter=4),
        'subtitle': s('subtitle',
                      fontName='Times-Italic', fontSize=11, leading=14,
                      alignment=TA_CENTER, textColor=ACCENT, spaceAfter=3),
        'author': s('author',
                    fontName='Times-Bold', fontSize=10.5, leading=14,
                    alignment=TA_CENTER, textColor=DARK_GRAY, spaceAfter=2),
        'meta': s('meta',
                  fontName='Times-Italic', fontSize=9, leading=13,
                  alignment=TA_CENTER, textColor=MID_GRAY, spaceAfter=8),
        'abstract_heading': s('abstract_heading',
                              fontName='Times-Bold', fontSize=10,
                              alignment=TA_CENTER, textColor=ACCENT, spaceAfter=3),
        'abstract': s('abstract',
                      fontName='Times-Italic', fontSize=9.5, leading=13.5,
                      alignment=TA_JUSTIFY, leftIndent=0.8*cm, rightIndent=0.8*cm,
                      spaceAfter=8),
        'h1': s('h1',
                fontName='Times-Bold', fontSize=12, leading=16,
                textColor=IITB_BLUE, spaceBefore=10, spaceAfter=4,
                alignment=TA_LEFT),
        'h2': s('h2',
                fontName='Times-Bold', fontSize=10.5, leading=14,
                textColor=ACCENT, spaceBefore=6, spaceAfter=3,
                alignment=TA_LEFT),
        'h3': s('h3',
                fontName='Times-Bold-Italic', fontSize=10, leading=13,
                textColor=DARK_GRAY, spaceBefore=4, spaceAfter=2,
                alignment=TA_LEFT),
        'body': s('body'),
        'body_tight': s('body_tight', spaceAfter=2, leading=13),
        'bullet': s('bullet', leftIndent=0.5*cm, spaceBefore=1, spaceAfter=1,
                    fontSize=9.5, leading=13),
        'caption': s('caption',
                     fontName='Times-Italic', fontSize=8.5, leading=12,
                     alignment=TA_CENTER, textColor=MID_GRAY, spaceAfter=4),
        'code': s('code',
                  fontName='Courier', fontSize=8.5, leading=12,
                  textColor=HexColor('#1a1a1a'), alignment=TA_LEFT,
                  leftIndent=0.4*cm, rightIndent=0.4*cm, spaceAfter=4),
        'table_head': s('table_head',
                        fontName='Helvetica-Bold', fontSize=8.5, leading=11,
                        textColor=white, alignment=TA_CENTER),
        'table_cell': s('table_cell',
                        fontName='Times-Roman', fontSize=8.5, leading=11,
                        textColor=DARK_GRAY, alignment=TA_CENTER),
        'table_cell_l': s('table_cell_l',
                          fontName='Times-Roman', fontSize=8.5, leading=11,
                          textColor=DARK_GRAY, alignment=TA_LEFT),
        'ref': s('ref',
                 fontName='Times-Roman', fontSize=8.5, leading=12,
                 alignment=TA_LEFT, leftIndent=0.6*cm, firstLineIndent=-0.6*cm,
                 spaceAfter=5),
        'footer': s('footer',
                    fontName='Times-Italic', fontSize=7.5, leading=10,
                    textColor=MID_GRAY, alignment=TA_CENTER),
    }
    return styles


# ── Header / Footer canvas callback ───────────────────────────────────────────
def on_page(canvas, doc, styles):
    canvas.saveState()
    w, h = A4
    # Header line
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(0.6)
    canvas.line(L_MARGIN, h - T_MARGIN + 4, w - R_MARGIN, h - T_MARGIN + 4)
    # Header text
    canvas.setFont('Times-Italic', 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(L_MARGIN, h - T_MARGIN + 6,
                      'IE 643 - Deep Learning for Computer Vision: Course Project')
    canvas.drawRightString(w - R_MARGIN, h - T_MARGIN + 6,
                           'VLM Teacher-RNN Student Knowledge Distillation')
    # Footer line
    canvas.line(L_MARGIN, B_MARGIN - 4, w - R_MARGIN, B_MARGIN - 4)
    canvas.setFont('Times-Italic', 7.5)
    canvas.drawCentredString(w / 2, B_MARGIN - 12,
                             f'Page {doc.page}')
    canvas.restoreState()


# ── Table builder helpers ──────────────────────────────────────────────────────
def build_table(data, col_widths, styles_dict, span_top=True):
    ST = styles_dict
    rows = []
    for i, row in enumerate(data):
        cells = []
        for j, cell in enumerate(row):
            if i == 0:
                cells.append(Paragraph(str(cell), ST['table_head']))
            else:
                style = ST['table_cell_l'] if j == 0 else ST['table_cell']
                cells.append(Paragraph(str(cell), style))
        rows.append(cells)

    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEAD),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, TABLE_ALT]),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#BBBBBB')),
        ('LINEBELOW', (0, 0), (-1, 0), 1.2, ACCENT),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ])
    t = Table(rows, colWidths=col_widths)
    t.setStyle(ts)
    return t


def section_header(number, title, styles_dict):
    return Paragraph(f'{number}. {title}'.upper(), styles_dict['h1'])


def subsection_header(number, title, styles_dict):
    return Paragraph(f'{number}  {title}', styles_dict['h2'])


def P(text, style):
    return Paragraph(text, style)


def bullet_list(items, style, bullet='-'):
    return [P(f'<b>{bullet}</b>  {item}', style) for item in items]


# ── Main report builder ────────────────────────────────────────────────────────
def build_report(output_path):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=L_MARGIN, rightMargin=R_MARGIN,
        topMargin=T_MARGIN + 0.6*cm, bottomMargin=B_MARGIN + 0.4*cm,
        title='IE643 Course Project Report - VLM-KD Anomaly Detection',
        author='IE643 Course Project Team',
        subject='IE 643 Deep Learning Course Project',
    )

    ST = build_styles()
    story = []

    def sp(n=1):
        story.append(Spacer(1, n * 0.25 * cm))

    def hr(color=ACCENT, thick=1.0):
        story.append(ColorHR(color=color, thickness=thick))
        sp(0.5)

    # ── Title block ────────────────────────────────────────────────────────────
    story.append(P('Vision-Language Model Guided Knowledge Distillation<br/>'
                   'for Time-Series Anomaly Detection', ST['title']))
    sp(0.5)
    story.append(P('IE 643 - Deep Learning for Computer Vision: Course Project Intensive Assessment',
                   ST['subtitle']))
    sp(0.5)
    story.append(P('IE643 Course Project Team  |  Course project implementation',
                   ST['author']))
    story.append(P('November 2026', ST['meta']))
    hr(thick=1.5)

    # ── Abstract ──────────────────────────────────────────────────────────────
    story.append(P('ABSTRACT', ST['abstract_heading']))
    story.append(P(
        'Detecting anomalies in multivariate time-series data is a fundamental challenge in industrial '
        'safety, healthcare monitoring, and financial fraud detection. Large Vision-Language Models (VLMs) '
        'such as CLIP exhibit impressive zero-shot reasoning across diverse visual domains; however, their '
        'large model size motivates transferring a teacher signal into a compact recurrent student. '
        'We propose a knowledge distillation (KD) framework in which a <i>frozen</i> CLIP ViT-B/32 teacher '
        'provides soft anomaly priors for training a compact two-layer Gated Recurrent Unit (GRU) student. '
        'Time-series sliding windows are rendered as 224 x 224 heatmap images and scored via contrastive '
        'text-image cosine similarity; the resulting soft probabilities supervise the student through a '
        'Hinton temperature-scaled KL-divergence loss (T=2.0, lambda_KD=0.5) combined with a '
        'step-level weighted BCE localization loss (w<sub>+</sub>=10) and a window-level cross-entropy '
        'classification loss. The multi-task objective enables simultaneous window-level anomaly '
        'classification and fine-grained timestep localization. Evaluated on five heterogeneous benchmark '
        'datasets - Credit Card Fraud, Daphnet Freezing of Gait, UCI Air Quality, 3W Oil-Well, and '
        'MetroPT-3. The repository contains a seven-page Streamlit dashboard. Benchmark datasets '
        'are not included in this workspace and no measured benchmark results are reported here.',
        ST['abstract']))
    hr()

    # ── 1. Introduction ───────────────────────────────────────────────────────
    story.append(section_header(1, 'Introduction', ST))
    sp(0.3)
    story.append(P(
        'Anomaly detection in time-series is a pervasive problem: credit-card fraud accounts for '
        'fewer than 0.2% of transactions, freezing-of-gait (FoG) episodes represent a small fraction '
        'of a Parkinson\'s patient\'s gait cycle, and compressor failures are rare yet catastrophic. '
        'This extreme class imbalance and diverse anomaly morphology - point spikes, level shifts, '
        'variance bursts, gradual drift - make supervised detection difficult when ground-truth labels '
        'are expensive and scarce. Classical statistical methods (3-sigma, CUSUM) lack contextual '
        'reasoning; deep reconstruction models (LSTM-AE) suffer from the anomaly-collapse problem; '
        'and fully supervised approaches demand large labeled datasets rarely available in industry.',
        ST['body']))
    sp(0.3)
    story.append(P(
        'Foundation models such as CLIP [1] have demonstrated that joint image-text pre-training on '
        '400M internet image-caption pairs yields visual representations that transfer well to new '
        'domains without task-specific training. Recent work has begun exploiting this capability for '
        'time-series by rendering sequences as images [2, 3]. However, running CLIP at inference time '
        'imposes unacceptable latency on embedded systems, and the model has no built-in temporal '
        'recurrence to capture sequential dependencies across windows. Gated Recurrent Units (GRUs) [4] '
        'offer compact temporal modeling and are naturally '
        'suited to streaming sensor data. Knowledge distillation (KD) [5] bridges these two regimes: '
        'the student learns from the teacher\'s soft probability outputs, capturing semantic "dark '
        'knowledge" that hard labels alone cannot convey.',
        ST['body']))
    sp(0.3)
    story.append(P('<b>Our contributions:</b>', ST['body']))
    items = [
        'A complete teacher-student distillation pipeline: frozen CLIP scores sliding window heatmaps, '
        'and GRU student learns via temperature-scaled KL divergence.',
        'A novel <b>multi-task loss</b> combining step-level localization (weighted BCE, w<sub>+</sub>=10), '
        'window-level classification (CE), and knowledge distillation (T^2 * KL, T=2, lambda_KD=0.5).',
        '<b>Strict leakage prevention</b>: chronological splits, train-only scaler/threshold fitting, '
        'and gap-aware windowing ensure no future information contaminates training.',
        'An <b>interactive 7-page Streamlit dashboard</b> covering the full pipeline from CSV upload to '
        'anomaly timeline export.',
        'Support for <b>five heterogeneous benchmark datasets</b> from healthcare, finance, environmental '
        'monitoring, and heavy industry.',
    ]
    for item in items:
        story.append(P(f'  -  {item}', ST['bullet']))
        sp(0.2)
    hr()

    # ── 2. Workflow ───────────────────────────────────────────────────────────
    story.append(section_header(2, 'Workflow', ST))
    sp(0.3)
    workflow_data = [
        ['Week', 'Activity', 'Deliverable'],
        ['1', 'Literature survey: VLMs, KD, RNN-based anomaly detection, benchmark datasets',
         'Reading list, problem scoping'],
        ['2', 'Data pipeline: loaders for all 5 datasets, preprocessing, windowing, synthetic labels',
         'src/data/ modules, test fixtures'],
        ['3', 'Teacher module: CLIP inference, heatmap image renderer, score caching',
         'src/teacher/ modules'],
        ['4', 'Student model (GRU dual-head) and multi-task loss functions',
         'src/student/ modules'],
        ['5', 'Training loop, early stopping, LR scheduler, evaluation metrics, threshold tuning',
         'src/training/, src/evaluation/'],
        ['6', 'Streamlit dashboard (7 pages), test suite, documentation, report',
         'app.py, src/ui/, tests/'],
    ]
    col_w = [1.3*cm, 8.0*cm, 5.0*cm]
    story.append(build_table(workflow_data, col_w, ST))
    sp(0.3)
    hr()

    # ── 3. Related Work ───────────────────────────────────────────────────────
    story.append(section_header(3, 'Related Work', ST))
    sp(0.3)

    story.append(subsection_header('3.1', 'Knowledge Distillation', ST))
    story.append(P(
        'Hinton et al. [5] introduced knowledge distillation showing that a compact student trained '
        'on a teacher\'s <i>soft</i> probability distributions generalises better than one trained on '
        'hard labels. The core insight is that soft targets encode inter-class similarity: in '
        'time-series, a "nearly anomalous" window being assigned p<sub>anomaly</sub>=0.7 carries more '
        'information than a binary label. The KD loss scales the KL divergence by T^2 to maintain '
        'gradient magnitudes as T increases: '
        'L_KD = T^2 * KL(p_T(T) || p_S(T)); both teacher and student are softened by T. '
        'We implement this exactly, with T=2.0 and gradient detachment on teacher targets.',
        ST['body']))
    sp(0.3)

    story.append(subsection_header('3.2', 'VLMs for Time-Series Analysis', ST))
    story.append(P(
        'Sun et al. [2] demonstrated that rendering time-series windows as 2D images and applying '
        'CLIP zero-shot classification yields non-trivial anomaly detection without any training data. '
        'Jin et al. [3] ("Time-LLM") reprogrammed large language models for time-series forecasting '
        'by prepending time-series patches with text prototypes. Both works confirm the transferability '
        'of foundation model representations to temporal domains. Our work is complementary: rather than '
        'using VLMs directly at inference time, we extract their soft priors into a lightweight RNN '
        'student that additionally learns temporal recurrence dynamics unavailable to frame-by-frame '
        'VLMs, and timestep-level localization masks unavailable from window-level VLM classification.',
        ST['body']))
    sp(0.3)

    story.append(subsection_header('3.3', 'RNN-based Temporal Anomaly Detection', ST))
    story.append(P(
        'Malhotra et al. [6] applied stacked LSTM auto-encoders for unsupervised anomaly detection '
        'via reconstruction error, establishing RNNs as a strong temporal baseline. Hundman et al. [7] '
        'applied LSTMs to NASA spacecraft telemetry with non-parametric dynamic thresholding, showing '
        'the critical importance of validation-based threshold selection to avoid overfitting. GRUs [4] '
        'merge the update and reset gates of the LSTM cell, reducing parameter count by ~25% while '
        'achieving comparable performance - a desirable property for edge deployment.',
        ST['body']))
    hr()

    # ── 4. Methods ────────────────────────────────────────────────────────────
    story.append(section_header(4, 'Methods and Approaches', ST))
    sp(0.3)

    story.append(subsection_header('4.1', 'System Overview', ST))
    story.append(P(
        'The pipeline has five stages: (1) data ingestion and preprocessing; '
        '(2) sliding window segmentation; (3) teacher CLIP scoring; (4) student GRU training; '
        '(5) threshold-tuned inference and evaluation. Figure 1 summarises the data flow.',
        ST['body']))
    sp(0.3)

    # Architecture diagram as table
    arch_data = [
        ['Stage', 'Component', 'Output'],
        ['1. Ingest', 'CSV loader -> sort -> dedup -> split -> scale', 'X_train, X_val, X_test (float32)'],
        ['2. Window', 'Sliding window (L=128, S=32, gap-aware)', 'X_windows (N,L,F), y_window (N,), y_step (N,L)'],
        ['3. Teacher', 'Window -> Heatmap (224 x 224) -> CLIP ViT-B/32 -> softmax', 'p_T (N,2): [p_normal, p_anomaly]'],
        ['4. Student', 'GRU(H=64, 2-layer) -> dual head -> multi-task loss', 'window_logits (B,2), step_logits (B,L)'],
        ['5. Infer', 'Sigmoid/softmax -> val-tuned threshold -> interval detection', 'Anomaly timeline, AUPRC, F1, IoU'],
    ]
    story.append(build_table(arch_data, [1.5*cm, 5.8*cm, 7.0*cm], ST))
    story.append(P('Table 1: End-to-end pipeline stages.', ST['caption']))
    sp(0.3)

    story.append(subsection_header('4.2', 'Teacher: Frozen CLIP ViT-B/32', ST))
    story.append(P(
        'We use <tt>openai/clip-vit-base-patch32</tt> (151M parameters) from Hugging Face Transformers. '
        'The model is loaded in eval() mode with all parameters frozen (requires_grad=False, '
        'verified by assertion). Each window X_w has shape (L,F):',
        ST['body']))
    steps = [
        '<b>Render heatmap:</b> Matplotlib Agg backend, Viridis colormap, features on Y-axis, '
        'time on X-axis, 100 DPI -> PIL Image resized to 224 x 224 (LANCZOS). No labels or '
        'ground-truth are visible in the rendered image.',
        '<b>Encode:</b> CLIPProcessor tokenizes both prompts and encodes the image. '
        'Prompts: <i>normal</i> = "A normal regular time series signal without anomalies"; '
        '<i>anomaly</i> = "An anomalous irregular time series signal with spikes or distortions".',
        '<b>Score:</b> logits_per_image (1x2) -> softmax -> p_T = [p_normal, p_anomaly] '
        '(two probabilities that sum to 1).',
        '<b>Cache:</b> Scores pickled to disk; up to 5,000 windows scored per dataset run '
        'to respect memory constraints.',
    ]
    for step in steps:
        story.append(P(f'  -  {step}', ST['bullet']))
        sp(0.15)
    sp(0.2)

    story.append(subsection_header('4.3', 'Student: GRU with Dual Prediction Heads', ST))
    story.append(P(
        'The student model GRUStudent implements a multi-layer GRU backbone with two task-specific '
        'prediction heads. The forward pass proceeds as follows:',
        ST['body']))

    arch2_data = [
        ['Component', 'Shape', 'Details'],
        ['Input', '(B, L, F)', 'Batch x Window-length x Features'],
        ['GRU backbone', '(B, L, 64)', '2 layers, H=64, dropout=0.3 (between layers), unidirectional'],
        ['Mean pooling', '(B, 64)', 'torch.mean(out, dim=1) across time steps'],
        ['Window head', '(B, 2)', 'Linear(64->64) -> ReLU -> Dropout(0.3) -> Linear(64->2)'],
        ['Step head', '(B, L)', 'Linear(64->32) -> ReLU -> Linear(32->1) -> squeeze(-1)'],
        ['Window prob', '(B, 2)', 'softmax(window_logits): [p_normal, p_anomaly]'],
        ['Step prob', '(B, L)', 'sigmoid(step_logits): step-level anomaly score in [0,1]'],
    ]
    story.append(build_table(arch2_data, [2.8*cm, 2.5*cm, 9.0*cm], ST))
    story.append(P('Table 2: GRUStudent architecture. The application computes the parameter count for the selected configuration.', ST['caption']))
    sp(0.3)

    story.append(subsection_header('4.4', 'Multi-Task Loss Function', ST))
    story.append(P(
        'Three losses are combined into one multi-task objective:', ST['body']))

    loss_data = [
        ['Loss Term', 'Formula', 'Config Value'],
        ['Localization (BCE)',
         'L_loc = weighted binary cross entropy over all batch steps',
         'w+ = 10.0'],
        ['Window Classification (CE)',
         'L_win = cross entropy over window labels',
         'beta = 1.0'],
        ['Knowledge Distillation (KL)',
         'L_KD = T^2 * KL(p_T(T) || p_S(T)); p_T detached',
         'T = 2.0, lambda_KD = 0.5'],
        ['Total Loss', 'L_total = L_loc + beta * L_win + lambda_KD * L_KD', '-'],
    ]
    story.append(build_table(loss_data, [3.0*cm, 8.5*cm, 2.8*cm], ST))
    story.append(P('Table 3: Multi-task loss components and configured coefficients.', ST['caption']))
    sp(0.2)
    story.append(P(
        'The positive class weight w<sub>+</sub>=10 amplifies the gradient contribution of anomalous '
        'steps to compensate for the typical 1-5% anomaly rate in real datasets. The KD temperature '
        'T=2 softens both the teacher and student distributions, encouraging the student to preserve '
        'the teacher\'s inter-class probability structure rather than collapsing to extreme predictions.',
        ST['body']))
    sp(0.3)

    story.append(subsection_header('4.5', 'Training Modes (Ablation)', ST))
    modes_data = [
        ['Mode', 'lambda_KD', 'Description'],
        ['student_baseline', '0.0', 'GRU trained with L_loc + L_win only; no knowledge distillation'],
        ['student_kd', '0.5', 'Full pipeline: L_loc + beta*L_win + lambda_KD*L_KD (default)'],
    ]
    story.append(build_table(modes_data, [3.5*cm, 1.5*cm, 9.3*cm], ST))
    story.append(P('Table 4: Training mode ablation configurations.', ST['caption']))
    hr()

    # ── 5. Data ───────────────────────────────────────────────────────────────
    story.append(section_header(5, 'Data', ST))
    sp(0.3)

    dataset_data = [
        ['Dataset', 'Domain', 'Samples', 'Features', 'Anomaly %', 'Window', 'Split By'],
        ['Credit Card\n(Kaggle)', 'Fraud\nDetection', '284,807', '29\n(V1-V28 + Amount)', '<0.2%', 'L=64, S=32', 'Chronological\n60/20/20'],
        ['Daphnet FoG\n(UCI #245)', 'Wearable\nHealthcare', '10 subjects,\n64 Hz', '9 accel.\nchannels', '~10%\n(FoG events)', 'L=256, S=64', 'Subject-based'],
        ['Air Quality\n(UCI #360)', 'Environmental\nIoT', '9,358 hrs.', '13 chemical\nsensors', 'Synthetic\n(semi-injected)', 'L=48, S=12', 'Chronological'],
        ['3W Oil Well\n(Petrobras)', 'Industrial\nOil & Gas', '1,984 instances\n(8 event types)', '8 pressure/\ntemp channels', '~15-30%\nper instance', 'L=120, S=60', 'Instance-based'],
        ['MetroPT-3\n(Zenodo)', 'Rail\nCompressor', 'Continuous,\n~6 months', '7 analog +\n8 digital', '4 failure\nevents', 'L=120, S=60', 'Event-based'],
    ]
    col_w = [2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 1.8*cm, 2.0*cm, 2.5*cm]
    story.append(build_table(dataset_data, col_w, ST))
    story.append(P('Table 5: Dataset statistics and windowing configuration.', ST['caption']))
    sp(0.3)

    story.append(subsection_header('5.1', 'Preprocessing Pipeline (Leakage-Free)', ST))
    preproc_steps = [
        '<b>Chronological sort</b> by timestamp (datetime parsing attempted automatically).',
        '<b>Duplicate removal</b> using "first" occurrence policy on duplicate timestamps.',
        '<b>Train/Val/Test split</b> (default 70/15/15) <i>before</i> any statistics computed.',
        '<b>Missing value imputation</b>: linear interpolation fitted on training split, applied '
        'identically to val and test. Columns with >50% missing are dropped.',
        '<b>Standard scaling</b>: StandardScaler fitted on training windows only; transform '
        'applied to all splits.',
        '<b>Gap-aware sliding windows</b>: Windows never cross split boundaries or data '
        'discontinuity indices. Window label = 1 if <i>any</i> timestep within is anomalous (configurable).',
        '<b>Leakage assertion</b>: Explicit assertion of zero timestamp overlap between splits.',
    ]
    for step in preproc_steps:
        story.append(P(f'  -  {step}', ST['bullet']))
        sp(0.15)
    sp(0.2)

    story.append(subsection_header('5.2', 'Synthetic Anomaly Injection (Air Quality)', ST))
    story.append(P(
        'The Air Quality dataset has no ground-truth labels. We apply four reproducible '
        '(seed=42) anomaly injection strategies to the val and test portions only:',
        ST['body']))
    synth_data = [
        ['Type', 'Parameters', 'Rate/Probability'],
        ['Point spikes', 'Magnitude = N(0,1) x 5 x std, random sign; applied to random steps',
         'Rate = 0.002 x T'],
        ['Level shifts', 'Shift = +/-3 x std applied to random segment of duration in [8,48) steps',
         'Prob = 0.05'],
        ['Drift (ramp)', 'Linear ramp 0 to +/-3.5 x std over segment in [16,64) steps',
         'Prob = 0.03'],
        ['Noise bursts', 'Additive N(0, 2.5 x std) over segment in [10,36) steps',
         'Prob = 0.04'],
    ]
    story.append(build_table(synth_data, [2.5*cm, 8.5*cm, 3.3*cm], ST))
    story.append(P('Table 6: Semi-synthetic anomaly injection for Air Quality dataset.', ST['caption']))
    hr()

    # ── 6. Experiments ────────────────────────────────────────────────────────
    story.append(section_header(6, 'Experiments', ST))
    sp(0.3)

    story.append(subsection_header('6.1', 'Training Hyperparameters', ST))
    hparam_data = [
        ['Hyperparameter', 'Value', 'Hyperparameter', 'Value'],
        ['Optimizer', 'Adam', 'KD Temperature T', '2.0'],
        ['Learning rate', '1 x 10^-3', 'KD weight lambda_KD', '0.5'],
        ['Weight decay', '1 x 10^-4', 'Window weight beta', '1.0'],
        ['LR Scheduler', 'ReduceLROnPlateau\n(factor=0.5, patience=2, min_lr=1 x 10^-6)',
         'Positive weight w+', '10.0'],
        ['Batch size', '32', 'GRU hidden size H', '64'],
        ['Max epochs', '30', 'GRU layers', '2'],
        ['Early stopping', 'patience=5 (val loss)', 'Dropout', '0.3'],
        ['Gradient clipping', 'max norm = 1.0', 'Bidirectional', 'False'],
        ['Best model restore', 'Yes (lowest val loss epoch)', 'Random seed', '42'],
    ]
    story.append(build_table(hparam_data, [3.5*cm, 4.2*cm, 3.5*cm, 3.1*cm], ST))
    story.append(P('Table 7: All training hyperparameters from config.yaml.', ST['caption']))
    sp(0.3)

    story.append(subsection_header('6.2', 'Hardware Configuration', ST))
    story.append(P(
        'Experiments were conducted on a consumer-grade laptop running Windows 11 '
        '(Intel Core i5 processor, 16 GB RAM) with CPU-only execution. CLIP inference '
        'is cached to disk to avoid repeated computation. The GRU student trains in '
        'under 5 minutes on synthetic datasets (N~2,000 windows). GPU support is '
        'available via the <tt>device: auto</tt> configuration flag in config.yaml - '
        'the Trainer auto-detects CUDA availability.',
        ST['body']))
    sp(0.3)

    story.append(subsection_header('6.3', 'Evaluation Protocol', ST))
    eval_items = [
        '<b>Threshold selection</b>: 100-point sweep over [0.01, 0.99] on validation set '
        'maximising F1-score; threshold frozen for test evaluation.',
        '<b>Window metrics</b>: Precision, Recall, F1, PR-AUC (primary), ROC-AUC on window-level predictions.',
        '<b>Step metrics</b>: Pointwise Precision, Recall, F1, PR-AUC on step-level anomaly scores.',
        '<b>Localization metrics</b>: Mean-IoU (intersection-over-union of predicted vs. true '
        'anomaly intervals), Event-F1 (overlap-based event counting), Point-Adjusted F1 (PA-F1, '
        'reported with documented inflation caveat [8]).',
        '<b>Teacher-only baseline</b>: CLIP p_anomaly scores thresholded at 0.5 (no student).',
        '<b>Ablation</b>: student_baseline (no KD) vs. student_kd (full KD) to quantify '
        'distillation gain.',
    ]
    for item in eval_items:
        story.append(P(f'  -  {item}', ST['bullet']))
        sp(0.15)
    sp(0.2)

    story.append(subsection_header('6.4', 'Code Repository', ST))
    story.append(P(
        'The codebase is included in the accompanying project folder; no public repository URL is set for this submission. '
        'The folder includes source modules (src/), dataset '
        'download scripts (scripts/), unit-test suite (tests/), synthetic data generator, '
        'config.yaml with all hyperparameters, and the Streamlit dashboard (app.py).',
        ST['body']))
    hr()

    # ── 7. Results ────────────────────────────────────────────────────────────
    story.append(section_header(7, 'Results', ST))
    sp(0.3)

    story.append(subsection_header('7.1', 'Quantitative Results', ST))
    story.append(P(
        'No measured test results are available in the supplied workspace. The numeric values '
        'previously drafted here were illustrative estimates, not observations from model runs. '
        'Populate this section only with saved held-out results after downloading the datasets and '
        'running both student modes.', ST['body']))
    sp(0.3)

    story.append(subsection_header('7.2', 'Computational Efficiency', ST))
    story.append(P(
        'Teacher/student latency and parameter counts have not been measured for this project. '
        'Measure both models on the same device and include chart rendering in the teacher cost '
        'before making an efficiency claim.', ST['body']))
    sp(0.3)

    story.append(subsection_header('7.3', 'Key Findings', ST))
    findings = [
        '<b>Research question:</b> does the CLIP soft signal improve the same GRU trained without KD? '
        'This must be answered from matched held-out runs.',
        '<b>Teacher caveat:</b> CLIP was not pretrained on sensor heatmaps, so its validation scores '
        'must be checked before trusting its distillation signal.',
        '<b>Localization:</b> report point- and event-level test metrics and show a held-out timeline '
        'after the experiment is complete.',
        '<b>Note on Air Quality results</b>: Anomaly labels are semi-synthetic (injected), so '
        'any later metrics will evaluate recovery of injected anomaly patterns, not real sensor failures.',
    ]
    for f in findings:
        story.append(P(f'  -  {f}', ST['bullet']))
        sp(0.2)
    hr()

    # ── 8. Plan for Novelty Assessment ────────────────────────────────────────
    story.append(section_header(8, 'Plan for Novelty Assessment', ST))
    sp(0.3)
    story.append(P(
        'The implementation provides a baseline for VLM-guided RNN distillation. The following '
        'directions can be considered after the baseline experiments are measured:',
        ST['body']))
    novelty = [
        '<b>Learnable Prompt Tuning (CoOp-style [9]):</b> Replace hand-crafted text prompts '
        'with learnable prompt embeddings (e.g. 16 context vectors) optimised on a small labeled '
        'validation set. This directly improves CLIP\'s anomaly discrimination for sensor heatmaps '
        'without modifying the frozen CLIP backbone. We hypothesise this can close the domain gap '
        'and boost teacher PR-AUC by 10-20 points.',
        '<b>Multi-Prompt Ensemble Teacher:</b> Rather than a single normal/anomaly prompt pair, '
        'use a bank of diverse semantic prompts and aggregate similarities via learned weighting '
        '(soft attention), producing a more robust and calibrated soft prior p<sub>T</sub>.',
        '<b>Temporal Self-Supervised Pre-training:</b> Pre-train the GRU backbone on masked '
        'time-step reconstruction (analogous to MAE) before KD fine-tuning, providing richer '
        'temporal representations that the KD stage can specialise for anomaly detection.',
        '<b>Online / Continual Distillation:</b> Extend the pipeline to streaming settings where '
        'the teacher scores incoming windows on-the-fly and the student updates continuously, '
        'exploring replay buffers and elastic weight consolidation (EWC) to mitigate catastrophic '
        'forgetting on non-stationary sensor distributions.',
        '<b>Cross-Dataset Transfer:</b> Evaluate zero-shot transfer of a student trained on '
        'one dataset (e.g. 3W Oil) directly applied to a related unseen dataset (e.g. MetroPT-3), '
        'measuring whether the distilled semantic priors generalise across sensor domains.',
    ]
    for i, item in enumerate(novelty, 1):
        story.append(P(f'  <b>({i})</b>  {item}', ST['bullet']))
        sp(0.25)
    hr()

    # ── 9. Conclusion ─────────────────────────────────────────────────────────
    story.append(section_header(9, 'Conclusion', ST))
    sp(0.3)
    story.append(P(
        'We implemented a VLM Teacher-RNN Student knowledge '
        'distillation framework for multivariate time-series anomaly detection. A frozen CLIP '
        'ViT-B/32 teacher converts sliding windows into heatmap images and provides soft anomaly '
        'priors via zero-shot contrastive cosine similarity. A compact two-layer GRU student '
        '(with configuration-dependent parameter count) is trained via a three-term multi-task loss: step-level '
        'localization (weighted BCE, w_plus=10), window-level classification (CE, beta=1.0), '
        'and Hinton temperature-scaled knowledge distillation (T=2.0, lambda_KD=0.5). '
        'Preprocessing includes group/time splits, train-only scaling, and validation-only threshold selection.',
        ST['body']))
    sp(0.3)
    story.append(P(
        'The five benchmark data files are not included and no experiment results or runtime '
        'comparisons are reported in this workspace. The dataset choices are TA-approved; next, '
        'download the data, confirm the metric protocol, run matched KD and supervised baselines, '
        'and replace this status with measured test metrics. '
        'The interactive dashboard supports data upload, CLIP scoring, training curves, evaluation, '
        'and anomaly timeline export.',
        ST['body']))
    hr()

    # ── References ────────────────────────────────────────────────────────────
    story.append(section_header('', 'References', ST))
    sp(0.3)

    refs = [
        '[1] Alec Radford, Jong Wook Kim, Chris Hallacy, et al. Learning Transferable Visual '
        'Models From Natural Language Supervision. <i>ICML</i>, 2021. '
        'https://arxiv.org/abs/2103.00020',

        '[2] Yansen Wang, Zhirui Wang, Yiqing Shen, and Tao Qin. Leveraging Vision-Language '
        'Models for Granular Market Change Prediction. <i>arXiv preprint</i>, 2024. '
        'https://arxiv.org/abs/2409.03022',

        '[3] Ming Jin, Shiyu Wang, Lintao Ma, et al. Time-LLM: Time Series Forecasting by '
        'Reprogramming Large Language Models. <i>ICLR</i>, 2024. '
        'https://arxiv.org/abs/2310.01728',

        '[4] Kyunghyun Cho, Bart van Merrienboer, et al. Learning Phrase Representations using '
        'RNN Encoder-Decoder for Statistical Machine Translation. <i>EMNLP</i>, 2014. '
        'https://arxiv.org/abs/1406.1078',

        '[5] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the Knowledge in a '
        'Neural Network. <i>NIPS Deep Learning Workshop</i>, 2015. '
        'https://arxiv.org/abs/1503.02531',

        '[6] Pankaj Malhotra, Lovekesh Vig, Gautam Shroff, and Puneet Agarwal. Long Short Term '
        'Memory Networks for Anomaly Detection in Time Series. <i>ESANN</i>, 2015. '
        'https://www.esann.org/sites/default/files/proceedings/legacy/es2015-56.pdf',

        '[7] Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom '
        'Soderstrom. Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic '
        'Thresholding. <i>ACM SIGKDD</i>, 2018. https://arxiv.org/abs/1802.04431',

        '[8] Siwon Kim, Kukjin Choi, et al. Towards a Rigorous Evaluation of Time-Series '
        'Anomaly Detection. <i>AAAI</i>, 2022. https://arxiv.org/abs/2109.05257',

        '[9] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to Prompt '
        'for Vision-Language Models. <i>IJCV</i>, 2022. https://arxiv.org/abs/2109.01134',

        '[10] Andrea Dal Pozzolo. Credit Card Fraud Detection Dataset. '
        'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. Accessed: October 2026.',

        '[11] Marc Bachlin et al. Daphnet Freezing of Gait Dataset, UCI Repository #245. '
        'https://archive.ics.uci.edu/dataset/245. Accessed: October 2026.',

        '[12] Saverio De Vito. Air Quality Dataset, UCI Repository #360. '
        'https://archive.ics.uci.edu/dataset/360. Accessed: October 2026.',

        '[13] Ricardo Vargas. 3W Oil Well Dataset. '
        'https://github.com/ricardovvargas/3w_dataset. Accessed: October 2026.',

        '[14] Carlos Ferreira. MetroPT-3 Dataset, Zenodo Record 3657010. '
        'https://zenodo.org/record/3657010. Accessed: October 2026.',

        '[15] Hugging Face. openai/clip-vit-base-patch32 model card. '
        'https://huggingface.co/openai/clip-vit-base-patch32. Accessed: October 2026.',
    ]

    for ref in refs:
        story.append(P(ref, ST['ref']))
        sp(0.1)

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(
        story,
        onFirstPage=lambda c, d: on_page(c, d, ST),
        onLaterPages=lambda c, d: on_page(c, d, ST),
    )
    print(f'[OK] Report saved to: {output_path}')


if __name__ == '__main__':
    report_dir = os.path.dirname(__file__)
    for report_name in ('IE643_Course_Project_Report.pdf', 'IE643_Report_VLM_KD_Anomaly_Detection.pdf'):
        build_report(os.path.join(report_dir, report_name))

