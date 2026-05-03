import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import io
import tempfile
from datetime import date, datetime
import zipfile

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ROOM_STRUCTURE = {
    "TOFIS":  ["standard"],
    "DRAKOS": ["Front", "Back"],
    "TASOS":  ["Front_Left", "Front_Right", "Back_Left", "Back_Right"]
}

TASOS_SENSOR_FOLDER = {
    "Front_Left":  "TASOS_front_left",
    "Front_Right": "TASOS_front_left",
    "Back_Left":   "TASOS_back_right",
    "Back_Right":  "TASOS_back_right",
}

MONTH_FILE_ALIASES = {
    '09': ['sep', 'sept', 'september'],
    '10': ['oct', 'october'],
    '11': ['nov', 'november'],
    '12': ['dec', 'december'],
    '01': ['jan', 'january', 'junu', 'june', 'juan'],
}

MONTH_FEEDBACK_NAME = {
    '09': 'Sep',
    '10': 'Oct',
    '11': 'Nov',
    '12': 'Dec',
    '01': 'Jan',
}

FEEDBACK_WEIGHTS = {
    'comfortable': 1.0,
    'pleasant':    1.0,
    'neutral':     0.5,
    'noticeable':  0.5,
    'irritating':  0.0,
    'too hot':     0.0,
    'too cold':    0.0,
    'too dry':     0.0,
    'too humid':   0.0,
    'unpleasant':  0.0,
}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_seasonal_context(sel_date):
    d = sel_date
    if date(d.year, 9, 1) <= d <= date(d.year, 11, 20):
        return "ΠΕΡΙΟΔΟΣ 1"
    if d >= date(d.year, 11, 21) or d <= date(d.year, 1, 31):
        return "ΠΕΡΙΟΔΟΣ 2"
    return "ΕΚΤΟΣ ΠΕΡΙΟΔΟΥ"


def find_sensor_file(uploaded_files, room, part, month_num):
    """
    Βρίσκει το sensor CSV για συγκεκριμένη αίθουσα/τμήμα/μήνα.
    
    Λογική ονοματολογίας:
      TOFIS   → sep.csv / oct.csv / ...  (χωρίς prefix)
      DRAKOS  → drakos_front_sep.csv  ή  drakos_back_oct.csv
                ή drakos front sep.csv  κλπ. (με κενό/underscore/χωρίς)
      TASOS   → tasos_front_left_sep.csv  κλπ.  (ή sensor folder name)
    """
    aliases = MONTH_FILE_ALIASES.get(month_num, [])
    room_l  = room.lower()
    part_l  = part.lower().replace(' ', '_')

    for uf in uploaded_files:
        name     = uf.name.lower()
        name_noext = os.path.splitext(name)[0]

        if 'feedback' in name:
            continue
        if not name.endswith('.csv'):
            continue

        # Έλεγχος αν το όνομα περιέχει κάποιο alias του μήνα
        month_match = any(alias in name_noext.split('_') or name_noext == alias
                          for alias in aliases)
        if not month_match:
            # fallback: απλός έλεγχος αν το alias εμφανίζεται οπουδήποτε στο όνομα
            month_match = any(alias in name_noext for alias in aliases)
        if not month_match:
            continue

        if room == "TOFIS":
            # Δεχόμαστε: sep.csv, tofis_sep.csv, tofis sep.csv
            if room_l in name_noext or name_noext in aliases:
                return uf
            # Αν το αρχείο είναι ΜΟΝΟ ο μήνας (π.χ. sep.csv) και δεν αναφέρει άλλη αίθουσα
            other_rooms = ['drakos', 'tasos']
            if not any(r in name_noext for r in other_rooms):
                return uf

        elif room == "DRAKOS":
            part_variants = [part.lower(), part_l]  # "front", "front"
            if 'drakos' in name_noext and any(p in name_noext for p in part_variants):
                return uf

        elif room == "TASOS":
            # Υποστηρίζουμε sensor folder names κιόλας
            sensor_folder = TASOS_SENSOR_FOLDER.get(part, '').lower()
            if 'tasos' in name_noext:
                if part_l in name_noext or sensor_folder.replace('tasos_','') in name_noext:
                    return uf

    return None


def find_feedback_file(uploaded_files, room, part, month_num):
    """
    Βρίσκει το feedback CSV.
    Αποδεκτές μορφές (case-insensitive):
      TOFIS_Oct_feedbacks.csv
      DRAKOS_Front_Oct_feedbacks.csv
      TASOS_Front_Left_Oct_feedbacks.csv
    """
    fb_month = MONTH_FEEDBACK_NAME.get(month_num, '')

    if room == "TOFIS":
        candidates = [
            f"tofis_{fb_month.lower()}_feedbacks.csv",
            f"tofis_{fb_month}_feedbacks.csv",
        ]
    elif room == "DRAKOS":
        candidates = [
            f"drakos_{part.lower()}_{fb_month.lower()}_feedbacks.csv",
            f"drakos_{part}_{fb_month}_feedbacks.csv",
        ]
    elif room == "TASOS":
        candidates = [
            f"tasos_{part.lower()}_{fb_month.lower()}_feedbacks.csv",
            f"tasos_{part}_{fb_month}_feedbacks.csv",
            f"tasos_{part_l}_{fb_month.lower()}_feedbacks.csv".replace(' ','_')
            if (part_l := part.lower().replace(' ','_')) else "",
        ]
    else:
        candidates = []

    for uf in uploaded_files:
        if uf.name.lower() in [c.lower() for c in candidates if c]:
            return uf

    # Fallback: fuzzy — αρχείο που περιέχει room+part+month+feedback
    fb_month_l = fb_month.lower()
    room_l     = room.lower()
    part_l     = part.lower().replace(' ', '_').replace('_', '')

    for uf in uploaded_files:
        name = uf.name.lower().replace(' ','_')
        if 'feedback' in name and room_l in name and fb_month_l in name:
            part_clean = part.lower().replace('_','').replace(' ','')
            if part_clean in name.replace('_',''):
                return uf

    return None


def read_csv_uploaded(uploaded_file):
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine='python', encoding=enc)
        except Exception:
            continue
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  DECISION TREES
# ─────────────────────────────────────────────────────────────────────────────
def decision_tree_p1(v, comfort_pct, total_f):
    results = []

    T = v.get('T')
    if T == "N/A" or T is None:
        results.append(('info', "Θερμοκρασία", "Δεν υπάρχουν δεδομένα."))
    elif T < 23:
        results.append(('warn', f"Θερμοκρασία ({T} °C)", "Μείωση κλιματισμού ή κλείσιμο παραθύρων."))
    elif T <= 27:
        results.append(('ok', f"Θερμοκρασία ({T} °C)", "Βέλτιστη για Περίοδο 1 (23–27°C, ASHRAE)."))
    else:
        results.append(('alert', f"Θερμοκρασία ({T} °C)", "Άνοιγμα παραθύρων ή ενεργοποίηση A/C."))

    H = v.get('H')
    if H == "N/A" or H is None:
        results.append(('info', "Υγρασία", "Δεν υπάρχουν δεδομένα."))
    elif H < 30:
        results.append(('alert', f"Υγρασία ({H} %)", "Χρήση υγραντήρα ή λειτουργία υγρασίας A/C."))
    elif H < 40:
        results.append(('warn', f"Υγρασία ({H} %)", "Ελαφρά χαμηλή, συνιστάται αύξηση υγρασίας."))
    elif H <= 60:
        results.append(('ok', f"Υγρασία ({H} %)", "Βέλτιστη για Περίοδο 1 (40–60%, ASHRAE)."))
    elif H <= 65:
        results.append(('warn', f"Υγρασία ({H} %)", "Ελαφρύ άνοιγμα παραθύρων ή χρήση αφυγραντήρα."))
    else:
        results.append(('alert', f"Υγρασία ({H} %)", "Άνοιγμα παραθύρων & χρήση αφυγραντήρα."))

    C = v.get('C')
    if C == "N/A" or C is None:
        results.append(('info', "CO2", "Δεν υπάρχουν δεδομένα."))
    elif C < 600:
        results.append(('ok', f"CO2 ({C} ppm)", "Εξαιρετική ποιότητα αέρα."))
    elif C <= 1000:
        results.append(('ok', f"CO2 ({C} ppm)", "Καλό επίπεδο για Περίοδο 1."))
    elif C <= 1500:
        results.append(('warn', f"CO2 ({C} ppm)", "Άνοιγμα παραθύρων για φυσικό αερισμό."))
    else:
        results.append(('alert', f"CO2 ({C} ppm)", "Άμεσος αερισμός – άνοιγμα παραθύρων & πόρτων."))

    VOC = v.get('VOC')
    if VOC == "N/A" or VOC is None:
        results.append(('info', "VOC", "Δεν υπάρχουν δεδομένα."))
    else:
        voc_ugm3 = round(VOC * 3, 0)
        if voc_ugm3 < 300:
            results.append(('ok', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Αποδεκτή ποιότητα."))
        elif voc_ugm3 <= 500:
            results.append(('warn', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Αύξηση αερισμού μέσω ανοίγματος παραθύρων."))
        elif voc_ugm3 <= 1000:
            results.append(('warn', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Έντονος αερισμός – άνοιγμα παραθύρων & πόρτων."))
        else:
            results.append(('alert', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Άμεσος αερισμός & εντοπισμός χημικών πηγών."))

    return results


def decision_tree_p2(v, comfort_pct, total_f):
    results = []

    T = v.get('T')
    if T == "N/A" or T is None:
        results.append(('info', "Θερμοκρασία", "Δεν υπάρχουν δεδομένα."))
    elif T < 20:
        results.append(('alert', f"Θερμοκρασία ({T} °C)", "Αύξηση θέρμανσης & κλείσιμο παραθύρων."))
    elif T <= 24:
        results.append(('ok', f"Θερμοκρασία ({T} °C)", "Βέλτιστη για Περίοδο 2 (20–24°C, ASHRAE)."))
    else:
        results.append(('warn', f"Θερμοκρασία ({T} °C)", "Μείωση θέρμανσης ή χρήση HVAC/A/C."))

    H = v.get('H')
    if H == "N/A" or H is None:
        results.append(('info', "Υγρασία", "Δεν υπάρχουν δεδομένα."))
    elif H < 30:
        results.append(('alert', f"Υγρασία ({H} %)", "Χρήση υγραντήρα (τάση μείωσης υγρασίας χειμερινής περιόδου)."))
    elif H <= 50:
        results.append(('ok', f"Υγρασία ({H} %)", "Βέλτιστη για Περίοδο 2 (30–50%, ASHRAE)."))
    elif H <= 60:
        results.append(('warn', f"Υγρασία ({H} %)", "Χρήση αφυγραντήρα ή HVAC/A/C."))
    else:
        results.append(('alert', f"Υγρασία ({H} %)", "Χρήση αφυγραντήρα ή HVAC/A/C."))

    C = v.get('C')
    if C == "N/A" or C is None:
        results.append(('info', "CO2", "Δεν υπάρχουν δεδομένα."))
    elif C < 1000:
        results.append(('ok', f"CO2 ({C} ppm)", "Καλό επίπεδο για Περίοδο 2."))
    elif C <= 1200:
        results.append(('ok', f"CO2 ({C} ppm)", "Αποδεκτό επίπεδο χειμερινών συνθηκών."))
    elif C <= 1500:
        results.append(('warn', f"CO2 ({C} ppm)", "Ενεργοποίηση μηχανικού εξαερισμού HVAC/A/C."))
    else:
        results.append(('alert', f"CO2 ({C} ppm)", "Άμεση ενεργοποίηση HVAC/A/C."))

    VOC = v.get('VOC')
    if VOC == "N/A" or VOC is None:
        results.append(('info', "VOC", "Δεν υπάρχουν δεδομένα."))
    else:
        voc_ugm3 = round(VOC * 3, 0)
        if voc_ugm3 < 300:
            results.append(('ok', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Αποδεκτή ποιότητα."))
        elif voc_ugm3 <= 500:
            results.append(('warn', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Αύξηση αερισμού μέσω HVAC/A/C."))
        elif voc_ugm3 <= 1000:
            results.append(('warn', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Ενεργοποίηση HVAC/A/C & αποφυγή χημικών πηγών."))
        else:
            results.append(('alert', f"VOC ({VOC} ppb ≈ {voc_ugm3:.0f} μg/m³)", "Άμεση ενεργοποίηση HVAC/A/C & εντοπισμός χημικών πηγών."))

    return results


def decision_tree_common(v, comfort_pct, total_f, P1):
    results = []

    PM1 = v.get('PM1')
    if PM1 == "N/A" or PM1 is None:
        results.append(('info', "PM1", "Δεν υπάρχουν δεδομένα."))
    elif PM1 < 10:
        results.append(('ok', f"PM1 ({PM1} μg/m³)", "Καθαρός αέρας."))
    elif PM1 <= 20:
        if P1:
            results.append(('warn', f"PM1 ({PM1} μg/m³)", "Αύξηση φιλτραρίσματος αέρα μέσω ανοίγματος παραθύρων."))
        else:
            results.append(('warn', f"PM1 ({PM1} μg/m³)", "Χρήση φίλτρου αέρα / HVAC."))
    else:
        if P1:
            results.append(('alert', f"PM1 ({PM1} μg/m³)", "Άμεσος αερισμός & χρήση καθαριστή αέρα."))
        else:
            results.append(('alert', f"PM1 ({PM1} μg/m³)", "Χρήση καθαριστή αέρα / HVAC – αποφύγετε άνοιγμα παραθύρων."))

    PM25 = v.get('PM25')
    if PM25 == "N/A" or PM25 is None:
        results.append(('info', "PM2.5", "Δεν υπάρχουν δεδομένα."))
    elif PM25 < 12:
        results.append(('ok', f"PM2.5 ({PM25} μg/m³)", "Καλή ποιότητα αέρα (WHO)."))
    elif PM25 <= 25:
        if P1:
            results.append(('warn', f"PM2.5 ({PM25} μg/m³)", "Αύξηση αερισμού μέσω ανοίγματος παραθύρων."))
        else:
            results.append(('warn', f"PM2.5 ({PM25} μg/m³)", "Χρήση καθαριστή αέρα / HVAC."))
    else:
        if P1:
            results.append(('alert', f"PM2.5 ({PM25} μg/m³)", "Άμεσος αερισμός & χρήση καθαριστή αέρα."))
        else:
            results.append(('alert', f"PM2.5 ({PM25} μg/m³)", "Άμεση χρήση καθαριστή αέρα / HVAC – αποφύγετε άνοιγμα παραθύρων."))

    N = v.get('N')
    if N == "N/A" or N is None:
        results.append(('info', "Θόρυβος", "Δεν υπάρχουν δεδομένα."))
    elif N < 35:
        results.append(('ok', f"Θόρυβος ({N} dBA)", "Ήσυχο περιβάλλον."))
    elif N <= 50:
        results.append(('warn', f"Θόρυβος ({N} dBA)", "Κλείσιμο παραθύρων για μείωση εξωτερικού θορύβου."))
    else:
        results.append(('alert', f"Θόρυβος ({N} dBA)", "Εντοπισμός & μείωση πηγών θορύβου, κλείσιμο παραθύρων."))

    P_val = v.get('P')
    if P_val == "N/A" or P_val is None:
        results.append(('info', "Πίεση", "Δεν υπάρχουν δεδομένα."))
    elif 980 <= P_val <= 1050:
        results.append(('ok', f"Πίεση ({P_val} hPa)", "Κανονική ατμοσφαιρική πίεση."))
    else:
        results.append(('alert', f"Πίεση ({P_val} hPa)", "Εκτός ορίων (980–1050 hPa) – παρακολούθηση."))

    if total_f >= 3 and comfort_pct is not None:
        if comfort_pct < 40:
            results.append(('alert', f"Ικανοποίηση φοιτητών ({comfort_pct:.1f}%)", "Χαμηλή άνεση, απαιτείται παρέμβαση."))
        elif comfort_pct < 65:
            results.append(('warn', f"Ικανοποίηση φοιτητών ({comfort_pct:.1f}%)", "Μέτρια άνεση, παρακολούθηση συνιστάται."))
        else:
            results.append(('ok', f"Ικανοποίηση φοιτητών ({comfort_pct:.1f}%)", "Υψηλή θερμική άνεση."))

    return results

# ─────────────────────────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def generate_dual_axis_chart(v, comfort_pct, target_time, season_id):
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    labels = ['Θερμ.\n(°C)', 'Υγρασία\n(%)', 'CO2\n(ppm/10)', 'PM2.5\n(μg/m³)', 'Θόρυβος\n(dBA)']
    raw    = [v.get('T'), v.get('H'), v.get('C'), v.get('PM25'), v.get('N')]
    values = [(x / 10 if i == 2 else x) if (x != 'N/A' and x is not None) else 0
              for i, x in enumerate(raw)]
    colors = ['#E74C3C', '#3498DB', '#27AE60', '#9B59B6', '#F39C12']

    fig, ax1 = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#F8F9FA')
    ax1.set_facecolor('#FFFFFF')
    bars = ax1.bar(labels, values, color=colors, alpha=0.82, width=0.5, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, raw):
        if val != 'N/A' and val is not None:
            display = f'{val:.1f}' if isinstance(val, float) else str(val)
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     display, ha='center', va='bottom', fontsize=7.5, color='#2C3E50')
    ax1.set_ylabel('Τιμή αισθητήρα', fontsize=9, color='#2C3E50')
    ax1.set_ylim(0, max((v for v in values if v), default=10) * 1.45)
    ax1.tick_params(axis='both', labelsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = ax1.twinx()
    sat = comfort_pct if comfort_pct is not None else 0
    color_sat = '#27AE60' if sat >= 65 else ('#F39C12' if sat >= 40 else '#E74C3C')
    ax2.axhline(y=sat, color=color_sat, linestyle='--', linewidth=2.2,
                label=f'Satisfaction Index: {sat:.1f}%')
    ax2.fill_between(range(len(labels)), sat, alpha=0.08, color=color_sat)
    ax2.set_ylabel('Satisfaction Index (%)', fontsize=9, color=color_sat)
    ax2.set_ylim(0, 115)
    ax2.tick_params(axis='y', labelcolor=color_sat, labelsize=8)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax2.spines['top'].set_visible(False)

    period_short = 'Π1' if 'ΠΕΡΙΟΔΟΣ 1' in season_id else 'Π2'
    ax1.set_title(
        f"Sensor Data & Student Satisfaction Index  |  "
        f"{target_time.strftime('%d/%m/%Y %H:%M')}  |  {period_short}",
        fontsize=10, pad=10, color='#2C3E50', fontweight='bold')
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return buf


def generate_correlation_heatmap(v, category_counts, total_f):
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    NEGATIVE = {'too hot', 'too cold', 'too dry', 'too humid', 'irritating', 'unpleasant'}
    NEUTRAL  = {'neutral', 'noticeable'}
    POSITIVE = {'comfortable', 'pleasant'}
    total    = max(total_f, 1)
    pos_pct  = sum(c for l, c in category_counts.items() if l.lower() in POSITIVE) / total
    neu_pct  = sum(c for l, c in category_counts.items() if l.lower() in NEUTRAL)  / total
    neg_pct  = sum(c for l, c in category_counts.items() if l.lower() in NEGATIVE) / total

    sensor_params = ['Θερμοκρασία', 'Υγρασία', 'CO2', 'VOC', 'PM1', 'PM2.5', 'Θόρυβος']
    sensor_vals   = [v.get('T','N/A'), v.get('H','N/A'), v.get('C','N/A'),
                     v.get('VOC','N/A'), v.get('PM1','N/A'), v.get('PM25','N/A'), v.get('N','N/A')]
    fb_labels  = ['Θετικό\n(Comf./Pleas.)', 'Ουδέτερο\n(Neutral/Not.)', 'Αρνητικό\n(Hot/Cold/…)']
    matrix = np.array([[pos_pct, neu_pct, neg_pct] for _ in sensor_params])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#FFFFFF')
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Ποσοστό feedback', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_xticks(range(len(fb_labels)))
    ax.set_xticklabels(fb_labels, fontsize=8.5)
    ax.set_yticks(range(len(sensor_params)))
    y_labels = [f"{p}  [{sv if sv != 'N/A' else '—'}]" for p, sv in zip(sensor_params, sensor_vals)]
    ax.set_yticklabels(y_labels, fontsize=8.5)
    for i in range(len(sensor_params)):
        for j in range(len(fb_labels)):
            val = matrix[i][j]
            txt_color = 'white' if val > 0.65 or val < 0.15 else '#1A1A1A'
            ax.text(j, i, f'{val*100:.0f}%', ha='center', va='center',
                    fontsize=9, color=txt_color, fontweight='bold')
    ax.set_title('Correlation Heatmap – Sensor Parameters vs Student Feedback',
                 fontsize=10, pad=10, color='#2C3E50', fontweight='bold')
    ax.spines[:].set_visible(False)
    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return buf


def generate_feedback_pie(category_counts, total_f, comfort_pct):
    if not category_counts or total_f == 0:
        return None
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    labels = list(category_counts.keys())
    sizes  = list(category_counts.values())
    color_map = {
        'comfortable': '#27AE60', 'pleasant': '#2ECC71',
        'neutral':     '#F39C12', 'noticeable': '#E67E22',
        'too hot':     '#E74C3C', 'too cold':  '#3498DB',
        'too dry':     '#E67E22', 'too humid': '#1ABC9C',
        'irritating':  '#C0392B', 'unpleasant':'#8E44AD',
    }
    colors  = [color_map.get(l.lower(), '#95A5A6') for l in labels]
    explode = [0.04] * len(labels)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fig.patch.set_facecolor('#F8F9FA')
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', startangle=140,
        textprops={'fontsize': 8.5},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color('white')
        at.set_fontweight('bold')
    ax.set_title(
        f'Κατανομή Feedback Φοιτητών\nSatisfaction Index: {comfort_pct:.1f}%  |  N={total_f}',
        fontsize=9.5, pad=10, color='#2C3E50', fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────────────────
#  PDF EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(report_text, v, comfort_pct, category_counts, total_f, target_time, season_id):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.pdfmetrics import stringWidth
    except ImportError:
        return None, "reportlab not installed"

    PW, PH = A4
    ML = 18*mm; MR = PW - 18*mm; TW = MR - ML; LINE_H = 5.8*mm
    C_BLUE   = rl_colors.HexColor('#2980B9')
    C_DARK   = rl_colors.HexColor('#2C3E50')
    C_GRAY   = rl_colors.HexColor('#7F8C8D')
    C_RED    = rl_colors.HexColor('#C0392B')
    C_ORANGE = rl_colors.HexColor('#D4890A')
    C_GREEN  = rl_colors.HexColor('#27AE60')

    # Try register DejaVu fonts
    FONT_DIRS = ['/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/dejavu',
                 'C:\\Windows\\Fonts']
    def find_font(name):
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.exists(p): return p
        return None
    sans_r = find_font('DejaVuSans.ttf')
    sans_b = find_font('DejaVuSans-Bold.ttf')
    if sans_r:
        try: pdfmetrics.registerFont(TTFont('Sans', sans_r))
        except: sans_r = None
    if sans_b:
        try: pdfmetrics.registerFont(TTFont('Sans-Bold', sans_b))
        except: sans_b = None
    FM  = 'Sans'      if sans_r else 'Helvetica'
    FMB = 'Sans-Bold' if sans_b else 'Helvetica-Bold'

    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)

    def draw_footer(page_num):
        c.setStrokeColor(C_GRAY); c.setLineWidth(0.3)
        c.line(ML, 14*mm, MR, 14*mm)
        c.setFont(FM, 7); c.setFillColor(C_GRAY)
        c.drawString(ML, 10*mm, 'Domognostics Professional Report – Generated automatically')
        c.drawRightString(MR, 10*mm, f'Σελίδα {page_num}')

    def page_title(title='DOMOGNOSTICS PROFESSIONAL REPORT',
                   sub='IAQ Analysis  |  Objective Measurements & Subjective Perception'):
        c.setFont(FMB, 14); c.setFillColor(C_DARK)
        c.drawCentredString(PW/2, PH-18*mm, title)
        c.setFont(FM, 9); c.setFillColor(C_GRAY)
        c.drawCentredString(PW/2, PH-24*mm, sub)
        c.setStrokeColor(C_BLUE); c.setLineWidth(1)
        c.line(ML, PH-26*mm, MR, PH-26*mm)

    def color_for_line(l):
        if 'DOMOGNOSTICS' in l or 'CLIMATE CONTEXT' in l: return C_BLUE, True
        if '(!)' in l or 'ΑΜΕΣΕΣ' in l: return C_RED, True
        if 'ΠΑΡΑΚΟΛΟΥΘΗΣΗ' in l: return C_ORANGE, True
        if '(ok)' in l or 'ΒΕΛΤΙΣΤΕΣ' in l: return C_GREEN, True
        if any(k in l for k in ['SENSOR DATA','STUDENT SATISFACTION','DECISION TREE','ΣΥΣΧΕΤΙΣΗ','ΑΠΟΤΕΛΕΣΜΑΤΑ']):
            return C_GRAY, True
        return C_DARK, False

    # Page 1 – text
    page_title()
    y = PH - 31*mm
    page_num = 1

    for raw_line in report_text.split('\n'):
        if y < 18*mm:
            draw_footer(page_num); c.showPage(); page_num += 1
            page_title(); y = PH - 31*mm
        clean = (raw_line.replace('🚨','(!)').replace('⚠️','(!)').replace('✅','(ok)')
                         .replace('ℹ️','(i)').replace('❌','X').replace('·','.'))
        col, bold = color_for_line(clean)
        fn, fs = (FMB if bold else FM), 8.5
        c.setFont(fn, fs); c.setFillColor(col)
        max_w = TW - 2*mm
        if stringWidth(clean, fn, fs) <= max_w:
            c.drawString(ML, y, clean); y -= LINE_H
        else:
            words = clean.split(' '); row = ''
            indent = '    ' if clean.startswith('    ') else ''
            for word in words:
                test = (row + ' ' + word).lstrip() if row else word
                if stringWidth(indent + test, fn, fs) <= max_w:
                    row = test
                else:
                    c.drawString(ML, y, indent + row); y -= LINE_H; row = word
                    if y < 18*mm:
                        draw_footer(page_num); c.showPage(); page_num += 1
                        page_title(); y = PH - 31*mm
                        c.setFont(fn, fs); c.setFillColor(col)
            if row:
                c.drawString(ML, y, indent + row); y -= LINE_H
    draw_footer(page_num)

    # Page 2 – charts
    c.showPage(); page_num += 1
    page_title(title='Visual Analysis',
               sub='Sensor Data & Satisfaction Index  |  Correlation Heatmap')
    y2 = PH - 31*mm

    def embed_chart(buf_io, label, y_pos, h_ratio=0.44):
        c.setFont(FMB, 10); c.setFillColor(C_DARK)
        c.drawString(ML, y_pos, label); y_pos -= 4*mm
        img_h = TW * h_ratio
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(buf_io.read()); tmp.close()
        c.drawImage(tmp.name, ML, y_pos - img_h, width=TW, height=img_h,
                    preserveAspectRatio=True, mask='auto')
        os.unlink(tmp.name)
        return y_pos - img_h - 8*mm

    ch1 = generate_dual_axis_chart(v, comfort_pct, target_time, season_id)
    y2  = embed_chart(ch1, '1. Sensor Data & Satisfaction Index', y2, 0.44)
    ch2 = generate_correlation_heatmap(v, category_counts, total_f)
    y2  = embed_chart(ch2, '2. Correlation Heatmap: Parameters vs Feedback', y2, 0.50)
    draw_footer(page_num)

    c.save()
    buf.seek(0)
    return buf, None

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(uploaded_files, room, part, sel_date, hour, minute):
    season_id = get_seasonal_context(sel_date)
    month_num = sel_date.strftime('%m')

    target_time = pd.Timestamp(year=sel_date.year, month=sel_date.month, day=sel_date.day,
                                hour=hour, minute=minute)
    s_start = target_time - pd.Timedelta(minutes=30)
    s_end   = target_time + pd.Timedelta(minutes=30)
    f_start = target_time - pd.Timedelta(minutes=90)
    f_end   = target_time + pd.Timedelta(minutes=90)

    s_file = find_sensor_file(uploaded_files, room, part, month_num)
    f_file = find_feedback_file(uploaded_files, room, part, month_num)

    result = {
        'season_id':   season_id,
        'target_time': target_time,
        's_file': s_file.name if s_file else None,
        'f_file': f_file.name if f_file else None,
        'error': None,
        'v': {}, 'comfort_pct': None, 'category_counts': {}, 'total_f': 0,
        'decisions_main': [], 'decisions_common': [], 'P1': 'ΠΕΡΙΟΔΟΣ 1' in season_id,
        's_start': s_start, 's_end': s_end, 'f_start': f_start, 'f_end': f_end,
        'fb_col': None, 'room': room, 'part': part,
    }

    if not s_file or not f_file:
        result['error'] = 'missing_files'
        return result

    df_s = read_csv_uploaded(s_file)
    df_f = read_csv_uploaded(f_file)
    if df_s is None or df_f is None:
        result['error'] = 'read_error'
        return result

    df_s['Timestamp'] = pd.to_datetime(df_s['Time'], utc=True, errors='coerce').dt.tz_localize(None)
    df_f['Timestamp'] = pd.to_datetime(df_f['Timestamp'], utc=False, errors='coerce')

    snap     = df_s[(df_s['Timestamp'] >= s_start) & (df_s['Timestamp'] <= s_end)]
    window_f = df_f[(df_f['Timestamp'] >= f_start) & (df_f['Timestamp'] <= f_end)].copy()

    if snap.empty:
        result['error'] = 'no_sensor_data'
        return result

    m_col = next((c for c in snap.columns if 'measurement' in c.lower() and 'type' in c.lower()),
                 next((c for c in snap.columns if 'meas' in c.lower()), None))
    v_col = next((c for c in snap.columns if c.lower() == 'value'),
                 next((c for c in snap.columns if 'valu' in c.lower()), None))

    if m_col is None or v_col is None:
        result['error'] = f'no_columns: {list(snap.columns)}'
        return result

    def get_val(names):
        mask = snap[m_col].astype(str).str.strip().str.lower().isin([n.lower() for n in names])
        vals = pd.to_numeric(snap.loc[mask, v_col], errors='coerce').dropna()
        return round(vals.mean(), 1) if not vals.empty else "N/A"

    v = {
        'T':    get_val(['Temperature']),
        'H':    get_val(['Humidity']),
        'C':    get_val(['Carbon Dioxide', 'CO2']),
        'VOC':  get_val(['VOC', 'Volatile Organic Compounds']),
        'PM1':  get_val(['PM1', 'PM 1', 'PM1.0', 'pm1.0']),
        'PM25': get_val(['PM2.5', 'PM 2.5', 'pm2.5']),
        'N':    get_val(['Noise']),
        'P':    get_val(['Pressure']),
    }
    result['v'] = v

    total_f = len(window_f)
    result['total_f'] = total_f
    comfort_pct = None
    fb_col = None
    category_counts = {}

    if total_f > 0:
        fb_col = 'Temperature_Feedback' if 'Temperature_Feedback' in window_f.columns else None
        if fb_col is None:
            fb_col = next((c for c in window_f.columns
                           if any(k in c.lower() for k in ('temp','feed','comfort'))), None)
        if fb_col:
            normalized      = window_f[fb_col].astype(str).str.strip()
            category_counts = normalized.value_counts().to_dict()
            weighted_sum = sum(FEEDBACK_WEIGHTS.get(l.lower(), 0.0) * cnt
                               for l, cnt in category_counts.items())
            comfort_pct = round((weighted_sum / total_f) * 100, 1)

    result.update({'comfort_pct': comfort_pct, 'category_counts': category_counts,
                   'fb_col': fb_col})

    P1 = result['P1']
    result['decisions_main']   = decision_tree_p1(v, comfort_pct, total_f) if P1 else decision_tree_p2(v, comfort_pct, total_f)
    result['decisions_common'] = decision_tree_common(v, comfort_pct, total_f, P1)

    return result

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD REPORT TEXT  (for PDF)
# ─────────────────────────────────────────────────────────────────────────────
def build_report_text(r):
    lines = []
    a = lines.append
    a(f"  DOMOGNOSTICS ANALYSIS | {r['target_time'].strftime('%d/%m/%Y %H:%M')}")
    a(f"  CLIMATE CONTEXT: {r['season_id']}")
    a("=" * 65)
    a(f"  Sensor CSV   : {r['s_file'] or 'ΔΕΝ ΒΡΕΘΗΚΕ'}")
    a(f"  Feedback CSV : {r['f_file'] or 'ΔΕΝ ΒΡΕΘΗΚΕ'}")
    a("=" * 65)

    v = r['v']
    a(f"\n  1. SENSOR DATA  (±30 λεπτά | {r['s_start'].strftime('%H:%M')} – {r['s_end'].strftime('%H:%M')})")
    a("-" * 65)
    a(f"     Θερμοκρασία   : {v.get('T')} °C")
    a(f"     Υγρασία        : {v.get('H')} %")
    a(f"     CO2            : {v.get('C')} ppm")
    a(f"     VOC            : {v.get('VOC')} ppb")
    a(f"     PM1            : {v.get('PM1')} μg/m³")
    a(f"     PM2.5          : {v.get('PM25')} μg/m³")
    a(f"     Θόρυβος        : {v.get('N')} dBA")
    a(f"     Πίεση          : {v.get('P')} hPa")

    total_f = r['total_f']; comfort_pct = r['comfort_pct']
    a(f"\n  2. STUDENT SATISFACTION  (±90 λεπτά | {r['f_start'].strftime('%H:%M')} – {r['f_end'].strftime('%H:%M')})")
    a("-" * 65)
    if total_f == 0:
        a("     Feedbacks    : 0  →  Δεν υπάρχουν δεδομένα satisfaction.")
    elif not r['fb_col']:
        a(f"     Feedbacks    : {total_f}  →  Δεν βρέθηκε στήλη feedback.")
    else:
        a(f"     Feedbacks    : {total_f}")
        a(f"     Satisfaction : {comfort_pct:.1f}%")
        a("     " + "·" * 40)
        for label, count in sorted(r['category_counts'].items(), key=lambda x: -x[1]):
            pct = (count / total_f) * 100
            a(f"     {label:<22}: {count:>2}  ({pct:.1f}%)")

    decisions = r['decisions_main'] + r['decisions_common']
    has_alert = any(s == 'alert' for s, *_ in decisions)
    has_warn  = any(s == 'warn'  for s, *_ in decisions)
    P1 = r['P1']
    period_title = ("ΠΕΡΙΟΔΟΣ 1 – Θερμή/Μεταβατική  [Σεπ – 20 Νοε]" if P1
                    else "ΠΕΡΙΟΔΟΣ 2 – Ψυχρή  [20 Νοε – Ιαν]")
    a(f"\n  3. ΑΠΟΤΕΛΕΣΜΑΤΑ DECISION TREE  |  {period_title}")
    a("=" * 65)
    out_of_range = [(s, p, m) for s, p, m in decisions if s != 'ok']
    if not out_of_range:
        a("  ✅  Όλες οι παράμετροι εντός ορίων.")
    else:
        for status, param, msg in out_of_range:
            if status == 'warn':
                a(f"  ⚠️   {param}\n        → {msg}")
            elif status == 'alert':
                a(f"  🚨  {param}\n        → {msg}")
            else:
                a(f"  ℹ️   {param}\n        → {msg}")

    a("=" * 65)
    if has_alert:
        a("  🚨  ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΑΠΑΙΤΟΥΝΤΑΙ ΑΜΕΣΕΣ ΕΝΕΡΓΕΙΕΣ")
    elif has_warn:
        a("  ⚠️   ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΧΡΕΙΑΖΕΤΑΙ ΠΑΡΑΚΟΛΟΥΘΗΣΗ")
    else:
        a("  ✅  ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΒΕΛΤΙΣΤΕΣ ΣΥΝΘΗΚΕΣ")

    return '\n'.join(lines)

# ─────────────────────────────────────────────────────────────────────────────
#  STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Domognostics Pro",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 800; color: #2980B9; margin-bottom: 0; }
    .sub-title  { font-size: 1rem; color: #7F8C8D; margin-top: 0; }
    .metric-card {
        background: #F8F9FA; border-radius: 10px; padding: 16px;
        border-left: 4px solid #2980B9; margin-bottom: 10px;
    }
    .status-ok    { color: #27AE60; font-weight: 700; }
    .status-warn  { color: #D4890A; font-weight: 700; }
    .status-alert { color: #C0392B; font-weight: 700; }
    .status-info  { color: #2471A3; font-weight: 600; }
    .decision-row {
        padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.92rem;
    }
    .decision-ok    { background: #EAFAF1; border-left: 4px solid #27AE60; }
    .decision-warn  { background: #FEF9E7; border-left: 4px solid #D4890A; }
    .decision-alert { background: #FDEDEC; border-left: 4px solid #C0392B; }
    .decision-info  { background: #EBF5FB; border-left: 4px solid #2471A3; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏛️ Domognostics Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">IAQ Analysis · Decision Tree · Student Satisfaction</p>', unsafe_allow_html=True)
st.divider()

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Ρυθμίσεις Ανάλυσης")
    st.subheader("1. Αίθουσα")
    room = st.selectbox("Αίθουσα", list(ROOM_STRUCTURE.keys()))
    part = st.selectbox("Τμήμα", ROOM_STRUCTURE[room])

    st.subheader("2. Ημερομηνία & Ώρα")
    sel_date = st.date_input("Ημερομηνία", value=date(2025, 10, 1))
    col_h, col_m = st.columns(2)
    with col_h: hour   = st.number_input("Ώρα",   min_value=8,  max_value=22, value=10)
    with col_m: minute = st.number_input("Λεπτά", min_value=0,  max_value=59, value=0)

    st.subheader("3. Αρχεία CSV — Bulk Upload")
    st.markdown("""
> 💡 **Ανέβασε ΟΛΑ τα CSV μαζί** από τους φακέλους  
> `drakos/`, `tasos/`, `Tofis/` — η εφαρμογή τα αναγνωρίζει αυτόματα.
""")

    uploaded_files = st.file_uploader(
        "Επέλεξε όλα τα CSV αρχεία (από όλους τους φακέλους)",
        type=['csv'],
        accept_multiple_files=True,
        help=(
            "Κράτα Ctrl (ή Cmd) για πολλαπλή επιλογή. "
            "Μπορείς να επιλέξεις αρχεία από διαφορετικούς φακέλους κάνοντας "
            "navigate μέσα στο dialog."
        )
    )

    # ── Απογραφή ανεβασμένων αρχείων ─────────────────────────────────────────
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} αρχεία φορτώθηκαν")

        # Κατηγοριοποίηση
        sensors   = [f for f in uploaded_files if 'feedback' not in f.name.lower()]
        feedbacks = [f for f in uploaded_files if 'feedback' in f.name.lower()]

        with st.expander(f"📂 Sensor CSVs ({len(sensors)})", expanded=False):
            for f in sorted(sensors, key=lambda x: x.name):
                st.markdown(f"• `{f.name}`")

        with st.expander(f"📋 Feedback CSVs ({len(feedbacks)})", expanded=False):
            for f in sorted(feedbacks, key=lambda x: x.name):
                st.markdown(f"• `{f.name}`")

        # Quick-check: βρίσκουμε αυτό που θα χρησιμοποιηθεί για τις επιλεγμένες ρυθμίσεις
        month_num = sel_date.strftime('%m')
        s_preview = find_sensor_file(uploaded_files, room, part, month_num)
        f_preview = find_feedback_file(uploaded_files, room, part, month_num)
        st.markdown("**Αντιστοίχιση για επιλεγμένες ρυθμίσεις:**")
        st.markdown(
            f"{'✅' if s_preview else '❌'} Sensor: `{s_preview.name if s_preview else 'ΔΕΝ ΒΡΕΘΗΚΕ'}`"
        )
        st.markdown(
            f"{'✅' if f_preview else '❌'} Feedback: `{f_preview.name if f_preview else 'ΔΕΝ ΒΡΕΘΗΚΕ'}`"
        )

    run_btn = st.button("▶️  RUN ANALYSIS", type="primary", use_container_width=True,
                        disabled=not uploaded_files)

# ── Main Area ─────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.info("👈 Ανέβασε **όλα τα CSV** από το sidebar και πάτα **RUN ANALYSIS**.")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
**🔬 Sensor CSVs** — Όνομα αρχείου = μήνας

| Αίθουσα | Παράδειγμα ονόματος |
|---------|---------------------|
| TOFIS | `sep.csv` ή `tofis_sep.csv` |
| DRAKOS Front | `drakos_front_oct.csv` |
| DRAKOS Back | `drakos_back_oct.csv` |
| TASOS Front Left | `tasos_front_left_nov.csv` |
| TASOS Back Right | `tasos_back_right_dec.csv` |

Αποδεκτές συντομογραφίες: `sep` · `oct` · `nov` · `dec` · `jan`
""")
    with col_g2:
        st.markdown("""
**📋 Feedback CSVs** — Μορφή: `{ROOM}_{Part}_{Month}_feedbacks.csv`

| | Παράδειγμα |
|-|------------|
| TOFIS | `TOFIS_Sep_feedbacks.csv` |
| DRAKOS Front | `DRAKOS_Front_Oct_feedbacks.csv` |
| TASOS FL | `TASOS_Front_Left_Dec_feedbacks.csv` |

*Τα ονόματα είναι case-insensitive.*
""")

    st.info("""
**🗂️ Πώς να ανεβάσεις γρήγορα (Windows):**  
Άνοιξε το Upload dialog → πήγαινε στον φάκελο `thesis/Tofis/` → Ctrl+A → μετά Ctrl+κλικ στα αρχεία από `drakos/` και `tasos/` → Άνοιγμα.  
Όλα ανεβαίνουν μαζί! ✅
""")
    st.stop()

if run_btn or ('last_result' in st.session_state and st.session_state.get('auto_run')):
    with st.spinner("Ανάλυση..."):
        r = run_analysis(uploaded_files, room, part, sel_date, hour, minute)
        st.session_state['last_result'] = r
        st.session_state['auto_run'] = False

r = st.session_state.get('last_result')
if not r:
    st.stop()

# ── Status Banner ─────────────────────────────────────────────────────────────
season_id  = r['season_id']
decisions  = r['decisions_main'] + r['decisions_common']
has_alert  = any(s == 'alert' for s, *_ in decisions)
has_warn   = any(s == 'warn'  for s, *_ in decisions)

if r.get('error') == 'missing_files':
    st.error(f"❌ Δεν βρέθηκαν αρχεία για: **{room} / {part}** — Μήνας: **{sel_date.strftime('%m')}**\n\n"
             f"- Sensor: {'✅ ' + r['s_file'] if r['s_file'] else '❌ ΔΕΝ ΒΡΕΘΗΚΕ'}\n"
             f"- Feedback: {'✅ ' + r['f_file'] if r['f_file'] else '❌ ΔΕΝ ΒΡΕΘΗΚΕ'}")
    st.stop()
elif r.get('error'):
    st.error(f"❌ Σφάλμα: {r['error']}")
    st.stop()

season_label = "🌤️ Περίοδος 1 — Θερμή/Μεταβατική" if 'ΠΕΡΙΟΔΟΣ 1' in season_id else "❄️ Περίοδος 2 — Ψυχρή"
col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("🗓️ Ημ/νία", r['target_time'].strftime('%d/%m/%Y %H:%M'))
col_s2.metric("🏛️ Αίθουσα", f"{r['room']} / {r['part']}")
col_s3.metric("🌡️ Κλιματική Περίοδος", season_label)

if has_alert:
    st.error("🚨 **ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΑΠΑΙΤΟΥΝΤΑΙ ΑΜΕΣΕΣ ΕΝΕΡΓΕΙΕΣ**")
elif has_warn:
    st.warning("⚠️ **ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΧΡΕΙΑΖΕΤΑΙ ΠΑΡΑΚΟΛΟΥΘΗΣΗ**")
else:
    st.success("✅ **ΣΥΝΟΛΙΚΗ ΚΑΤΑΣΤΑΣΗ: ΒΕΛΤΙΣΤΕΣ ΣΥΝΘΗΚΕΣ**")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Μετρήσεις", "🌿 Decision Tree", "📈 Γραφήματα", "📄 Εξαγωγή PDF"])

# ─ Tab 1: Sensor Data ────────────────────────────────────────────────────────
with tab1:
    v = r['v']
    st.subheader(f"Sensor Data  |  ±30 λεπτά  |  {r['s_start'].strftime('%H:%M')} – {r['s_end'].strftime('%H:%M')}")

    cols = st.columns(4)
    sensor_display = [
        ("🌡️ Θερμοκρασία", v.get('T'), "°C"),
        ("💧 Υγρασία",      v.get('H'), "%"),
        ("💨 CO2",          v.get('C'), "ppm"),
        ("🧪 VOC",          v.get('VOC'), "ppb"),
        ("🌫️ PM1",          v.get('PM1'), "μg/m³"),
        ("🌫️ PM2.5",        v.get('PM25'), "μg/m³"),
        ("🔊 Θόρυβος",      v.get('N'), "dBA"),
        ("🧭 Πίεση",        v.get('P'), "hPa"),
    ]
    for i, (label, val, unit) in enumerate(sensor_display):
        with cols[i % 4]:
            display = f"{val} {unit}" if val != "N/A" else "N/A"
            st.metric(label, display)

    st.divider()
    st.subheader(f"Student Satisfaction  |  ±90 λεπτά  |  {r['f_start'].strftime('%H:%M')} – {r['f_end'].strftime('%H:%M')}")
    total_f = r['total_f']; comfort_pct = r['comfort_pct']

    if total_f == 0:
        st.info("ℹ️ Δεν υπάρχουν feedbacks στο χρονικό παράθυρο.")
    elif not r['fb_col']:
        st.warning(f"⚠️ {total_f} feedbacks αλλά δεν βρέθηκε στήλη feedback.")
    else:
        col_f1, col_f2 = st.columns(2)
        col_f1.metric("📋 Αριθμός Feedbacks", total_f)
        col_f2.metric("😊 Satisfaction Index", f"{comfort_pct:.1f}%")
        st.progress(int(comfort_pct) if comfort_pct else 0)
        if r['category_counts']:
            df_fb = pd.DataFrame(list(r['category_counts'].items()), columns=['Κατηγορία', 'Πλήθος'])
            df_fb['%'] = (df_fb['Πλήθος'] / total_f * 100).round(1)
            df_fb = df_fb.sort_values('Πλήθος', ascending=False).reset_index(drop=True)
            st.dataframe(df_fb, use_container_width=True, hide_index=True)

# ─ Tab 2: Decision Tree ──────────────────────────────────────────────────────
with tab2:
    P1 = r['P1']
    period_title = ("🌤️ ΠΕΡΙΟΔΟΣ 1 – Θερμή/Μεταβατική  [Σεπ – 20 Νοε]" if P1
                    else "❄️ ΠΕΡΙΟΔΟΣ 2 – Ψυχρή  [20 Νοε – Ιαν]")
    st.subheader(period_title)

    ICONS = {'ok': '✅', 'warn': '⚠️', 'alert': '🚨', 'info': 'ℹ️'}
    CSS_CLASS = {'ok': 'decision-ok', 'warn': 'decision-warn',
                 'alert': 'decision-alert', 'info': 'decision-info'}

    for status, param, msg in decisions:
        icon = ICONS.get(status, '')
        css  = CSS_CLASS.get(status, '')
        st.markdown(
            f'<div class="decision-row {css}">'
            f'<strong>{icon} {param}</strong><br>'
            f'<span style="color:#555">→ {msg}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Correlation
    if total_f >= 1 and comfort_pct is not None and r['fb_col']:
        st.divider()
        st.subheader("🔗 Συσχέτιση Μετρήσεων – Αντίληψης Φοιτητών")
        NEGATIVE = {'too hot','too cold','too dry','too humid','irritating','unpleasant'}
        category_counts = r['category_counts']
        neg_labels  = [l for l in category_counts if l.lower() in NEGATIVE]
        neg_count   = sum(category_counts[l] for l in neg_labels)
        sensor_ok   = not any(s in ('alert','warn') for s, *_ in r['decisions_main'])

        if sensor_ok and neg_count > 0:
            neg_pct = (neg_count / total_f) * 100
            neg_str = ", ".join(f"{l} ({category_counts[l]})" for l in neg_labels)
            st.warning(f"⚠️ **ΑΣΥΜΦΩΝΙΑ**: Αποδεκτές μετρήσεις αλλά {neg_count}/{total_f} φοιτητές ({neg_pct:.0f}%) δηλώνουν: {neg_str}.\n\n"
                       "→ Πιθανή αιτία: τοπική ανομοιογένεια (θέση καθίσματος, ηλιακή ακτινοβολία, εγγύτητα σε A/C ή θερμαντικό σώμα).\n"
                       "→ Εισήγηση: Επανεξέταση διάταξης χώρου & θέσεων φοιτητών.")
        elif not sensor_ok and comfort_pct == 100 and total_f >= 2:
            st.info(f"ℹ️ **ΠΑΡΑΤΗΡΗΣΗ**: Παρά τις εκτός ορίων μετρήσεις, {total_f} φοιτητές δηλώνουν 100% άνεση.\n\n"
                    "→ Πιθανή αιτία: προσαρμογή στις συνθήκες ή σύντομη διάρκεια έκθεσης.")
        elif not sensor_ok and neg_count > 0:
            neg_pct = (neg_count / total_f) * 100
            neg_str = ", ".join(f"{l} ({category_counts[l]})" for l in neg_labels)
            st.error(f"🚨 **ΣΥΜΦΩΝΙΑ ΠΡΟΒΛΗΜΑΤΟΣ**: Εκτός ορίων μετρήσεις ΚΑΙ {neg_count}/{total_f} φοιτητές ({neg_pct:.0f}%) δηλώνουν: {neg_str}.\n\n"
                     "→ Απαιτείται άμεση παρέμβαση για βελτίωση ποιότητας αέρα.")
        else:
            st.success(f"✅ **ΣΥΜΦΩΝΙΑ**: Αποδεκτές μετρήσεις & {comfort_pct:.0f}% ικανοποίηση φοιτητών.")

# ─ Tab 3: Charts ─────────────────────────────────────────────────────────────
with tab3:
    st.subheader("📊 Οπτική Ανάλυση")
    v = r['v']; target_time = r['target_time']

    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        st.markdown("**Sensor Data & Satisfaction Index**")
        ch1 = generate_dual_axis_chart(v, r['comfort_pct'], target_time, r['season_id'])
        st.image(ch1)

    with col_c2:
        if r['total_f'] > 0 and r['category_counts']:
            st.markdown("**Κατανομή Feedback**")
            ch3 = generate_feedback_pie(r['category_counts'], r['total_f'], r['comfort_pct'] or 0)
            if ch3:
                st.image(ch3)

    st.markdown("**Correlation Heatmap – Parameters vs Feedback**")
    ch2 = generate_correlation_heatmap(v, r['category_counts'], r['total_f'])
    st.image(ch2)

# ─ Tab 4: PDF Export ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("📄 Εξαγωγή PDF Αναφοράς")
    st.info("Το PDF περιλαμβάνει: πλήρη αναφορά κειμένου, dual-axis chart και correlation heatmap.")

    if st.button("📥 Δημιουργία & Λήψη PDF", type="primary"):
        with st.spinner("Δημιουργία PDF..."):
            report_text = build_report_text(r)
            pdf_buf, err = generate_pdf(
                report_text, r['v'], r['comfort_pct'],
                r['category_counts'], r['total_f'],
                r['target_time'], r['season_id']
            )
        if err:
            st.error(f"❌ Σφάλμα δημιουργίας PDF: {err}")
        else:
            fname = f"Domognostics_{r['room']}_{r['part']}_{r['target_time'].strftime('%Y%m%d_%H%M')}.pdf"
            st.download_button(
                label="⬇️ Κατέβασε το PDF",
                data=pdf_buf,
                file_name=fname,
                mime="application/pdf"
            )
            st.success("✅ Το PDF είναι έτοιμο!")

    st.divider()
    st.subheader("📋 Προεπισκόπηση Κειμένου Αναφοράς")
    if r:
        st.code(build_report_text(r), language=None)
