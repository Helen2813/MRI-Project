# =============================================================================
#  Проверка скачанных файлов MAMA-MIA + статистика для статьи
#  Запускать на ноутбуке в той же папке
# =============================================================================

import os, re
import pandas as pd

BASE_DIR   = r"C:\Users\olegk\Desktop\MRI Project"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
SEG_DIR    = os.path.join(BASE_DIR, "segmentations", "expert")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
SEP = "=" * 65

def get_size_gb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except:
                pass
    return total / (1024 ** 3)

def classify(name):
    n = name.strip().upper()
    if n.startswith("DUKE"):   return "DUKE"
    if n.startswith("ISPY1"):  return "ISPY1"
    if n.startswith("ISPY2"):  return "ISPY2"
    if n.startswith("NACT"):   return "NACT"
    return "OTHER"

print(SEP)
print("  СТАТИСТИКА СКАЧАННОГО MAMA-MIA EXTERNAL SET")
print(SEP)

# ── 1. Маски ─────────────────────────────────────────────────────────────────
print("\n[1/3] Папка segmentations/expert/")
if os.path.exists(SEG_DIR):
    seg_files = [f for f in os.listdir(SEG_DIR) if f.endswith(".nii.gz") or f.endswith(".nii")]
    seg_size  = get_size_gb(SEG_DIR)

    seg_by_cohort = {}
    for f in seg_files:
        c = classify(f)
        seg_by_cohort[c] = seg_by_cohort.get(c, 0) + 1

    print(f"  Файлов масок (.nii.gz): {len(seg_files)}")
    print(f"  Размер на диске:        {seg_size:.2f} GB")
    print("  По когортам:")
    for c, n in sorted(seg_by_cohort.items()):
        print(f"    {c:<10}: {n}")
else:
    print("  ПАПКА НЕ НАЙДЕНА")

# ── 2. Изображения ────────────────────────────────────────────────────────────
print("\n[2/3] Папка images/")
if os.path.exists(IMAGES_DIR):
    img_dirs = [d for d in os.listdir(IMAGES_DIR)
                if os.path.isdir(os.path.join(IMAGES_DIR, d))]
    img_size = get_size_gb(IMAGES_DIR)

    img_by_cohort = {}
    for d in img_dirs:
        c = classify(d)
        img_by_cohort[c] = img_by_cohort.get(c, 0) + 1

    print(f"  Папок с MRI:            {len(img_dirs)}")
    print(f"  Размер на диске:        {img_size:.2f} GB")
    print("  По когортам:")
    for c, n in sorted(img_by_cohort.items()):
        print(f"    {c:<10}: {n}")

    # Проверить что у каждой MRI-папки есть маска
    img_names   = set(d.upper() for d in img_dirs)
    seg_stems   = set()
    if os.path.exists(SEG_DIR):
        for f in os.listdir(SEG_DIR):
            stem = re.sub(r"\.nii(\.gz)?$", "", f).upper()
            # убрать суффикс маски если есть
            stem = re.sub(r"_MASK$|_SEG$|_EXPERT$", "", stem)
            seg_stems.add(stem)

    missing_mask = [d for d in img_dirs if d.upper() not in seg_stems]
    missing_img  = [s for s in seg_stems if s not in img_names]
    print(f"\n  MRI без маски:          {len(missing_mask)}")
    print(f"  Маска без MRI:          {len(missing_img)}")
    if missing_mask[:5]:
        print(f"  Примеры MRI без маски: {missing_mask[:5]}")
else:
    print("  ПАПКА НЕ НАЙДЕНА")

# ── 3. Клиническая таблица ────────────────────────────────────────────────────
print("\n[3/3] Клиническая таблица")
xlsx_files = [f for f in os.listdir(TABLES_DIR) if f.endswith(".xlsx")] if os.path.exists(TABLES_DIR) else []
if xlsx_files:
    clinical_path = os.path.join(TABLES_DIR, xlsx_files[0])
    mama = pd.read_excel(clinical_path, dtype=str)
    print(f"  Файл: {xlsx_files[0]}")
    print(f"  Строк: {len(mama)}  |  Колонок: {len(mama.columns)}")
    print(f"  Колонки: {list(mama.columns)}")

    id_low = {c.lower(): c for c in mama.columns}
    id_col = next((id_low[c] for c in ["patient_id","case_id","id"] if c in id_low), mama.columns[0])
    cohort_col = next((c for c in mama.columns if c.lower() in
                       ["cohort","dataset","collection","source"]), None)
    pcr_col = next((c for c in mama.columns if "pcr" in c.lower()), None)

    # фильтруем только non-ISPY2
    non_ispy2 = mama[~mama[id_col].str.upper().str.startswith("ISPY2")]
    print(f"\n  Non-ISPY2 строк в таблице: {len(non_ispy2)}")

    if cohort_col:
        print(f"\n  По когортам ('{cohort_col}'):")
        for c, n in non_ispy2[cohort_col].value_counts().items():
            print(f"    {str(c):<22}: {n}")

    if pcr_col:
        has_pcr = non_ispy2[pcr_col].notna() & (non_ispy2[pcr_col].str.strip() != "")
        print(f"\n  pCR колонка: '{pcr_col}'")
        print(f"  Случаев с pCR меткой: {has_pcr.sum()} / {len(non_ispy2)}")
        print("  Распределение pCR:")
        for v, n in non_ispy2.loc[has_pcr, pcr_col].value_counts().items():
            print(f"    pCR = {str(v):<15}: {n}")
else:
    print("  xlsx файл не найден в tables/")

# ── Итог ──────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  ИТОГ ДЛЯ СТАТЬИ")
print(SEP)
if os.path.exists(IMAGES_DIR) and os.path.exists(SEG_DIR):
    total_img  = len([d for d in os.listdir(IMAGES_DIR)
                      if os.path.isdir(os.path.join(IMAGES_DIR, d))])
    total_seg  = len([f for f in os.listdir(SEG_DIR)
                      if f.endswith(".nii.gz") or f.endswith(".nii")])
    total_size = get_size_gb(IMAGES_DIR) + get_size_gb(SEG_DIR)
    print(f"  MRI папок:              {total_img}")
    print(f"  Масок:                  {total_seg}")
    print(f"  Всего на диске:         {total_size:.2f} GB")
    print(f"  Готово к загрузке на инстанс: {'ДА' if total_img == total_seg else 'ПРОВЕРЬ — MRI и маски не совпадают'}")
print(SEP)