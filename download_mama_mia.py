# =============================================================================
#  Скачивание MRI изображений MAMA-MIA (non-ISPY2)
#  Только synapseclient, никакого CLI
#
#  СТРУКТУРА: images/ -> DUKE_001/ -> DUKE_001_0000.nii.gz, ...
#  Скачивает: DUKE (291) + ISPY1 (171) + NACT (64) = 526 случаев
#  Пропускает: ISPY2 (они в training set)
#
#  ЗАПУСК: python download_images.py
#  Можно прерывать и запускать повторно — пропустит уже скачанные файлы
# =============================================================================

import os, sys
import synapseclient

SYNAPSE_TOKEN = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIl0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc3OTU5MjM2NCwiaWF0IjoxNzc5NTkyMzY0LCJqdGkiOiIzODI2NiIsInN1YiI6IjM1NDc1MDAifQ.VToAhR5G_1IyoRYUs4UasCBpHRoG9yTyFiio_7WhLH3j_27-_vfZqhxyOp_ZORe9sEmcdLbYc2bvjy4NrLiokuk222Ohb6-oArk1uycPmeNMhQtb-94IVUUVwB5kBCVLCjaP7QSRKdYQm9N6zjCVbNu-47TOngvicwA3pPEEbH3L9Y6PFA6xdicJO9_t_QD1Ng4FDYdq3WFtGJ1gmgchmnFBU-r9pNXOx7Xenr9zcJvWH4HR2rmXZyW7LnrE402EdnOtytEl7hmHgxY-rRI0PMPaziLW4clohWpsrTZra3OX4DNO-WmPLRICYEPVyvDJ9LtGMJFeFareeHDpE92TcQ"

IMAGES_DIR = r"C:\Users\olegk\Desktop\MRI Project\images"
SYN_IMAGES = "syn64871114"

# ─────────────────────────────────────────────────────────────────────────────

if "ВАШ_ТОКЕН" in SYNAPSE_TOKEN:
    print("ОШИБКА: вставь токен в SYNAPSE_TOKEN")
    sys.exit(1)

os.makedirs(IMAGES_DIR, exist_ok=True)

syn = synapseclient.Synapse()
syn.login(authToken=SYNAPSE_TOKEN, silent=True)
print("Подключение успешно.")

# Получить все папки случаев
print("Сканируем images/ ...")
all_cases = list(syn.getChildren(SYN_IMAGES))
non_ispy2 = [c for c in all_cases
             if not c["name"].strip().upper().startswith("ISPY2")]
print(f"Всего в images/: {len(all_cases)}")
print(f"Будет скачано (non-ISPY2): {len(non_ispy2)}\n")

done_cases  = 0
done_files  = 0
skip_files  = 0
error_files = 0

for i, case in enumerate(non_ispy2, 1):
    case_name = case["name"]
    case_dir  = os.path.join(IMAGES_DIR, case_name)
    os.makedirs(case_dir, exist_ok=True)

    # Получить список файлов внутри папки случая
    try:
        files = list(syn.getChildren(case["id"]))
    except Exception as e:
        print(f"[{i:3d}/{len(non_ispy2)}] ОШИБКА при чтении {case_name}: {e}")
        error_files += 1
        continue

    # Проверить сколько уже скачано
    already = set(os.listdir(case_dir))
    needed  = [f for f in files if f["name"] not in already]

    if not needed:
        print(f"[{i:3d}/{len(non_ispy2)}] уже есть: {case_name} ({len(files)} файлов)")
        skip_files += len(files)
        done_cases += 1
        continue

    print(f"[{i:3d}/{len(non_ispy2)}] {case_name}: {len(already)} уже есть, скачиваем {len(needed)} ...")

    for fobj in needed:
        fname = fobj["name"]
        try:
            syn.get(fobj["id"], downloadLocation=case_dir, ifcollision="keep.local")
            done_files += 1
        except Exception as e:
            print(f"    ОШИБКА {fname}: {e}")
            error_files += 1

    done_cases += 1

print()
print("=" * 60)
print("  ГОТОВО")
print(f"  Папок обработано : {done_cases}")
print(f"  Файлов скачано   : {done_files}")
print(f"  Файлов пропущено : {skip_files} (уже были)")
print(f"  Ошибок           : {error_files}")
print(f"  Папка: {IMAGES_DIR}")
print("=" * 60)