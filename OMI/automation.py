import os
import requests
from collections import defaultdict
from tqdm import tqdm
import time
import subprocess

# ===============================
# CONFIG
# ===============================

URL_FILE = "down.txt"

DAILY_DIR = "omi_daily"
ANNUAL_DIR = "omi_annual"

RETRY_WAIT = 120

os.makedirs(DAILY_DIR, exist_ok=True)
os.makedirs(ANNUAL_DIR, exist_ok=True)

# ===============================
# READ URL LIST
# ===============================

urls_by_year = defaultdict(list)

with open(URL_FILE) as f:
    for line in f:

        url = line.strip()

        if not url.endswith(".he5"):
            continue

        year = url.split("/")[-2]

        urls_by_year[year].append(url)

years = sorted(urls_by_year.keys())

# ===============================
# DOWNLOAD FUNCTION
# ===============================

def download_file(url, filepath):

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000000:
        return True

    while True:

        try:

            r = requests.get(url, stream=True, timeout=120)

            with open(filepath, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

            if os.path.getsize(filepath) < 500000:
                raise Exception("File too small")

            return True

        except Exception as e:

            print("\nNetwork error:", e)
            print(f"Retry in {RETRY_WAIT} seconds\n")

            time.sleep(RETRY_WAIT)

# ===============================
# PROCESS YEARS
# ===============================

for year in years:

    output_file = os.path.join(
        ANNUAL_DIR,
        f"Africa_NO2_{year}_annual.nc"
    )

    if os.path.exists(output_file):

        print("Skip year:", year)
        continue

    print("\n======================")
    print("Processing year:", year)
    print("======================")

    urls = sorted(urls_by_year[year])

    # ===============================
    # DOWNLOAD DAILY FILES
    # ===============================

    print("Downloading daily files...")

    for url in tqdm(urls):

        filename = url.split("/")[-1]
        filepath = os.path.join(DAILY_DIR, filename)

        download_file(url, filepath)

    # ===============================
    # CALL AVERAGE SCRIPT
    # ===============================

    print("Calling average_days.py")

    subprocess.run(
        ["python", "check.py", year],
        check=True
    )

    # ===============================
    # DELETE DAILY FILES AFTER AVERAGE
    # ===============================

    print("Deleting daily files...")

    for url in urls:

        filename = url.split("/")[-1]
        filepath = os.path.join(DAILY_DIR, filename)

        if os.path.exists(filepath):
            os.remove(filepath)

print("\nALL DONE")