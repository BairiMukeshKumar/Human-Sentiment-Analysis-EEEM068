from pathlib import Path
import zipfile, urllib.request, shutil
ROOT=Path("data/raw"); ROOT.mkdir(parents=True, exist_ok=True); TARGET=ROOT/"MSCTD"; ZIP=ROOT/"MSCTD.zip"; URL="https://github.com/XL2248/MSCTD/archive/refs/heads/main.zip"
if TARGET.exists() and (TARGET/"MSCTD_data").exists(): print("MSCTD already exists at", TARGET); raise SystemExit(0)
print("Downloading official MSCTD GitHub ZIP..."); urllib.request.urlretrieve(URL, ZIP)
with zipfile.ZipFile(ZIP,"r") as z: z.extractall(ROOT)
ex=ROOT/"MSCTD-main"
if TARGET.exists(): shutil.rmtree(TARGET)
ex.rename(TARGET)
print("MSCTD ready at", TARGET)
