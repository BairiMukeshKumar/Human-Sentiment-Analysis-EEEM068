from pathlib import Path
import sys, pandas as pd
sys.path.append(str(Path.cwd()))
from src.config import RAW_DATA_DIR, MANIFEST_PATH
from src.dataset_utils import read_lines, convert_label
ROOT=RAW_DATA_DIR/"MSCTD_data"/"ende"
def build(split,en,de,sent):
    e=read_lines(ROOT/en); g=read_lines(ROOT/de); y=read_lines(ROOT/sent)
    if not (len(e)==len(g)==len(y)): raise ValueError(f"Mismatch {split}: {len(e)} {len(g)} {len(y)}")
    rows=[]
    for i in range(len(e)):
        label=convert_label(y[i])
        rows.append({"sample_id":f"{split}_{i}","split":split,"english_text":e[i],"german_text":g[i],"image_index":"","image_path":"","label":label,"sentiment":{0:"negative",1:"neutral",2:"positive"}[label]})
    return rows
rows=[]; rows+=build("train","english_train.txt","german_train.txt","sentiment_train.txt"); rows+=build("val","english_dev.txt","german_dev.txt","sentiment_dev.txt"); rows+=build("test","english_test.txt","german_test.txt","sentiment_test.txt")
df=pd.DataFrame(rows); MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True); df.to_csv(MANIFEST_PATH,index=False); print("Saved",MANIFEST_PATH); print(df.shape); print(df.split.value_counts()); print(df.sentiment.value_counts())
