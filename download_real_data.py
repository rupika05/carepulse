import httpx
from pathlib import Path

data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

files = {
    "train_ml03.csv": "12dzzaz6K0lQSgTjX42RJHKwm3903AP5M",
    "test_pred_ml03.csv": "11Kndw7gIxficmHC22r0u9fUp2RB-cSgi",
}

client = httpx.Client(follow_redirects=True, timeout=60.0)

for filename, file_id in files.items():
    print(f"Downloading {filename} (id: {file_id})...")
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    res = client.get(url)
    if res.status_code == 200 and len(res.content) > 1000:
        target = data_dir / filename
        target.write_bytes(res.content)
        print(f"[OK] Saved {target} ({len(res.content)} bytes)")
    else:
        # Fallback to alternate export url
        url2 = f"https://drive.google.com/uc?export=download&id={file_id}"
        res2 = client.get(url2)
        target = data_dir / filename
        target.write_bytes(res2.content)
        print(f"Alternate download: {target} ({len(res2.content)} bytes)")
