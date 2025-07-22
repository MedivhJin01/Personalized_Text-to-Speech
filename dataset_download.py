import os
import requests
from tqdm import tqdm
import zipfile

def download_vctk(output_dir="VCTK"):
    os.makedirs(output_dir, exist_ok=True)
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    zip_path = os.path.join(output_dir, "VCTK-Corpus-0.92.zip")

    if os.path.exists(zip_path):
        print("Zip file already downloaded.")
    else:
        print("Downloading VCTK...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))

    extract_path = os.path.join(output_dir, "VCTK-Corpus")
    if not os.path.exists(extract_path):
        print("Extracting zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    else:
        print("VCTK already extracted.")

    return extract_path



if __name__ == "__main__":
    vctk_path = download_vctk(output_dir="./dataset/VCTK")
    print(f"VCTK dataset downloaded and extracted to: {vctk_path}")
