import tarfile
import os
import requests
from tqdm import tqdm

OASIS_DIR = "D:/Liam Sullivan LTS/OASIS"

def download_surf():
    links = [
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc1.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc2.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc3.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc4.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc5.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc6.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc7.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc8.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc9.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc10.tar.gz",
        "https://download.nrg.wustl.edu/data/oasis_cs_freesurfer_disc11.tar.gz",
    ]

    for i in range(len(links)):
        with open(f"{OASIS_DIR}/FILE_{i}.tar.gz", 'wb') as f:
            with requests.get(links[i], stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))

                tqdm_params = {
                    'desc': links[i],
                    'total': total,
                    'miniters': 1,
                    'unit': 'B',
                    'unit_scale': True,
                    'unit_divisor': 1024,
                }
                with tqdm(**tqdm_params) as pb:
                    for chunk in r.iter_content(chunk_size=8192):
                        pb.update(len(chunk))
                        f.write(chunk)

def decompress():
    for file in tqdm(os.listdir(OASIS_DIR)):
        if file.endswith(".tar.gz"):
            with tarfile.open(f"{OASIS_DIR}/{file}") as tar:
                tar.extractall(f"{OASIS_DIR}/data")

    for dir in tqdm(os.listdir(f"{OASIS_DIR}/data")):
        fullpath = f"{OASIS_DIR}/data/{dir}"
        if os.path.isdir(fullpath):
            for i in os.listdir(fullpath):
                os.rename(f"{fullpath}/{i}", f"{OASIS_DIR}/data/{i}")

if __name__ == "__main__":
    download_surf()
    decompress()