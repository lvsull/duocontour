import requests
from tqdm import tqdm

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
    with open(f"FILE_{i}.tar.gz", 'wb') as f:
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