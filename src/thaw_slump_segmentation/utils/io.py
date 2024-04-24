# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import requests
from tqdm.autonotebook import tqdm


def download(url, dst, **args):
    """
    Downloads a file from a given url to a destination path.
    Resumes partial downloads and skips alrady downloaded files.
    Cf. https://gist.github.com/wy193777/0e2a4932e81afc6aa4c8f7a2984f34e2

    @param: `url` to download file
    @param: `dst` place to put the file
    @param: `args` additional keyword arguments for `requests.get`
    """
    file_size = int(requests.head(url, **args).headers["content-length"])
    if os.path.exists(dst):
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True, **args)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size
