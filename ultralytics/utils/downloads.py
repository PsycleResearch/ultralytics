# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from pathlib import Path
from urllib import parse, request

from ultralytics.utils import LOGGER, TQDM, checks

# Define Ultralytics GitHub assets maintained at https://github.com/ultralytics/assets
GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = (
    [f"yolov8{k}{suffix}.pt" for k in "nsmlx" for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + ["mobile_sam.pt"]
    + ["calibration_image_sample_data_20x128x128x3_float32.npy.zip"]
)
GITHUB_ASSETS_STEMS = [Path(k).stem for k in GITHUB_ASSETS_NAMES]


def is_url(url, check=False):
    """
    Validates if the given string is a URL and optionally checks if the URL exists online.

    Args:
        url (str): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.
            Defaults to False.

    Returns:
        (bool): Returns True for a valid URL. If 'check' is True, also returns True if the URL exists online.
            Returns False otherwise.

    Example:
        ```python
        valid = is_url("https://www.example.com")
        ```
    """
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # check if exists online
        return True
    return False


def delete_dsstore(path, files_to_delete=(".DS_Store", "__MACOSX")):
    """
    Deletes all ".DS_store" files under a specified directory.

    Args:
        path (str, optional): The directory path where the ".DS_store" files should be deleted.
        files_to_delete (tuple): The files to be deleted.

    Example:
        ```python
        from ultralytics.utils.downloads import delete_dsstore

        delete_dsstore('path/to/dir')
        ```

    Note:
        ".DS_store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"Deleting {file} files: {matches}")
        for f in matches:
            f.unlink()


def zip_directory(directory, compress=True, exclude=(".DS_Store", "__MACOSX"), progress=True):
    """
    Zips the contents of a directory, excluding files containing strings in the exclude list. The resulting zip file is
    named after the directory and placed alongside it.

    Args:
        directory (str | Path): The path to the directory to be zipped.
        compress (bool): Whether to compress the files while zipping. Default is True.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        (Path): The path to the resulting zip file.

    Example:
        ```python
        from ultralytics.utils.downloads import zip_directory

        file = zip_directory('path/to/dir')
        ```
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Unzip with progress bar
    files_to_zip = [f for f in directory.rglob("*") if f.is_file() and all(x not in f.name for x in exclude)]
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(files_to_zip, desc=f"Zipping {directory} to {zip_file}...", unit="file", disable=not progress):
            f.write(file, file.relative_to(directory))

    return zip_file  # return path to zip file


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False, progress=True):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.
        progress (bool, optional): Whether to display a progress bar. Defaults to True.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Example:
        ```python
        from ultralytics.utils.downloads import unzip_file

        dir = unzip_file('path/to/file.zip')
        ```
    """
    from zipfile import BadZipFile, ZipFile, is_zipfile

    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        files = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in files}

        # Decide to unzip directly or unzip into a directory
        unzip_as_dir = len(top_level_dirs) == 1  # (len(files) > 1 and not files[0].endswith("/"))
        if unzip_as_dir:
            # Zip has 1 top-level directory
            extract_path = path  # i.e. ../datasets
            path = Path(path) / list(top_level_dirs)[0]  # i.e. extract coco8/ dir to ../datasets/
        else:
            # Zip has multiple files at top level
            path = extract_path = Path(path) / Path(file).stem  # i.e. extract multiple files to ../datasets/coco8/

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.warning(f"WARNING ⚠️ Skipping {file} unzip as destination directory {path} is not empty.")
            return path

        for f in TQDM(files, desc=f"Unzipping {file} to {Path(path).resolve()}...", unit="file", disable=not progress):
            # Ensure the file is within the extract_path to avoid path traversal security vulnerability
            if ".." in Path(f).parts:
                LOGGER.warning(f"Potentially insecure file path: {f}, skipping extraction.")
                continue
            zipObj.extract(f, extract_path)

    return path  # return unzip dir


def check_disk_space(url="https://ultralytics.com/assets/coco8.zip", path=Path.cwd(), sf=1.5, hard=True):
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://ultralytics.com/assets/coco8.zip'.
        path (str | Path, optional): The path or drive to check the available free space on.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 1.5.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    raise RuntimeError("Network usage is disable in custom YoloV8 fork")


def get_google_drive_file_info(link):
    """
    Retrieves the direct download link and filename for a shareable Google Drive file link.

    Args:
        link (str): The shareable link of the Google Drive file.

    Returns:
        (str): Direct download URL for the Google Drive file.
        (str): Original filename of the Google Drive file. If filename extraction fails, returns None.

    Example:
        ```python
        from ultralytics.utils.downloads import get_google_drive_file_info

        link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        url, filename = get_google_drive_file_info(link)
        ```
    """
    raise RuntimeError("Network usage is disable in custom YoloV8 fork")


def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    exist_ok=False,
    progress=True,
):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping. Defaults to False.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.

    Example:
        ```python
        from ultralytics.utils.downloads import safe_download

        link = "https://ultralytics.com/assets/bus.jpg"
        path = safe_download(link)
        ```
    """

    raise RuntimeError("Network usage is disable in custom YoloV8 fork")


def get_github_assets(repo='ultralytics/assets', version='latest', retry=False):
    """Return GitHub repo tag and assets (i.e. ['yolov8n.pt', 'yolov8s.pt', ...])."""
    raise RuntimeError("Network usage is disable in custom YoloV8 fork")


def attempt_download_asset(file, repo="ultralytics/assets", release="v8.2.0", **kwargs):
    """
    Attempt to download a file from GitHub release assets if it is not found locally. The function checks for the file
    locally first, then tries to download it from the specified GitHub repository release.

    Args:
        file (str | Path): The filename or file path to be downloaded.
        repo (str, optional): The GitHub repository in the format 'owner/repo'. Defaults to 'ultralytics/assets'.
        release (str, optional): The specific release version to be downloaded. Defaults to 'v8.2.0'.
        **kwargs (any): Additional keyword arguments for the download process.

    Returns:
        (str): The path to the downloaded file.

    Example:
        ```python
        file_path = attempt_download_asset('yolov8n.pt', repo='ultralytics/assets', release='latest')
        ```
    """
    from ultralytics.utils import SETTINGS  # scoped for circular import

    # YOLOv3/5u updates
    file = str(file)
    file = checks.check_yolov5u_filename(file)
    file = Path(file.strip().replace("'", ""))
    if file.exists():
        return str(file)
    elif (SETTINGS["weights_dir"] / file).exists():
        return str(SETTINGS["weights_dir"] / file)
    else:
        raise RuntimeError("Network usage is disable in custom YoloV8 fork")


def download(url, dir=Path.cwd(), unzip=True, delete=False, curl=False, threads=1, retry=3):
    """Downloads and unzips files concurrently if threads > 1, else sequentially."""
    raise RuntimeError("Network usage is disable in custom YoloV8 fork")
