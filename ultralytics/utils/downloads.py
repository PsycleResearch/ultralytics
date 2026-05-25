# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shutil
from pathlib import Path
from urllib import parse, request

from ultralytics.utils import LOGGER, TQDM, checks

# Define Ultralytics GitHub assets maintained at https://github.com/ultralytics/assets
GITHUB_ASSETS_REPO = "ultralytics/assets"
GITHUB_ASSETS_NAMES = frozenset(
    [
        f"yolov8{k}{suffix}.pt"
        for k in "nsmlx"
        for suffix in ("", "-cls", "-seg", "-pose", "-obb", "-oiv7")
    ]
    + [
        f"yolo11{k}{suffix}.pt"
        for k in "nsmlx"
        for suffix in ("", "-cls", "-seg", "-pose", "-obb")
    ]
    + [
        f"yolo12{k}{suffix}.pt" for k in "nsmlx" for suffix in ("",)
    ]  # detect models only currently
    + [
        f"yolo26{k}{suffix}.pt"
        for k in "nsmlx"
        for suffix in ("", "-cls", "-seg", "-sem", "-pose", "-obb")
    ]
    + [f"yolov5{k}{resolution}u.pt" for k in "nsmlx" for resolution in ("", "6")]
    + [f"yolov3{k}u.pt" for k in ("", "-spp", "-tiny")]
    + [f"yolov8{k}-world.pt" for k in "smlx"]
    + [f"yolov8{k}-worldv2.pt" for k in "smlx"]
    + [f"yoloe-v8{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yoloe-11{k}{suffix}.pt" for k in "sml" for suffix in ("-seg", "-seg-pf")]
    + [f"yoloe-26{k}{suffix}.pt" for k in "nsmlx" for suffix in ("-seg", "-seg-pf")]
    + [f"yolov9{k}.pt" for k in "tsmce"]
    + [f"yolov10{k}.pt" for k in "nsmblx"]
    + [f"yolo_nas_{k}.pt" for k in "sml"]
    + [f"sam_{k}.pt" for k in "bl"]
    + [f"sam2_{k}.pt" for k in "blst"]
    + [f"sam2.1_{k}.pt" for k in "blst"]
    + [f"FastSAM-{k}.pt" for k in "sx"]
    + [f"rtdetr-{k}.pt" for k in "lx"]
    + [
        "mobile_sam.pt",
        "mobileclip_blt.ts",
        "yolo11n-grayscale.pt",
        "calibration_image_sample_data_20x128x128x3_float32.npy.zip",
    ]
)
GITHUB_ASSETS_STEMS = frozenset(k.rpartition(".")[0] for k in GITHUB_ASSETS_NAMES)


def is_url(url: str | Path, check: bool = False) -> bool:
    """Validate if the given string is a URL and optionally check if the URL exists online.

    Args:
        url (str | Path): The string to be validated as a URL.
        check (bool, optional): If True, performs an additional check to see if the URL exists online.

    Returns:
        (bool): True for a valid URL. If 'check' is True, also returns True if the URL exists online.

    Examples:
        >>> valid = is_url("https://www.example.com")
        >>> valid_and_exists = is_url("https://www.example.com", check=True)
    """
    try:
        url = str(url)
        result = parse.urlparse(url)
        if not (result.scheme and result.netloc):
            return False
        if check:
            r = request.urlopen(request.Request(url, method="HEAD"), timeout=3)
            return 200 <= r.getcode() < 400
        return True
    except Exception:
        return False


def delete_dsstore(
    path: str | Path, files_to_delete: tuple[str, ...] = (".DS_Store", "__MACOSX")
) -> None:
    """Delete all specified system files in a directory.

    Args:
        path (str | Path): The directory path where the files should be deleted.
        files_to_delete (tuple[str, ...]): The files to be deleted.

    Examples:
        >>> from ultralytics.utils.downloads import delete_dsstore
        >>> delete_dsstore("path/to/dir")

    Notes:
        ".DS_Store" files are created by the Apple operating system and contain metadata about folders and files. They
        are hidden system files and can cause issues when transferring files between different operating systems.
    """
    for file in files_to_delete:
        matches = list(Path(path).rglob(file))
        LOGGER.info(f"Deleting {file} files: {matches}")
        for f in matches:
            f.unlink()


def zip_directory(
    directory: str | Path,
    compress: bool = True,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    progress: bool = True,
) -> Path:
    """Zip the contents of a directory, excluding specified files.

    The resulting zip file is named after the directory and placed alongside it.

    Args:
        directory (str | Path): The path to the directory to be zipped.
        compress (bool): Whether to compress the files while zipping.
        exclude (tuple[str, ...], optional): A tuple of filename strings to be excluded.
        progress (bool, optional): Whether to display a progress bar.

    Returns:
        (Path): The path to the resulting zip file.

    Examples:
        >>> from ultralytics.utils.downloads import zip_directory
        >>> file = zip_directory("path/to/dir")
    """
    from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

    delete_dsstore(directory)
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Zip with progress bar
    files = [
        f
        for f in directory.rglob("*")
        if f.is_file() and all(x not in f.name for x in exclude)
    ]  # files to zip
    zip_file = directory.with_suffix(".zip")
    compression = ZIP_DEFLATED if compress else ZIP_STORED
    with ZipFile(zip_file, "w", compression) as f:
        for file in TQDM(
            files,
            desc=f"Zipping {directory} to {zip_file}...",
            unit="files",
            disable=not progress,
        ):
            f.write(file, file.relative_to(directory))

    return zip_file  # return path to zip file


def unzip_file(
    file: str | Path,
    path: str | Path | None = None,
    exclude: tuple[str, ...] = (".DS_Store", "__MACOSX"),
    exist_ok: bool = False,
    progress: bool = True,
) -> Path:
    """Unzip a *.zip file to the specified path, excluding specified files.

    If the zipfile does not contain a single top-level directory, the function will create a new directory with the same
    name as the zipfile (without the extension) to extract its contents. If a path is not provided, the function will
    use the parent directory of the zipfile as the default path.

    Args:
        file (str | Path): The path to the zipfile to be extracted.
        path (str | Path, optional): The path to extract the zipfile to.
        exclude (tuple[str, ...], optional): A tuple of filename strings to be excluded.
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist.
        progress (bool, optional): Whether to display a progress bar.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Examples:
        >>> from ultralytics.utils.downloads import unzip_file
        >>> directory = unzip_file("path/to/file.zip")
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
        unzip_as_dir = (
            len(top_level_dirs) == 1
        )  # (len(files) > 1 and not files[0].endswith("/"))
        if unzip_as_dir:
            # Zip has 1 top-level directory
            extract_path = path  # i.e. ../datasets
            path = Path(path) / next(
                iter(top_level_dirs)
            )  # i.e. extract coco8/ dir to ../datasets/
        else:
            # Zip has multiple files at top level
            path = extract_path = (
                Path(path) / Path(file).stem
            )  # i.e. extract multiple files to ../datasets/coco8/

        # Check if destination directory already exists and contains files
        if path.exists() and any(path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.warning(
                f"Skipping {file} unzip as destination directory {path} is not empty."
            )
            return path

        extract_path = Path(extract_path).resolve()
        for f in TQDM(
            files,
            desc=f"Unzipping {file} to {Path(path).resolve()}...",
            unit="files",
            disable=not progress,
        ):
            f_path = Path(f)
            target = (extract_path / f_path).resolve()
            if (
                f_path.is_absolute()
                or ".." in f_path.parts
                or target.parts[: len(extract_path.parts)] != extract_path.parts
            ):
                LOGGER.warning(
                    f"Potentially insecure file path: {f}, skipping extraction."
                )
                continue
            zipObj.extract(f, extract_path)

    return path  # return unzip dir


def check_disk_space(
    file_bytes: int,
    path: str | Path = Path.cwd(),
    sf: float = 1.5,
    hard: bool = True,
) -> bool:
    """Check if there is sufficient disk space to download and store a file.

    Args:
        file_bytes (int): The file size in bytes.
        path (str | Path, optional): The path or drive to check the available free space on.
        sf (float, optional): Safety factor, the multiplier for the required free space.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    _total, _used, free = shutil.disk_usage(path)  # bytes
    if file_bytes * sf < free:
        return True  # sufficient space

    def fmt_bytes(b):
        return f"{b / (1 << 20):.1f} MB" if b < (1 << 30) else f"{b / (1 << 30):.3f} GB"

    # Insufficient space
    text = (
        f"Insufficient free disk space {fmt_bytes(free)} < {fmt_bytes(int(file_bytes * sf))} required, "
        f"Please free {fmt_bytes(int(file_bytes * sf - free))} additional disk space and try again."
    )
    if hard:
        raise MemoryError(text)
    LOGGER.warning(text)
    return False


def get_google_drive_file_info(link: str) -> tuple[str, str | None]:
    """Retrieve the direct download link and filename for a shareable Google Drive file link.

    Args:
        link (str): The shareable link of the Google Drive file.

    Returns:
        url (str): Direct download URL for the Google Drive file.
        filename (str | None): Original filename of the Google Drive file. If filename extraction fails, returns None.

    Examples:
        >>> from ultralytics.utils.downloads import get_google_drive_file_info
        >>> link = "https://drive.google.com/file/d/1cqT-cJgANNrhIHCrEufUYhQ4RqiWG_lJ/view?usp=drive_link"
        >>> url, filename = get_google_drive_file_info(link)
    """
    raise RuntimeError("Network usage is disable in our ultralytics fork")


def safe_download(
    url: str | Path,
    file: str | Path | None = None,
    dir: str | Path | None = None,
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    retry: int = 3,
    min_bytes: float = 1e0,
    exist_ok: bool = False,
    progress: bool = True,
) -> Path | str:
    """Download files from a URL with options for retrying, unzipping, and deleting the downloaded file. Enhanced with
    robust partial download detection using Content-Length validation.

    Args:
        url (str | Path): The URL of the file to be downloaded.
        file (str | Path, optional): The filename of the downloaded file. If not provided, the file will be saved with
            the same name as the URL.
        dir (str | Path, optional): The directory to save the downloaded file. If not provided, the file will be saved
            in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file.
        delete (bool, optional): Whether to delete the downloaded file after unzipping.
        curl (bool, optional): Whether to use curl command line tool for downloading.
        retry (int, optional): The number of times to retry the download in case of failure.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping.
        progress (bool, optional): Whether to display a progress bar during the download.

    Returns:
        (Path | str): The path to the downloaded file or extracted directory.

    Examples:
        >>> from ultralytics.utils.downloads import safe_download
        >>> link = "https://ultralytics.com/assets/bus.jpg"
        >>> path = safe_download(link)
    """

    raise RuntimeError("Network usage is disable in our ultralytics fork")


def get_github_assets(
    repo: str = "ultralytics/assets",
    version: str = "latest",
    retry: bool = False,
) -> tuple[str, list[str]]:
    """Retrieve the specified version's tag and assets from a GitHub repository.

    If the version is not specified, the function fetches the latest release assets.

    Args:
        repo (str, optional): The GitHub repository in the format 'owner/repo'.
        version (str, optional): The release version to fetch assets from.
        retry (bool, optional): Flag to retry the request in case of a failure.

    Returns:
        tag (str): The release tag.
        assets (list[str]): A list of asset names.

    Examples:
        >>> tag, assets = get_github_assets(repo="ultralytics/assets", version="latest")
    """
    raise RuntimeError("Network usage is disable in our ultralytics fork")


def attempt_download_asset(
    file: str | Path,
    repo: str = "ultralytics/assets",
    release: str = "v8.4.0",
    **kwargs,
) -> str:
    """Attempt to download a file from GitHub release assets if it is not found locally.

    Args:
        file (str | Path): The filename or file path to be downloaded.
        repo (str, optional): The GitHub repository in the format 'owner/repo'.
        release (str, optional): The specific release version to be downloaded.
        **kwargs (Any): Additional keyword arguments for the download process.

    Returns:
        (str): The path to the downloaded file.

    Examples:
        >>> file_path = attempt_download_asset("yolo26n.pt", repo="ultralytics/assets", release="latest")
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
        raise RuntimeError("Network usage is disable in our ultralytics fork")


def download(
    url: str | list[str] | Path,
    dir: Path = Path.cwd(),
    unzip: bool = True,
    delete: bool = False,
    curl: bool = False,
    threads: int = 1,
    retry: int = 3,
    exist_ok: bool = False,
) -> None:
    """Download files from specified URLs to a given directory.

    Supports concurrent downloads if multiple threads are specified.

    Args:
        url (str | list[str] | Path): The URL or list of URLs of the files to be downloaded.
        dir (Path, optional): The directory where the files will be saved.
        unzip (bool, optional): Flag to unzip the files after downloading.
        delete (bool, optional): Flag to delete the zip files after extraction.
        curl (bool, optional): Flag to use curl for downloading.
        threads (int, optional): Number of threads to use for concurrent downloads.
        retry (int, optional): Number of retries in case of download failure.
        exist_ok (bool, optional): Whether to overwrite existing contents during unzipping.

    Examples:
        >>> download("https://ultralytics.com/assets/example.zip", dir="path/to/dir", unzip=True)
    """
    raise RuntimeError("Network usage is disable in our ultralytics fork")
