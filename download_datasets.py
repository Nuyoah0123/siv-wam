#!/usr/bin/env python3
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
REPO_ID = os.environ.get("SIV_WAM_DATASET_REPO", "robbyant/robotwin-clean-and-aug-lerobot")
ROOT = Path(os.environ.get("SIV_WAM_DATA_ROOT", "./data"))
TOP_DIRS = [
    "lerobot_robotwin_eef_aug_500",
    "lerobot_robotwin_eef_clean_50",
]
MAX_WORKERS = 2
PAGE_DELAY_SECONDS = 1.5
LIST_RETRIES = 12
DOWNLOAD_RETRIES = 12

os.environ["HF_ENDPOINT"] = ENDPOINT
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import huggingface_hub
try:
    from huggingface_hub import HfApi, RepoFile, RepoFolder, hf_hub_download
except ImportError:
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile, RepoFolder
import huggingface_hub.utils._pagination as pagination

# Some mirror responses put the next-page URL back on huggingface.co.
# Rewrite it to the mirror and deliberately slow pagination to avoid HTTP 429.
_original_get_next_page = pagination._get_next_page
_mirror = urlsplit(ENDPOINT)
_page_lock = threading.Lock()


def _next_page_via_mirror(response):
    url = _original_get_next_page(response)
    if not url:
        return url

    parts = urlsplit(url)
    if parts.hostname == "huggingface.co" and parts.path.startswith("/api/"):
        url = urlunsplit(parts._replace(scheme=_mirror.scheme, netloc=_mirror.netloc))

    # Pagination itself is sequential, but the lock makes this safe if the library changes.
    with _page_lock:
        time.sleep(PAGE_DELAY_SECONDS)
    return url


pagination._get_next_page = _next_page_via_mirror

api = HfApi(endpoint=ENDPOINT, token=False)
ROOT.mkdir(parents=True, exist_ok=True)


def entry_path(entry) -> str:
    path = getattr(entry, "path", None)
    if path is None:
        path = getattr(entry, "rfilename", None)
    if not path:
        raise RuntimeError(f"无法读取远端条目路径: {entry!r}")
    return path


def list_tree_with_retry(path_in_repo=None, recursive=False):
    for attempt in range(1, LIST_RETRIES + 1):
        try:
            return list(
                api.list_repo_tree(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    revision=REVISION,
                    path_in_repo=path_in_repo,
                    recursive=recursive,
                    expand=False,
                    token=False,
                )
            )
        except Exception as exc:
            if attempt == LIST_RETRIES:
                raise
            wait = min(300.0, 5.0 * (2 ** (attempt - 1))) + random.random() * 3.0
            print(
                f"\n列目录失败 [{path_in_repo or '/'}]：{type(exc).__name__}: {exc}\n"
                f"{wait:.1f} 秒后重试 ({attempt}/{LIST_RETRIES})...",
                flush=True,
            )
            time.sleep(wait)


def local_file_matches(item: RepoFile) -> bool:
    path = ROOT / entry_path(item)
    try:
        return path.is_file() and path.stat().st_size == int(item.size)
    except OSError:
        return False


def download_one(item: RepoFile):
    remote_path = entry_path(item)
    dest = ROOT / remote_path

    # Fast reuse: exact path + exact byte size means no network download.
    if local_file_matches(item):
        return "skip", int(item.size), remote_path

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=remote_path,
                revision=REVISION,
                local_dir=str(ROOT),
                endpoint=ENDPOINT,
                token=False,
                etag_timeout=30,
                force_download=False,
            )
            if not dest.is_file():
                raise RuntimeError(f"下载调用结束但目标文件不存在: {dest}")
            actual = dest.stat().st_size
            expected = int(item.size)
            if actual != expected:
                raise RuntimeError(
                    f"下载后大小不符: {remote_path}: local={actual}, remote={expected}"
                )
            return "download", expected, remote_path
        except Exception as exc:
            if attempt == DOWNLOAD_RETRIES:
                raise
            wait = min(300.0, 5.0 * (2 ** (attempt - 1))) + random.random() * 3.0
            print(
                f"\n下载失败 [{remote_path}]：{type(exc).__name__}: {exc}\n"
                f"{wait:.1f} 秒后重试 ({attempt}/{DOWNLOAD_RETRIES})...",
                flush=True,
            )
            time.sleep(wait)


def process_files(files, label):
    files = sorted(files, key=entry_path)
    total_bytes = sum(int(x.size) for x in files)
    existing = [x for x in files if local_file_matches(x)]
    todo = [x for x in files if not local_file_matches(x)]
    existing_bytes = sum(int(x.size) for x in existing)
    todo_bytes = total_bytes - existing_bytes

    def human(n):
        units = ["B", "KiB", "MiB", "GiB", "TiB"]
        x = float(n)
        for unit in units:
            if x < 1024 or unit == units[-1]:
                return f"{x:.2f} {unit}"
            x /= 1024

    print(
        f"\n[{label}] 远端文件 {len(files):,} 个，共 {human(total_bytes)}\n"
        f"  已复用 {len(existing):,} 个，共 {human(existing_bytes)}\n"
        f"  待下载 {len(todo):,} 个，共 {human(todo_bytes)}",
        flush=True,
    )

    if not todo:
        return

    failures = []
    done = 0
    downloaded_bytes = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_one, item): item for item in todo}
        for future in as_completed(futures):
            item = futures[future]
            try:
                status, size, path = future.result()
                done += 1
                if status == "download":
                    downloaded_bytes += size
                if done == 1 or done % 50 == 0 or done == len(todo):
                    print(
                        f"[{label}] 本轮完成 {done:,}/{len(todo):,}；"
                        f"实际新增 {human(downloaded_bytes)}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append((entry_path(item), exc))
                print(f"\n最终失败: {entry_path(item)}: {exc}", flush=True)

    if failures:
        sample = "\n".join(f"  - {p}: {e}" for p, e in failures[:10])
        raise RuntimeError(
            f"[{label}] 有 {len(failures)} 个文件最终失败。重新运行脚本会继续。\n{sample}"
        )


print(f"huggingface_hub: {huggingface_hub.__version__}", flush=True)
print(f"下载根目录: {ROOT}", flush=True)
print("现有同路径、同字节数文件将直接复用。", flush=True)

# Resolve main exactly once, then pin all file operations to the immutable commit.
info = api.repo_info(repo_id=REPO_ID, repo_type="dataset", revision="main", token=False)
REVISION = info.sha
if not REVISION:
    raise RuntimeError("镜像没有返回仓库 commit SHA")
print(f"固定仓库版本: {REVISION}", flush=True)

state_dir = ROOT / ".robotwin_resume" / REVISION
state_dir.mkdir(parents=True, exist_ok=True)

# Root-level files such as empty_emb.pt. Hidden .sumi folder is deliberately not traversed.
root_entries = list_tree_with_retry(path_in_repo=None, recursive=False)
root_files = [x for x in root_entries if isinstance(x, RepoFile)]
process_files(root_files, "仓库根文件")

for top in TOP_DIRS:
    print(f"\n===== 检查 {top} =====", flush=True)
    top_entries = list_tree_with_retry(path_in_repo=top, recursive=False)
    direct_files = [x for x in top_entries if isinstance(x, RepoFile)]
    if direct_files:
        process_files(direct_files, f"{top}/根文件")

    task_dirs = sorted(
        (entry_path(x) for x in top_entries if isinstance(x, RepoFolder))
    )
    print(f"发现 {len(task_dirs)} 个任务目录。", flush=True)

    done_file = state_dir / f"{top}.done"
    if done_file.exists():
        completed = {line.strip() for line in done_file.read_text().splitlines() if line.strip()}
    else:
        completed = set()

    for index, task in enumerate(task_dirs, 1):
        if task in completed:
            print(f"[{index}/{len(task_dirs)}] 已完成记录，跳过目录枚举: {task}", flush=True)
            continue

        print(f"\n[{index}/{len(task_dirs)}] 枚举: {task}", flush=True)
        entries = list_tree_with_retry(path_in_repo=task, recursive=True)
        files = [x for x in entries if isinstance(x, RepoFile)]
        process_files(files, task)

        # Mark only after every file in this task is present at the expected size.
        with done_file.open("a", encoding="utf-8") as f:
            f.write(task + "\n")
            f.flush()
            os.fsync(f.fileno())
        completed.add(task)

print("\n全部目标目录已检查并补齐。", flush=True)
print(f"状态目录: {state_dir}", flush=True)
