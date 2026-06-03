import argparse
import os
import sqlite3
import threading as th
from concurrent import futures
from hashlib import md5
from queue import Empty, Full, Queue
from typing import Iterable

import requests
from PIL import Image
from requests.exceptions import RequestException
from tqdm import tqdm

BLOCK_SIZE = 10000


def generate_rows_from_db(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA query_only = YES")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA locking_mode = NORMAL")
    conn.execute("PRAGMA page_size = 4096")
    conn.execute("PRAGMA mmap_size = 268435456")
    conn.execute("PRAGMA cache_size = -524288")
    cursor = conn.execute("SELECT photoid, downloadurl FROM yfcc100m_dataset")

    try:
        for row in cursor:
            yield row
    finally:
        cursor.close()
        conn.close()


class Yielder(th.Thread):
    def __init__(self, gen: Iterable, queue: Queue, end: object, error: object):
        super().__init__()
        self.daemon = True
        self.running = True
        self.gen = gen
        self.queue = queue
        self.end = end
        self.error = error

    def run(self) -> None:
        try:
            for obj in self.gen:
                if not self.running:
                    break
                while True:
                    try:
                        self.queue.put(obj, timeout=1)
                        break
                    except Full:
                        pass
            else:
                self.queue.put(self.end, timeout=1)
        finally:
            self.queue.put(self.error, timeout=1)

    def stop(self) -> None:
        self.running = False


def yield_threaded(gen: Iterable):
    end = object()
    error = object()
    queue: Queue = Queue(maxsize=1024)
    yielder = Yielder(gen, queue, end, error)
    try:
        yielder.start()
        while True:
            try:
                obj = queue.get(timeout=1)
                if obj is end:
                    break
                if obj is error:
                    raise RuntimeError()
                yield obj
            except Empty:
                pass
    finally:
        yielder.stop()


def create_aws_url(download_url: str) -> str:
    byte_map = {f"{v:02x}": f"{v:x}" for v in range(256)}
    h = md5(download_url.encode("utf-8")).hexdigest()
    file_name = "".join(byte_map[h[x : x + 2]] for x in range(0, 32, 2))
    first_three = file_name[:3]
    second_three = file_name[3:6]
    return (
        os.path.join(
            "https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/",
            first_three,
            second_three,
            file_name,
        )
        + ".jpg"
    )


parser = argparse.ArgumentParser()
parser.add_argument("--metadata_path", type=str, required=True, help="Path to the yfcc100m_dataset.sql metadata file")
parser.add_argument("--save_dir", type=str, required=True, help="Directory where images will be stored")
parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of images to attempt per shard")
parser.add_argument("--workers", type=int, default=32, help="Number of concurrent download workers")
parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards to split the dataset into for parallel runs")
parser.add_argument("--shard_id", type=int, default=0, help="Which shard to process in this run (0 <= shard_id < num_shards)")
args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)

headers = {"User-Agent": "i1-yfcc-downloader/1.0"}

SUCCESSFUL_COUNT = 0
ATTEMPTED_COUNT = 0
FAIL_COUNT = 0
SKIPPED_COUNT = 0
success_lock = th.Lock()
thread_local = th.local()


def get_session() -> requests.Session:
    if not hasattr(thread_local, "session"):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=args.workers * 2,
            pool_maxsize=args.workers * 2,
            max_retries=2,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        thread_local.session = session
    return thread_local.session


def download_and_process(idx: int, photoid: str, downloadurl: str, progress_bar: tqdm) -> bool:
    global SUCCESSFUL_COUNT, ATTEMPTED_COUNT, FAIL_COUNT, SKIPPED_COUNT
    if photoid is None or str(photoid).strip() == "":
        with success_lock:
            SKIPPED_COUNT += 1
        progress_bar.update(1)
        return False
    photoid = str(photoid).strip()

    subfolder = os.path.join(args.save_dir, str(idx // BLOCK_SIZE))
    os.makedirs(subfolder, exist_ok=True)

    image_path = os.path.join(subfolder, f"{photoid}.jpg")

    if os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            with success_lock:
                SUCCESSFUL_COUNT += 1
                ATTEMPTED_COUNT += 1
                progress_bar.set_description(f"Successful: {SUCCESSFUL_COUNT}")
            progress_bar.update(1)
            return True
        except Exception:
            # If the existing file is corrupt/incomplete, fall through and re-download it.
            pass

    with success_lock:
        ATTEMPTED_COUNT += 1

    image_url = create_aws_url(downloadurl)
    try:
        session = get_session()
        response = session.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        img_data = response.content
    except RequestException as e:
        progress_bar.write(
            f"[download_fail] idx={idx} url={image_url} error={e}"
        )
        progress_bar.update(1)
        with success_lock:
            FAIL_COUNT += 1
        return False

    try:
        # Write the original bytes to disk first.
        with open(image_path, "wb") as img_file:
            img_file.write(img_data)

        # Validate that the image is decodable.
        with Image.open(image_path) as img:
            img.verify()

        with success_lock:
            SUCCESSFUL_COUNT += 1
            progress_bar.set_description(f"Successful: {SUCCESSFUL_COUNT}")
        progress_bar.update(1)
        return True
    except Exception as e:
        progress_bar.write(
            f"[verify_fail] idx={idx} url={image_url} error={e}"
        )
        try:
            os.remove(image_path)
        except OSError:
            pass
        progress_bar.update(1)
        with success_lock:
            FAIL_COUNT += 1
        return False


def stream_rows(limit: int):
    assert os.path.exists(args.metadata_path), f"Could not find {args.metadata_path}"
    assert args.num_shards >= 1, "--num_shards must be >= 1"
    assert 0 <= args.shard_id < args.num_shards, "--shard_id must be in [0, num_shards)"
    gen = generate_rows_from_db(args.metadata_path)

    produced = 0
    for global_idx, row in enumerate(yield_threaded(gen)):
        if args.num_shards > 1 and global_idx % args.num_shards != args.shard_id:
            continue
        if limit is not None and produced >= limit:
            break

        photoid, downloadurl = row
        yield global_idx, photoid, downloadurl
        produced += 1


if __name__ == "__main__":
    progress_total = args.limit if (args.limit is not None) else 100_000_000 // max(1, args.num_shards)
    progress_bar = tqdm(total=progress_total, desc="Downloading images", smoothing=0.01)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        in_flight = set()
        for idx, photoid, downloadurl in stream_rows(args.limit):
            future = executor.submit(download_and_process, idx, photoid, downloadurl, progress_bar)
            in_flight.add(future)

            if len(in_flight) >= args.workers * 4:
                _, in_flight = futures.wait(
                    in_flight, return_when=futures.FIRST_COMPLETED
                )

        while in_flight:
            _, in_flight = futures.wait(in_flight, return_when=futures.FIRST_COMPLETED)

    progress_bar.close()
    print(
        f"Finished. Attempts: {ATTEMPTED_COUNT}, Successes: {SUCCESSFUL_COUNT}, "
        f"Failures: {FAIL_COUNT}, Skipped: {SKIPPED_COUNT}"
    )