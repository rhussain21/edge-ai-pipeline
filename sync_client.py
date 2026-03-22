

import os
import json
import subprocess
import requests
import logging
from datetime import datetime, timezone
from device_config import config

from db_relational import relationalDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths

DB_SYNC_META = os.getenv("DB_SYNC_META")
VDB_SYNC_META = os.getenv("VDB_SYNC_META")

JETSON_URL = os.getenv('JETSON_URL')
JETSON_IP = os.getenv('JETSON_IP')
JETSON_USER = os.getenv('JETSON_USER')

JETSON_PROJECT_DIR = os.getenv("JETSON_PROJECT_DIR")

DB_KEY = os.getenv("REL_DB_PATH")
VECTOR_KEY = os.getenv("VECTOR_DB_PATH")


def load_sync_metadata(meta_path):
    """Load sync metadata JSON, return dict."""
    if not os.path.exists(meta_path):
        return {"instances": {}}
    with open(meta_path, 'r') as f:
        return json.load(f)


def save_sync_metadata(meta_path, metadata):
    """Save sync metadata JSON."""
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Updated sync metadata: {meta_path}")


def check_jetson_health():
    """Check if Jetson API is reachable."""
    try:
        resp = requests.get(f"{JETSON_URL}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.info(f"Jetson healthy: {data}")
            return True
        else:
            logger.error(f"Jetson health check failed: {resp.status_code}")
            return False
    except requests.ConnectionError:
        logger.error(f"Cannot reach Jetson at {JETSON_URL}. Is it running? Check Tailscale.")
        return False
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return False


def sync_relational_db():
    """
    Pull new/updated content from Jetson via /api/content/sync endpoint.
    Uses the last_synced timestamp from sync_metadata.json to do incremental sync.
    """
    meta = load_sync_metadata(DB_SYNC_META)
    DB_KEY = "Database/industry_signals.db"

    if DB_KEY not in meta.get("instances", {}):
        meta.setdefault("instances", {})[DB_KEY] = {
            "last_updated": None,
            "last_synced": None,
            "sync_status": "initialized"
        }

    instance = meta["instances"][DB_KEY]
    last_synced = instance.get("last_synced")

    # If never synced, use epoch
    if not last_synced:
        last_synced = "2000-01-01T00:00:00Z"

    logger.info(f"Syncing relational DB since: {last_synced}")

    try:
        resp = requests.get(
            f"{JETSON_URL}/api/content/sync",
            params={"since": last_synced, "limit": 1000},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        count = data.get("count", 0)
        has_more = data.get("has_more", False)
        records = data.get("data", [])

        logger.info(f"Received {count} records from Jetson (has_more={has_more})")

        if records:
            local_db = relationalDB(DB_KEY)
            result = local_db.upsert_records(records)
            logger.info(f"  Upsert result: {result['inserted']} inserted, {result['updated']} updated, {result['skipped']} skipped")

        # Update sync metadata
        now = datetime.now(timezone.utc).isoformat()
        instance["last_synced"] = now
        instance["last_updated"] = now
        instance["sync_status"] = "synced"
        instance["last_record_count"] = count

        save_sync_metadata(DB_SYNC_META, meta)

        if has_more:
            logger.warning("More records available. Run sync again to continue.")

        return count

    except requests.HTTPError as e:
        logger.error(f"API error during DB sync: {e}")
        instance["sync_status"] = "error"
        save_sync_metadata(DB_SYNC_META, meta)
        return 0
    except Exception as e:
        logger.error(f"DB sync failed: {e}")
        instance["sync_status"] = "error"
        save_sync_metadata(DB_SYNC_META, meta)
        return 0


def sync_signals_db():
    """
    Pull new/updated signals from Jetson via /api/signals/sync endpoint.
    Uses the last_synced timestamp from sync_metadata.json to do incremental sync.
    """
    meta = load_sync_metadata(DB_SYNC_META)
    SIGNALS_KEY = "Database/signals"

    if SIGNALS_KEY not in meta.get("instances", {}):
        meta.setdefault("instances", {})[SIGNALS_KEY] = {
            "last_updated": None,
            "last_synced": None,
            "sync_status": "initialized"
        }

    instance = meta["instances"][SIGNALS_KEY]
    last_synced = instance.get("last_synced") or "2000-01-01T00:00:00Z"

    logger.info(f"Syncing signals since: {last_synced}")

    try:
        resp = requests.get(
            f"{JETSON_URL}/api/signals/sync",
            params={"since": last_synced, "limit": 1000},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()

        count = data.get("count", 0)
        has_more = data.get("has_more", False)
        records = data.get("data", [])

        logger.info(f"Received {count} signals from Jetson (has_more={has_more})")

        if records:
            local_db = relationalDB(DB_KEY)
            result = local_db.upsert_signals(records)
            logger.info(f"  Upsert result: {result['inserted']} inserted, {result['updated']} updated, {result['skipped']} skipped")

        now = datetime.now(timezone.utc).isoformat()
        instance["last_synced"] = now
        instance["last_updated"] = now
        instance["sync_status"] = "synced"
        instance["last_record_count"] = count

        save_sync_metadata(DB_SYNC_META, meta)

        if has_more:
            logger.warning("More signals available. Run sync again to continue.")

        return count

    except requests.HTTPError as e:
        logger.error(f"API error during signals sync: {e}")
        instance["sync_status"] = "error"
        save_sync_metadata(DB_SYNC_META, meta)
        return 0
    except Exception as e:
        logger.error(f"Signals sync failed: {e}")
        instance["sync_status"] = "error"
        save_sync_metadata(DB_SYNC_META, meta)
        return 0


def _scp_file(remote_path: str, local_path: str, file_key: str, meta: dict) -> bool:
    """SCP a single file from Jetson. Returns True on success."""
    if file_key not in meta.get("instances", {}):
        meta.setdefault("instances", {})[file_key] = {
            "last_updated": None,
            "last_synced": None,
            "sync_status": "initialized"
        }

    logger.info(f"SCP: {remote_path} -> {local_path}")
    try:
        result = subprocess.run(
            ["scp", "-o", "ConnectTimeout=10", remote_path, local_path],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            now = datetime.now(timezone.utc).isoformat()
            file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            meta["instances"][file_key].update({
                "last_synced": now,
                "last_updated": now,
                "sync_status": "synced",
                "file_size_bytes": file_size
            })
            logger.info(f"  OK: {os.path.basename(local_path)} ({file_size / 1024 / 1024:.1f} MB)")
            return True
        else:
            logger.error(f"  SCP failed for {os.path.basename(local_path)}: {result.stderr.strip()}")
            meta["instances"][file_key]["sync_status"] = "error"
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  SCP timed out for {os.path.basename(local_path)}")
        meta["instances"][file_key]["sync_status"] = "timeout"
        return False
    except Exception as e:
        logger.error(f"  SCP error for {os.path.basename(local_path)}: {e}")
        meta["instances"][file_key]["sync_status"] = "error"
        return False


def sync_vector_db():
    """
    Pull FAISS vector files from Jetson via SCP.
    Syncs both corpus vectors and signal vectors to local Vectors/ directory.
    Both live as sibling subdirectories under the vectors base path.
    """
    meta = load_sync_metadata(VDB_SYNC_META)

    remote_base = os.getenv("JETSON_VECTOR_PATH", "/mnt/nvme/vectors/").rstrip("/")
    local_base = VECTOR_KEY.rstrip("/") if VECTOR_KEY else "Vectors"
    os.makedirs(local_base, exist_ok=True)

    # Corpus vector files (in corpus_vectors/ subdirectory)
    corpus_files = [
        "corpus_vectors/corpus_vectors.faiss",
        "corpus_vectors/corpus_vectors.pkl"
    ]

    # Signal vector files (in signal_vectors/ subdirectory)
    signal_files = [
        "signal_vectors/signal_vectors.faiss",
        "signal_vectors/signal_vectors.pkl"
    ]

    success_count = 0

    # Sync corpus vectors
    logger.info("Syncing corpus vectors...")
    local_corpus_dir = os.path.join(local_base, "corpus_vectors")
    os.makedirs(local_corpus_dir, exist_ok=True)
    for filepath in corpus_files:
        remote_path = f"{JETSON_USER}@{JETSON_IP}:{remote_base}/{filepath}"
        local_path = os.path.join(local_base, filepath)
        file_key = f"Vectors/{filepath}"
        if _scp_file(remote_path, local_path, file_key, meta):
            success_count += 1

    # Sync signal vectors
    logger.info("Syncing signal vectors...")
    local_signal_dir = os.path.join(local_base, "signal_vectors")
    os.makedirs(local_signal_dir, exist_ok=True)
    for filepath in signal_files:
        remote_path = f"{JETSON_USER}@{JETSON_IP}:{remote_base}/{filepath}"
        local_path = os.path.join(local_base, filepath)
        file_key = f"Vectors/{filepath}"
        if _scp_file(remote_path, local_path, file_key, meta):
            success_count += 1

    save_sync_metadata(VDB_SYNC_META, meta)
    return success_count


def full_sync():
    """Run full sync: health check -> relational DB -> vector DB."""
    logger.info("=" * 50)
    logger.info("Starting full sync from Jetson")
    logger.info(f"Jetson URL: {JETSON_URL}")
    logger.info(f"Jetson IP:  {JETSON_IP}")
    logger.info("=" * 50)

    # 1. Health check
    if not check_jetson_health():
        logger.error("Jetson not reachable. Aborting sync.")
        return False

    # 2. Sync relational DB via API
    logger.info("\n--- Syncing Relational DB (API) ---")
    db_count = sync_relational_db()
    logger.info(f"Relational DB: {db_count} records synced")

    # 3. Sync signals via API
    logger.info("\n--- Syncing Signals (API) ---")
    signals_count = sync_signals_db()
    logger.info(f"Signals: {signals_count} records synced")

    # 4. Sync vector DB via SCP (content + signal vectors)
    logger.info("\n--- Syncing Vector DB (SCP) ---")
    vdb_count = sync_vector_db()
    logger.info(f"Vector DB: {vdb_count}/4 files synced")

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Sync complete!")
    logger.info(f"  Content records:  {db_count}")
    logger.info(f"  Signal records:   {signals_count}")
    logger.info(f"  VDB files:        {vdb_count}/4 (2 corpus + 2 signal)")
    logger.info("=" * 50)

    return True


if __name__ == "__main__":
    #print(load_sync_metadata(DB_SYNC_META))
    full_sync()
