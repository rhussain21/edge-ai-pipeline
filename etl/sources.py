import os
import re
import json
import mimetypes
import logging
from datetime import datetime
from urllib.parse import urlparse, unquote
import feedparser
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

class ContentSources:
    def __init__(self, content_path, db=None, health_tracker=None):
        self.content_path = content_path
        self.db = db
        self.health_tracker = health_tracker
        self._setup_directories()
    
    def _setup_directories(self):
        """Create standardized directory structure"""
        self.audio_dir = os.path.join(self.content_path, "audio")
        self.text_dir = os.path.join(self.content_path, "text")
        self.pdf_dir = os.path.join(self.content_path, "pdf")
        self.video_dir = os.path.join(self.content_path, "video")
        
        for directory in [self.audio_dir, self.text_dir, self.pdf_dir, self.video_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _sanitize_filename(self, filename):
        clean_name = re.sub(r'[^\w\s-]', '', filename)
        clean_name = re.sub(r'\s+', '_', clean_name)
        return clean_name.strip('_')
    
    def get_podcasts(self, rss_links, num_episodes=None):
        if isinstance(rss_links, str):
            rss_links = [rss_links]
        
        if isinstance(num_episodes, int):
            num_episodes = [num_episodes] * len(rss_links)
        elif num_episodes is None:
            num_episodes = [10] * len(rss_links)
        
        for rss_url, n in zip(rss_links, num_episodes):
            self._download_podcast_feed(rss_url, n)
    
    def _download_podcast_feed(self, rss_url, n_episodes=10):
        try:
            feed = feedparser.parse(rss_url)
            if not feed.entries:
                print(f"No entries found in RSS feed: {rss_url}")
                return
            
            podcast_name = feed.feed.get("title", "Unknown_Podcast")
            podcast_name_clean = self._sanitize_filename(podcast_name)
            
            print(f"\nFetching latest {n_episodes} episodes from {podcast_name}")
            
            downloaded_episodes = []
            
            for ep in feed.entries[:n_episodes]:
                audio_url = ep.enclosures[0].href if ep.enclosures else None
                if not audio_url:
                    print(f"No audio found for episode: {ep.title}")
                    continue
                
                episode_name_clean = self._sanitize_filename(ep.title)
                filename = f"{podcast_name_clean}_{episode_name_clean}.mp3"
                filepath = os.path.join(self.audio_dir, filename)
                
                # Check DB first, then disk
                if self.db and self.db.file_exists(filepath):
                    print(f"Already in database: {filename}")
                    continue
                if os.path.exists(filepath):
                    print(f"Already downloaded: {filename}")
                    continue
                
                metadata = self._create_episode_metadata(ep, podcast_name, filepath, audio_url)
                print(f"Downloading: {ep.title}")
                try:
                    r = requests.get(audio_url, timeout=30)
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded: {filename}")
                    
                    # Write to DB (source of truth)
                    if self.db:
                        file_size_mb = round(len(r.content) / (1024 * 1024), 2)
                        content_data = {
                            'title': metadata.get('episode_title', ep.title),
                            'content_type': 'audio',
                            'source_type': 'podcast',
                            'source_name': podcast_name,
                            'file_path': filepath,
                            'audio_url': audio_url,
                            'transcript': '',
                            'pub_date': metadata.get('pub_date', ''),
                            'duration_seconds': metadata.get('duration'),
                            'file_size_mb': file_size_mb,
                            'content_hash': None,
                            'segments': [],
                            'metadata': metadata
                        }
                        content_id = self.db.add_content_metadata(content_data)
                        if content_id:
                            self.db.add_content_metadata_record(content_id, metadata)
                            print(f"  DB record created: content_id={content_id}")
                    else:
                        # Fallback: write JSON if no DB
                        metadata_file = filepath.replace('.mp3', '_metadata.json')
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    
                    downloaded_episodes.append(metadata)
                    
                except Exception as e:
                    print(f"Failed to download {ep.title}: {e}")
                    continue
            
            print(f"Done fetching episodes for {podcast_name}")
            print(f"Downloaded {len(downloaded_episodes)} new episodes")
            return downloaded_episodes
            
        except Exception as e:
            print(f"Error processing RSS feed {rss_url}: {e}")
            return []
    
    def _create_episode_metadata(self, episode, podcast_name, filepath, audio_url):
        return {
            'podcast_name': podcast_name,
            'episode_title': episode.title,
            'pub_date': episode.get('published', ''),
            'description': episode.get('description', ''),
            'audio_url': audio_url,
            'file_path': filepath,
            'content_type': 'audio',
            'duration': getattr(episode, 'itunes_duration', None),
            'author': getattr(episode, 'itunes_author', None),
            'summary': getattr(episode, 'itunes_summary', None),
            'subtitle': getattr(episode, 'itunes_subtitle', None),
            'keywords': getattr(episode, 'itunes_keywords', None),
            'explicit': getattr(episode, 'itunes_explicit', None),
            'episode_type': getattr(episode, 'itunes_episode_type', None),
            'season': getattr(episode, 'itunes_season', None),
            'episode_number': getattr(episode, 'itunes_episode', None),
            'download_timestamp': datetime.now().isoformat(),
            'file_size_mb': None,
            'processing_status': 'downloaded'
        }

    def download_episodes(self, approved_items):
        """
        Download specific episodes identified by source discovery.

        Args:
            approved_items: List of dicts from SourceDiscoveryService.get_approved_for_ingestion().
                Each dict should have: url (episode page URL), feed_url, title, publisher.

        Returns:
            List of metadata dicts for successfully downloaded episodes.
        """
        # Group approved episodes by feed_url
        from collections import defaultdict
        by_feed = defaultdict(list)
        direct_downloads = []

        for item in approved_items:
            feed_url = item.get('feed_url')
            if feed_url:
                by_feed[feed_url].append(item)
            elif item.get('url'):
                direct_downloads.append(item)

        all_downloaded = []

        # For each feed, parse RSS and match approved episodes
        for feed_url, episodes in by_feed.items():
            try:
                feed = feedparser.parse(feed_url)
                if not feed.entries:
                    print(f"No entries in feed: {feed_url}")
                    continue

                podcast_name = feed.feed.get("title", "Unknown_Podcast")

                # Build lookup: normalize titles for fuzzy matching
                episode_titles = {self._normalize_title(ep['title']) for ep in episodes}
                episode_urls = {ep.get('url', '') for ep in episodes if ep.get('url')}

                matched = 0
                for entry in feed.entries:
                    # Match by episode page URL or normalized title
                    entry_link = entry.get('link', '')
                    entry_title_norm = self._normalize_title(entry.get('title', ''))

                    if entry_link not in episode_urls and entry_title_norm not in episode_titles:
                        continue

                    audio_url = entry.enclosures[0].href if entry.enclosures else None
                    if not audio_url:
                        print(f"No audio found for matched episode: {entry.title}")
                        continue

                    result = self._download_single_episode(entry, podcast_name, audio_url)
                    if result:
                        all_downloaded.append(result)
                    matched += 1

                print(f"Matched {matched}/{len(episodes)} approved episodes from {podcast_name}")

            except Exception as e:
                print(f"Error processing feed {feed_url}: {e}")

        # Direct URL downloads (no feed_url — e.g. web pages with audio)
        for item in direct_downloads:
            print(f"Skipping direct download (no feed_url): {item.get('title', 'Unknown')}")

        print(f"\nTotal downloaded: {len(all_downloaded)} episodes")
        return all_downloaded

    def _normalize_title(self, title):
        """Normalize title for fuzzy matching."""
        return re.sub(r'[^\w\s]', '', title.lower()).strip()

    def _download_single_episode(self, entry, podcast_name, audio_url):
        """Download a single episode entry. Shared by get_podcasts and download_episodes."""
        podcast_name_clean = self._sanitize_filename(podcast_name)
        episode_name_clean = self._sanitize_filename(entry.title)
        filename = f"{podcast_name_clean}_{episode_name_clean}.mp3"
        filepath = os.path.join(self.audio_dir, filename)

        # Check DB first, then disk
        if self.db and self.db.file_exists(filepath):
            print(f"Already in database: {filename}")
            return None
        if os.path.exists(filepath):
            print(f"Already downloaded: {filename}")
            return None

        metadata = self._create_episode_metadata(entry, podcast_name, filepath, audio_url)
        print(f"Downloading: {entry.title}")
        try:
            r = requests.get(audio_url, timeout=30)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(r.content)
            print(f"Downloaded: {filename}")

            if self.health_tracker:
                self.health_tracker.log_download(
                    source_url=audio_url, item_title=entry.title,
                    adapter="rss", success=True
                )

            # Write to DB (source of truth)
            if self.db:
                file_size_mb = round(len(r.content) / (1024 * 1024), 2)
                content_data = {
                    'title': metadata.get('episode_title', entry.title),
                    'content_type': 'audio',
                    'source_type': 'podcast',
                    'source_name': podcast_name,
                    'file_path': filepath,
                    'audio_url': audio_url,
                    'transcript': '',
                    'pub_date': metadata.get('pub_date', ''),
                    'duration_seconds': metadata.get('duration'),
                    'file_size_mb': file_size_mb,
                    'content_hash': None,
                    'segments': [],
                    'metadata': metadata
                }
                content_id = self.db.add_content_metadata(content_data)
                if content_id:
                    self.db.add_content_metadata_record(content_id, metadata)
                    print(f"  DB record created: content_id={content_id}")
            else:
                metadata_file = filepath.replace('.mp3', '_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
            return metadata

        except Exception as e:
            print(f"Failed to download {entry.title}: {e}")
            error_type = str(e.response.status_code) if hasattr(e, 'response') and e.response is not None else type(e).__name__
            if self.health_tracker:
                self.health_tracker.log_download(
                    source_url=audio_url, item_title=entry.title,
                    adapter="rss", success=False,
                    error_type=error_type, error_msg=str(e)
                )
            return None

    # ── Content type mapping ──
    CONTENT_TYPE_MAP = {
        'pdf':       {'dir_attr': 'pdf_dir',   'ext': '.pdf'},
        'html':      {'dir_attr': 'text_dir',  'ext': '.html'},
        'text':      {'dir_attr': 'text_dir',  'ext': '.txt'},
        'audio':     {'dir_attr': 'audio_dir', 'ext': '.mp3'},
        'video':     {'dir_attr': 'video_dir', 'ext': '.mp4'},
        'markdown':  {'dir_attr': 'text_dir',  'ext': '.md'},
        'xml':       {'dir_attr': 'text_dir',  'ext': '.xml'},
        'unknown':   {'dir_attr': 'text_dir',  'ext': '.html'},
    }

    def download_approved(self, approved_items):
        """
        Unified download entry point. Routes items by source_type:
          - podcast → download_episodes() (RSS feed matching)
          - everything else → download_web_content() (direct URL download)

        Args:
            approved_items: List of dicts from SourceDiscoveryService.get_approved_for_ingestion().

        Returns:
            List of metadata dicts for all successfully downloaded items.
        """
        podcasts = [item for item in approved_items if item.get('source_type') == 'podcast']
        web_items = [item for item in approved_items if item.get('source_type') != 'podcast']

        all_downloaded = []

        if podcasts:
            print(f"\n--- Downloading {len(podcasts)} podcast episodes ---")
            all_downloaded.extend(self.download_episodes(podcasts))

        if web_items:
            print(f"\n--- Downloading {len(web_items)} web items ---")
            all_downloaded.extend(self.download_web_content(web_items))

        print(f"\n=== Total downloaded: {len(all_downloaded)} items ===")
        return all_downloaded

    def download_web_content(self, approved_items):
        """
        Download PDFs, HTML pages, and other web content from approved URLs.

        Args:
            approved_items: List of dicts with at least: url, title, adapter, doc_type.

        Returns:
            List of metadata dicts for successfully downloaded items.
        """
        all_downloaded = []

        for item in approved_items:
            url = item.get('url', '')
            if not url:
                logger.warning(f"Skipping item with no URL: {item.get('title', 'Unknown')}")
                continue

            title = item.get('title', 'Untitled')
            adapter = item.get('adapter', 'web')
            publisher = item.get('publisher', '')

            result = self._download_url(
                url=url,
                title=title,
                publisher=publisher,
                adapter=adapter,
                extra_metadata=item,
            )
            if result:
                all_downloaded.append(result)

        print(f"Downloaded {len(all_downloaded)}/{len(approved_items)} web items")
        return all_downloaded

    def _detect_content_type(self, url, response):
        """
        Detect content type from URL extension and HTTP Content-Type header.
        Returns one of: pdf, html, text, audio, video, markdown, xml, unknown.
        """
        # Check URL extension first
        parsed = urlparse(url)
        path = unquote(parsed.path).lower()

        ext_map = {
            '.pdf': 'pdf',
            '.html': 'html', '.htm': 'html',
            '.txt': 'text',
            '.md': 'markdown',
            '.xml': 'xml',
            '.mp3': 'audio', '.wav': 'audio', '.m4a': 'audio',
            '.mp4': 'video', '.webm': 'video',
        }

        for ext, ctype in ext_map.items():
            if path.endswith(ext):
                return ctype

        # Fall back to Content-Type header
        content_type = response.headers.get('Content-Type', '').lower()
        if 'pdf' in content_type:
            return 'pdf'
        elif 'html' in content_type:
            return 'html'
        elif 'text/plain' in content_type:
            return 'text'
        elif 'audio' in content_type:
            return 'audio'
        elif 'video' in content_type:
            return 'video'
        elif 'xml' in content_type:
            return 'xml'

        return 'unknown'

    def _download_url(self, url, title, publisher, adapter, extra_metadata=None):
        """
        Download a single URL to the appropriate content directory.
        Creates content + content_metadata records in DB (no JSON files).
        For HTML content, delegates to _download_html for smart scraping.
        arXiv sources are routed to PDF ingestion instead of HTML scraping.
        """
        # ── PDF routing ──
        # arXiv /abs/ pages only contain metadata + abstract, not the full paper.
        # Route arXiv and any source with a pdf_url to PDF download + text extraction.
        from tools.pdf_extractor import PDFExtractor
        pdf_url = (extra_metadata or {}).get('pdf_url', '')
        is_arxiv = PDFExtractor.is_arxiv_url(url)
        if is_arxiv and not pdf_url:
            pdf_url = PDFExtractor.construct_arxiv_pdf_url(url)
        if is_arxiv or (pdf_url and pdf_url.lower().endswith('.pdf')):
            result = self._download_pdf_source(
                source_url=url, pdf_url=pdf_url,
                title=title, publisher=publisher,
                adapter=adapter, extra_metadata=extra_metadata,
            )
            if result == self._DEDUP_SKIP:
                return None  # Already processed — silent skip
            if result:
                return result
            if is_arxiv:
                # Don't fall through to HTML scraper — arXiv /abs/ only has abstract
                logger.warning(f"PDF extraction failed, skipping: {title}")
                return None
            # Non-arXiv PDF failed — fall through to HTML scraper as fallback

        print(f"Downloading: {title}")
        try:
            r = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; IndustrySignalsBot/1.0)'
            })
            r.raise_for_status()

            content_type = self._detect_content_type(url, r)

            # Route HTML through smart scraper
            if content_type in ('html', 'unknown'):
                html_result = self._download_html(
                    url, title, publisher, adapter, extra_metadata
                )
                if html_result:
                    return html_result
                # Fall through to raw download if scraper fails
            type_info = self.CONTENT_TYPE_MAP.get(content_type, self.CONTENT_TYPE_MAP['unknown'])
            target_dir = getattr(self, type_info['dir_attr'])
            ext = type_info['ext']

            # Build filename: publisher_title.ext
            publisher_clean = self._sanitize_filename(publisher) if publisher else ''
            title_clean = self._sanitize_filename(title)
            if publisher_clean:
                filename = f"{publisher_clean}_{title_clean}{ext}"
            else:
                filename = f"{title_clean}{ext}"

            # Truncate overly long filenames
            if len(filename) > 200:
                filename = filename[:195] + ext

            filepath = os.path.join(target_dir, filename)

            # Check DB first (source of truth), then disk
            if self.db and self.db.file_exists(filepath):
                print(f"Already in database: {filename}")
                return None
            if os.path.exists(filepath):
                print(f"Already downloaded: {filename}")
                return None

            # Write content
            write_mode = 'wb' if content_type in ('pdf', 'audio', 'video') else 'w'
            with open(filepath, write_mode) as f:
                if write_mode == 'wb':
                    f.write(r.content)
                else:
                    f.write(r.text)

            print(f"Downloaded [{content_type}]: {filename}")

            file_size_mb = round(len(r.content) / (1024 * 1024), 2)

            # Build metadata dict
            metadata = self._create_content_metadata(
                url=url,
                title=title,
                publisher=publisher,
                filepath=filepath,
                content_type=content_type,
                file_size_bytes=len(r.content),
                extra=extra_metadata,
            )

            # Write to DB (source of truth)
            if self.db:
                # Map detected content_type to general category
                GENERAL_TYPE_MAP = {
                    'pdf': 'text', 'html': 'html', 'text': 'text',
                    'markdown': 'text', 'xml': 'text',
                    'audio': 'audio', 'video': 'video', 'unknown': 'text',
                }
                general_ctype = GENERAL_TYPE_MAP.get(content_type, 'text')
                content_data = {
                    'title': title,
                    'content_type': general_ctype,
                    'source_type': content_type,  # specific: pdf, html, text, etc.
                    'source_name': publisher or 'Unknown',
                    'file_path': filepath,
                    'audio_url': 'N/A',
                    'transcript': '',
                    'pub_date': (extra_metadata or {}).get('pub_date', ''),
                    'duration_seconds': None,
                    'file_size_mb': file_size_mb,
                    'content_hash': None,
                    'segments': [],
                    'metadata': metadata
                }
                content_id = self.db.add_content_metadata(content_data)
                if content_id:
                    self.db.add_content_metadata_record(content_id, metadata)
                    print(f"  DB record created: content_id={content_id}")
            else:
                # Fallback: write JSON if no DB available
                metadata_file = os.path.splitext(filepath)[0] + '_metadata.json'
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

            if self.health_tracker:
                self.health_tracker.log_download(
                    source_url=url, item_title=title,
                    adapter=adapter, success=True,
                    content_type=content_type,
                )

            return metadata

        except Exception as e:
            print(f"Failed to download {title}: {e}")
            error_type = str(e.response.status_code) if hasattr(e, 'response') and e.response is not None else type(e).__name__
            if self.health_tracker:
                self.health_tracker.log_download(
                    source_url=url, item_title=title,
                    adapter=adapter, success=False,
                    error_type=error_type, error_msg=str(e),
                )
            return None

    def _create_content_metadata(self, url, title, publisher, filepath,
                                  content_type, file_size_bytes=None, extra=None):
        """Create metadata dict for any downloaded content (PDF, HTML, etc.)."""
        metadata = {
            'title': title,
            'url': url,
            'publisher': publisher,
            'content_type': content_type,
            'file_path': filepath,
            'file_size_mb': round(file_size_bytes / (1024 * 1024), 2) if file_size_bytes else None,
            'download_timestamp': datetime.now().isoformat(),
            'processing_status': 'downloaded',
        }
        # Carry over discovery metadata if available
        if extra:
            metadata['doc_type'] = extra.get('doc_type', '')
            metadata['confidence'] = extra.get('confidence', 0.0)
            metadata['reason'] = extra.get('reason', '')
            metadata['topic_tags'] = extra.get('topic_tags', [])
            metadata['authority'] = extra.get('authority', 'unknown')
            metadata['adapter'] = extra.get('adapter', 'unknown')
            metadata['discovered_at'] = extra.get('discovered_at', '')
        return metadata

    def get_pending_content(self):
        """
        Return all downloaded but unprocessed content across all directories.
        Extends get_pending_episodes to cover all content types.
        """
        pending = []
        dirs_and_exts = [
            (self.audio_dir, ['.mp3', '.wav', '.m4a']),
            (self.text_dir, ['.html', '.htm', '.txt', '.md']),
            (self.pdf_dir, ['.pdf']),
            (self.video_dir, ['.mp4', '.webm']),
        ]
        for directory, extensions in dirs_and_exts:
            if not os.path.exists(directory):
                continue
            for filename in os.listdir(directory):
                if not any(filename.endswith(ext) for ext in extensions):
                    continue
                filepath = os.path.join(directory, filename)
                stem = os.path.splitext(filepath)[0]
                metadata_file = stem + '_metadata.json'
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    # Backfill content_type if missing (old metadata files)
                    if 'content_type' not in metadata:
                        metadata['content_type'] = self._ext_to_content_type(filename)
                    if metadata.get('processing_status') == 'downloaded':
                        pending.append(metadata)
                else:
                    metadata = {
                        'title': Path(filename).stem,
                        'file_path': filepath,
                        'content_type': self._ext_to_content_type(filename),
                        'processing_status': 'downloaded',
                        'download_timestamp': datetime.now().isoformat(),
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    pending.append(metadata)
        return pending

    def _ext_to_content_type(self, filename):
        """Map a filename extension to a content type string."""
        ext = os.path.splitext(filename)[1].lower()
        return {
            '.pdf': 'pdf', '.html': 'html', '.htm': 'html',
            '.txt': 'text', '.md': 'markdown', '.xml': 'xml',
            '.mp3': 'audio', '.wav': 'audio', '.m4a': 'audio',
            '.mp4': 'video', '.webm': 'video',
        }.get(ext, 'unknown')

    def get_pending_episodes(self):
        """Get episodes pending transcription from DB."""
        if self.db:
            rows = self.db.query("""
                SELECT id, title, file_path, source_name, content_type
                FROM content
                WHERE extraction_status = 'pending'
                AND content_type = 'audio'
                ORDER BY created_at ASC
            """)
            return [dict(r) for r in rows]
        
        # Fallback: scan disk (legacy)
        pending_episodes = []
        for filename in os.listdir(self.audio_dir):
            if filename.endswith('.mp3'):
                filepath = os.path.join(self.audio_dir, filename)
                metadata_file = filepath.replace('.mp3', '_metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    if metadata.get('processing_status') == 'downloaded':
                        pending_episodes.append(metadata)
        return pending_episodes
    
    def mark_episode_processed(self, filepath, status='processed'):
        """Mark episode as processed in DB."""
        if self.db:
            rows = self.db.query("SELECT id FROM content WHERE file_path = ?", [filepath])
            if rows:
                self.db.update_record(rows[0]['id'], {
                    'extraction_status': 'completed'
                })
            return
        
        # Fallback: update JSON (legacy)
        metadata_file = filepath.replace('.mp3', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            metadata['processing_status'] = status
            metadata['processed_timestamp'] = datetime.now().isoformat()
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def get_download_stats(self):
        stats = {
            'total_audio_files': 0,
            'downloaded': 0,
            'processed': 0,
            'failed': 0,
            'pending': 0
        }
        
        for filename in os.listdir(self.audio_dir):
            if filename.endswith('.mp3'):
                stats['total_audio_files'] += 1
                
                metadata_file = os.path.join(self.audio_dir, filename.replace('.mp3', '_metadata.json'))
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    status = metadata.get('processing_status', 'unknown')
                    if status in stats:
                        stats[status] += 1
        
        stats['pending'] = stats['downloaded']
        return stats

    # Sentinel value: _download_pdf_source returns this for dedup (not a failure)
    _DEDUP_SKIP = "dedup_skip"

    def _download_pdf_source(self, source_url, pdf_url, title, publisher, adapter, extra_metadata=None):
        """
        Download a PDF and extract full text (works for any PDF source).

        Saves:
          - .pdf file in pdf_dir (the actual paper)
          - .txt file in text_dir (extracted clean text, used as transcript)
          - DB record with content_type='text', source_type='pdf',
            transcript=extracted_text

        Returns metadata dict, _DEDUP_SKIP for already-processed items, or None on failure.
        """
        from tools.pdf_extractor import PDFExtractor

        # ── Dedup check BEFORE downloading ──
        publisher_clean = self._sanitize_filename(publisher) if publisher else ''
        title_clean = self._sanitize_filename(title)
        if publisher_clean:
            base_name = f"{publisher_clean}_{title_clean}"
        else:
            base_name = title_clean
        if len(base_name) > 195:
            base_name = base_name[:195]

        txt_filepath = os.path.join(self.text_dir, f"{base_name}.txt")

        if self.db and self.db.file_exists(txt_filepath):
            print(f"Already in database: {base_name}.txt")
            return self._DEDUP_SKIP
        if os.path.exists(txt_filepath):
            print(f"Already downloaded: {base_name}.txt")
            return self._DEDUP_SKIP

        print(f"Downloading PDF: {title}")
        extractor = PDFExtractor(pdf_dir=self.pdf_dir)
        result = extractor.download_and_extract(
            pdf_url=pdf_url, title=title, source_url=source_url,
        )

        if not result:
            if self.health_tracker:
                self.health_tracker.log_download(
                    source_url=source_url, item_title=title,
                    adapter=adapter, success=False,
                    error_type="extraction_failed",
                    error_msg="PDF download or text extraction failed",
                )
            return None

        extracted_text = result["text"]
        pdf_path = result["pdf_path"]

        # Save clean text
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(extracted_text)

        file_size_mb = round(len(extracted_text.encode('utf-8')) / (1024 * 1024), 2)
        print(f"PDF extracted [{result['page_count']} pages, {result['char_count']} chars]: {base_name}")

        provider = (extra_metadata or {}).get('provider', '')
        metadata = self._create_content_metadata(
            url=source_url, title=title, publisher=publisher,
            filepath=txt_filepath, content_type='pdf',
            file_size_bytes=len(extracted_text.encode('utf-8')),
            extra=extra_metadata,
        )
        metadata['source_url'] = source_url
        metadata['pdf_url'] = result['pdf_url']
        metadata['pdf_path'] = pdf_path
        metadata['page_count'] = result['page_count']
        metadata['source_provider'] = provider

        if self.db:
            content_data = {
                'title': title,
                'content_type': 'text',
                'source_type': 'pdf',
                'source_name': publisher or 'unknown',
                'file_path': txt_filepath,
                'audio_url': 'N/A',
                'transcript': extracted_text,
                'pub_date': (extra_metadata or {}).get('pub_date', ''),
                'duration_seconds': None,
                'file_size_mb': file_size_mb,
                'content_hash': None,
                'segments': [],
                'metadata': metadata,
            }
            content_id = self.db.add_content_metadata(content_data)
            if content_id:
                self.db.add_content_metadata_record(content_id, metadata)
                print(f"  DB record created: content_id={content_id}")
        else:
            metadata_file = txt_filepath.replace('.txt', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        if self.health_tracker:
            self.health_tracker.log_download(
                source_url=source_url, item_title=title,
                adapter=adapter, success=True, content_type='pdf',
            )

        return metadata

    def _download_html(self, url, title, publisher, adapter, extra_metadata=None):
        """
        Download an HTML page, scrape clean text via WebScraper, and store both.

        Saves:
          - .html file with raw HTML (for re-processing if needed)
          - .txt file with extracted clean text (used as transcript by pipeline)
          - DB record with content_type='html', transcript=clean_text

        Returns metadata dict or None on failure.
        """
        from tools.web_scraper import WebScraper

        scraper = WebScraper(timeout=30)
        result = scraper.scrape(url)

        if not result['success']:
            logger.warning(f"Scraper failed for {url}: {result['error']}")
            # Fall back to raw download via _download_url
            return None

        clean_text = result['text']
        scraped_meta = result['metadata']
        page_title = result['title'] or title

        # Build filename
        publisher_clean = self._sanitize_filename(publisher) if publisher else ''
        title_clean = self._sanitize_filename(page_title)
        if publisher_clean:
            base_name = f"{publisher_clean}_{title_clean}"
        else:
            base_name = title_clean
        if len(base_name) > 195:
            base_name = base_name[:195]

        # Save clean text as .txt (this is what pipeline reads as transcript)
        txt_filepath = os.path.join(self.text_dir, f"{base_name}.txt")
        # Save raw HTML for reference
        html_filepath = os.path.join(self.text_dir, f"{base_name}.html")

        # Check DB / disk dedup
        if self.db and self.db.file_exists(txt_filepath):
            print(f"Already in database: {base_name}.txt")
            return None
        if os.path.exists(txt_filepath):
            print(f"Already downloaded: {base_name}.txt")
            return None

        # Fetch raw HTML for archival
        try:
            r = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; IndustrySignalsBot/1.0)'
            })
            r.raise_for_status()
            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(r.text)
        except Exception as e:
            logger.warning(f"Could not save raw HTML for {url}: {e}")

        # Save clean text
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(clean_text)

        file_size_mb = round(len(clean_text.encode('utf-8')) / (1024 * 1024), 2)
        print(f"Scraped [{scraped_meta.get('word_count', 0)} words]: {base_name}")

        metadata = self._create_content_metadata(
            url=url, title=page_title, publisher=publisher,
            filepath=txt_filepath, content_type='html',
            file_size_bytes=len(clean_text.encode('utf-8')),
            extra=extra_metadata,
        )
        metadata['scraped_author'] = scraped_meta.get('author', '')
        metadata['scraped_pub_date'] = scraped_meta.get('pub_date', '')
        metadata['scraped_description'] = scraped_meta.get('description', '')
        metadata['raw_html_path'] = html_filepath

        if self.db:
            content_data = {
                'title': page_title,
                'content_type': 'html',
                'source_type': 'html',
                'source_name': publisher or scraped_meta.get('domain', 'Unknown'),
                'file_path': txt_filepath,
                'audio_url': 'N/A',
                'transcript': clean_text,
                'pub_date': scraped_meta.get('pub_date', '')
                            or (extra_metadata or {}).get('pub_date', ''),
                'duration_seconds': None,
                'file_size_mb': file_size_mb,
                'content_hash': None,
                'segments': [],
                'metadata': metadata,
            }
            content_id = self.db.add_content_metadata(content_data)
            if content_id:
                self.db.add_content_metadata_record(content_id, metadata)
                print(f"  DB record created: content_id={content_id}")
        else:
            metadata_file = txt_filepath.replace('.txt', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        if self.health_tracker:
            self.health_tracker.log_download(
                source_url=url, item_title=page_title,
                adapter=adapter, success=True, content_type='html',
            )

        return metadata

    def get_file_path(self, content_type, filename, source_name=None):
        clean_filename = self._sanitize_filename(filename)
        
        if content_type == 'audio':
            return os.path.join(self.audio_dir, clean_filename)
        elif content_type == 'text':
            return os.path.join(self.text_dir, clean_filename)
        elif content_type == 'pdf':
            return os.path.join(self.pdf_dir, clean_filename)
        elif content_type == 'video':
            return os.path.join(self.video_dir, clean_filename)
        else:
            return os.path.join(self.content_path, clean_filename)

if __name__ == '__main__':
    from device_config import config

    print("ContentSources module loaded.")

    content_path = config.MEDIA_DIR
    
    sources = ContentSources(content_path)
    
    MANUFACTURING_PODCAST_RSS = [
        "https://feeds.captivate.fm/manufacturing-happy-hour/",
        "https://feeds.transistor.fm/manufacturing-hub",
        "https://feeds.buzzsprout.com/1027735.rss",
        "https://feeds.castos.com/8j1v",
        "https://feeds.resonaterecordings.com/the-manufacturing-executive-podcast",
        "https://rss.libsyn.com/shows/219497/destinations/2381312.xml",
        "https://rss.libsyn.com/shows/61271/destinations/237805.xml",
    ]

    sources.get_podcasts(MANUFACTURING_PODCAST_RSS,1)

    

