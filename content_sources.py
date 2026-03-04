import os
import re
import json
from datetime import datetime
import feedparser
import requests
from pathlib import Path

class ContentSources:
    def __init__(self, content_path):
        self.content_path = content_path
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
                
                if os.path.exists(filepath):
                    print(f"Already downloaded: {filename}")
                    metadata_file = filepath.replace('.mp3', '_metadata.json')
                    if not os.path.exists(metadata_file):
                        print(f"Creating missing metadata for: {filename}")
                        metadata = self._create_episode_metadata(ep, podcast_name, filepath, audio_url)
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    continue
                
                metadata = self._create_episode_metadata(ep, podcast_name, filepath, audio_url)
                print(f"Downloading: {ep.title}")
                try:
                    r = requests.get(audio_url, timeout=30)
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded: {filename}")
                    
                    metadata_file = filepath.replace('.mp3', '_metadata.json')
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    print(f"Metadata saved: {metadata_file}")
                    
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

    def get_pending_episodes(self):
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
                else:
                    print(f"Creating missing metadata for: {filename}")
                    metadata = {
                        'podcast_name': 'Unknown',
                        'episode_title': Path(filename).stem,
                        'file_path': filepath,
                        'processing_status': 'downloaded',
                        'download_timestamp': datetime.now().isoformat()
                    }
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    pending_episodes.append(metadata)
        
        return pending_episodes
    
    def mark_episode_processed(self, filepath, status='processed'):
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
    from dotenv import load_dotenv
    load_dotenv()

    print("ContentSources module loaded.")  
     
    content_path = os.getenv("MEDIA_DIR", "~/Documents/ai-projects/ai_industry_signals/media")
    
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

    

