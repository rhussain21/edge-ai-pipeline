import os
import re
import json
from datetime import datetime
import feedparser
import requests
from pathlib import Path

class ContentSources:
    """Handle content acquisition from various sources"""
    
    def __init__(self, content_path):
        self.content_path = content_path
        self._setup_directories()
    
    def _setup_directories(self):
        """Create standardized directory structure"""
        self.audio_dir = os.path.join(self.content_path, "audio")
        self.text_dir = os.path.join(self.content_path, "text")
        self.pdf_dir = os.path.join(self.content_path, "pdf")
        self.video_dir = os.path.join(self.content_path, "video")
        
        # Create all directories
        for directory in [self.audio_dir, self.text_dir, self.pdf_dir, self.video_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _sanitize_filename(self, filename):
        """Clean filename for safe file system use"""
        # Remove special characters, replace spaces with underscores
        clean_name = re.sub(r'[^\w\s-]', '', filename)
        clean_name = re.sub(r'\s+', '_', clean_name)
        return clean_name.strip('_')
    
    def get_podcasts(self, rss_links, num_episodes=None):
        """
        Download podcasts from RSS feeds.
        
        Args:
            rss_links: List of RSS feed URLs or single URL string
            num_episodes: Number of episodes per feed (int) or list of ints
        """
        # Handle single link or list of links
        if isinstance(rss_links, str):
            rss_links = [rss_links]
        
        # Handle single number or list of numbers
        if isinstance(num_episodes, int):
            num_episodes = [num_episodes] * len(rss_links)
        elif num_episodes is None:
            num_episodes = [10] * len(rss_links)
        
        # Download podcasts
        for rss_url, n in zip(rss_links, num_episodes):
            self._download_podcast_feed(rss_url, n)
    
    def _download_podcast_feed(self, rss_url, n_episodes=10):
        """Download episodes from a single RSS feed - DECOUPLED approach"""
        try:
            # Parse RSS feed
            feed = feedparser.parse(rss_url)
            if not feed.entries:
                print(f"No entries found in RSS feed: {rss_url}")
                return
            
            # Extract podcast name
            podcast_name = feed.feed.get("title", "Unknown_Podcast")
            podcast_name_clean = self._sanitize_filename(podcast_name)
            
            print(f"\nFetching latest {n_episodes} episodes from {podcast_name}")
            
            downloaded_episodes = []
            
            # Process episodes
            for ep in feed.entries[:n_episodes]:
                audio_url = ep.enclosures[0].href if ep.enclosures else None
                if not audio_url:
                    print(f"No audio found for episode: {ep.title}")
                    continue
                
                # Create standardized filename: podcast_name_episode_name
                episode_name_clean = self._sanitize_filename(ep.title)
                filename = f"{podcast_name_clean}_{episode_name_clean}.mp3"
                filepath = os.path.join(self.audio_dir, filename)
                
                # Check if already downloaded
                if os.path.exists(filepath):
                    print(f"Already downloaded: {filename}")
                    # Check if metadata exists, create if missing
                    metadata_file = filepath.replace('.mp3', '_metadata.json')
                    if not os.path.exists(metadata_file):
                        print(f"Creating missing metadata for: {filename}")
                        metadata = self._create_episode_metadata(ep, podcast_name, filepath, audio_url)
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=2)
                    continue
                
                # Extract metadata BEFORE download
                metadata = self._create_episode_metadata(ep, podcast_name, filepath, audio_url)
                
                # Download audio file
                print(f"Downloading: {ep.title}")
                try:
                    r = requests.get(audio_url, timeout=30)
                    r.raise_for_status()
                    with open(filepath, "wb") as f:
                        f.write(r.content)
                    print(f"Downloaded: {filename}")
                    
                    # Save metadata file
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
        """Create comprehensive metadata for episode"""
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
            'file_size_mb': None,  # Will be filled after download
            'processing_status': 'downloaded'  # Track processing status
        }

    def get_pending_episodes(self):
        """Get list of downloaded episodes pending ETL processing"""
        pending_episodes = []
        
        for filename in os.listdir(self.audio_dir):
            if filename.endswith('.mp3'):
                filepath = os.path.join(self.audio_dir, filename)
                metadata_file = filepath.replace('.mp3', '_metadata.json')
                
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check if already processed
                    if metadata.get('processing_status') == 'downloaded':
                        pending_episodes.append(metadata)
                else:
                    # Create metadata for existing files without metadata
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
        """Mark episode as processed to avoid reprocessing"""
        metadata_file = filepath.replace('.mp3', '_metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['processing_status'] = status
            metadata['processed_timestamp'] = datetime.now().isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def get_download_stats(self):
        """Get statistics about downloaded content"""
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
        """Get standardized file path for content type"""
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
            # Default to content_path root
            return os.path.join(self.content_path, clean_filename)

if __name__ == '__main__':
    # Example usage

    print("ContentSources module loaded.")  
     
    content_path = os.path.expanduser('~/Documents/ai-projects/ai_industry_signals/test_files')
    
    sources = ContentSources(content_path)
    
    # Example: Download from a podcast RSS feed
    MANUFACTURING_PODCAST_RSS = [
        # Manufacturing Happy Hour
        "https://feeds.captivate.fm/manufacturing-happy-hour/",

        # Manufacturing Hub (OT / IIoT / UNS heavy)
        "https://feeds.transistor.fm/manufacturing-hub",

        # EECO Asks Why (controls, motors, industrial networking)
        "https://feeds.buzzsprout.com/1027735.rss",

        # The Robot Industry Podcast
        "https://feeds.castos.com/8j1v",

        # The Manufacturing Executive
        "https://feeds.resonaterecordings.com/the-manufacturing-executive-podcast",

        # Advanced Manufacturing Now (SME)
        "https://rss.libsyn.com/shows/219497/destinations/2381312.xml",

        # MakingChips (job shops / machining)
        "https://rss.libsyn.com/shows/61271/destinations/237805.xml",
    ]

    sources.get_podcasts(MANUFACTURING_PODCAST_RSS,3)

    

