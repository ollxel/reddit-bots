import pandas as pd
import time
import json
from typing import List, Dict, Optional
from datetime import datetime
import os
import sys

# Add YARS to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from yars.yars import YARS


class RedditParser:
    
    def __init__(self):
        self.miner = YARS()
        self.collected_data = []
    
    def save_to_csv(self, filename: str = None) -> str:
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_comments_{timestamp}.csv"
        
        df = pd.DataFrame(self.collected_data)
        df.to_csv(filename, index=False)
        print(f"\nData saved to '{filename}'")
        return filename
    
    def ask_continue(self) -> bool:
        response = input("\nContinue parsing? (yes/no): ").strip().lower()
        return response in ['yes', 'y']
    
    def get_user_info(self, username: str) -> Dict:
        """Fetch user information using YARS"""
        try:
            user_data = self.miner.scrape_user_data(username, limit=1)
            if user_data and len(user_data) > 0:
                # Extract account age and karma if available
                account_created = user_data[0].get('account_created', None)
                total_karma = user_data[0].get('total_karma', 0)
                
                if account_created:
                    account_age_days = (time.time() - account_created) / 86400
                else:
                    account_age_days = None
                    
                return {
                    'account_age_days': round(account_age_days, 2) if account_age_days else None,
                    'total_karma': total_karma
                }
        except Exception as e:
            print(f"Warning: Could not fetch user info for {username}: {e}")
        
        return {'account_age_days': None, 'total_karma': 0}
    
    def process_comments(self, comments: List[Dict], post_author: str, 
                        post_time: float, post_title: str = None, 
                        post_url: str = None, target_comments: int = 700) -> bool:
        """
        Process comments and add to collected data
        Returns True if should continue, False if should stop
        """
        for comment in comments:
            author = comment.get('author', None)
            
            # Skip deleted comments or post author's comments
            if not author or author == post_author or author == '[deleted]':
                continue
            
            try:
                # Get user information
                user_info = self.get_user_info(author)
                
                # Calculate time difference
                comment_created = comment.get('created_utc', 0)
                time_diff_seconds = int(comment_created - post_time) if comment_created else 0
                
                comment_data = {
                    'username': author,
                    'account_age_days': user_info['account_age_days'],
                    'total_karma': user_info['total_karma'],
                    'time_diff_seconds': time_diff_seconds,
                    'comment_id': comment.get('id', ''),
                    'comment_score': comment.get('score', 0)
                }
                
                # Add post information if available
                if post_title:
                    comment_data['post_title'] = post_title
                if post_url:
                    comment_data['post_url'] = post_url
                
                self.collected_data.append(comment_data)
                
                total_collected = len(self.collected_data)
                
                if total_collected % 100 == 0:
                    print(f"  Progress: {total_collected} comments")
                
                # Check if target reached
                if total_collected >= target_comments:
                    print(f"\n{'='*60}")
                    print(f"Reached target: {total_collected} comments")
                    print(f"{'='*60}")
                    self.save_to_csv()
                    
                    if not self.ask_continue():
                        return False
                    else:
                        print(f"Continuing parsing...")
                        return True
                
            except Exception as e:
                print(f"Warning: Error processing comment: {e}")
                continue
        
        return True
    
    def parse_post_comments(self, post_url: str, target_comments: int = 700) -> pd.DataFrame:
        """Parse comments from a single post URL"""
        try:
            # Extract permalink from URL
            if 'reddit.com' in post_url:
                permalink = post_url.split('reddit.com')[1]
            else:
                permalink = post_url
            
            print(f"Parsing post from: {post_url}")
            
            # Scrape post details including comments
            post_details = self.miner.scrape_post_details(permalink)
            
            if not post_details:
                print("Failed to scrape post details")
                return pd.DataFrame()
            
            post_author = post_details.get('author', None)
            post_time = post_details.get('created_utc', time.time())
            post_title = post_details.get('title', '')
            
            print(f"Post title: {post_title}")
            print(f"Post author: {post_author}")
            
            comments = post_details.get('comments', [])
            print(f"Found {len(comments)} comments")
            
            # Process comments
            self.process_comments(comments, post_author, post_time, 
                                post_title, post_url, target_comments)
            
            df = pd.DataFrame(self.collected_data)
            print(f"Collected {len(df)} comments total\n")
            return df
        
        except Exception as e:
            print(f"Error parsing post: {e}\n")
            return pd.DataFrame()
    
    def parse_subreddit_comments(self, 
                                  subreddit_name: str, 
                                  posts_limit: int = 10,
                                  category: str = 'hot',
                                  time_filter: str = 'week',
                                  target_comments: int = 700) -> pd.DataFrame:
        """
        Parse comments from multiple posts in a subreddit
        
        Args:
            subreddit_name: Name of the subreddit
            posts_limit: Number of posts to parse
            category: 'hot', 'new', 'top', 'rising'
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'
            target_comments: Target number of comments to collect
        """
        try:
            print(f"Parsing subreddit: r/{subreddit_name}")
            print(f"Category: {category}, Time filter: {time_filter}")
            print(f"Target comments: {target_comments}")
            print("-" * 60)
            
            # Fetch subreddit posts
            posts = self.miner.fetch_subreddit_posts(
                subreddit_name, 
                limit=posts_limit, 
                category=category, 
                time_filter=time_filter
            )
            
            if not posts:
                print("No posts found")
                return pd.DataFrame()
            
            print(f"Found {len(posts)} posts")
            
            post_count = 0
            session_comments = []
            
            for post in posts:
                post_count += 1
                title = post.get('title', '')[:60]
                print(f"\n[{post_count}/{len(posts)}] {title}...")
                
                # Get post details with comments
                permalink = post.get('permalink', '')
                post_details = self.miner.scrape_post_details(permalink)
                
                if not post_details:
                    print("  Could not fetch post details")
                    continue
                
                post_author = post_details.get('author', None)
                post_time = post_details.get('created_utc', time.time())
                post_title = post_details.get('title', '')
                post_url = post.get('permalink', '')
                
                comments = post_details.get('comments', [])
                print(f"  Found {len(comments)} comments")
                
                # Process comments
                initial_count = len(self.collected_data)
                should_continue = self.process_comments(
                    comments, post_author, post_time, 
                    post_title, post_url, target_comments
                )
                
                session_comments.extend(self.collected_data[initial_count:])
                
                print(f"  Session: {len(session_comments)} | Total: {len(self.collected_data)}")
                
                if not should_continue:
                    df = pd.DataFrame(self.collected_data)
                    return df
                
                # Update target if we continued
                if len(self.collected_data) >= target_comments:
                    target_comments += 700
                    session_comments = []
            
            df = pd.DataFrame(self.collected_data)
            print("\n" + "=" * 60)
            print(f"Parsing complete: {len(df)} total comments")
            print("=" * 60 + "\n")
            self.save_to_csv()
            return df
        
        except Exception as e:
            print(f"Error parsing subreddit: {e}\n")
            return pd.DataFrame()
    
    def parse_multiple_posts(self, post_urls: List[str], target_comments: int = 700) -> pd.DataFrame:
        """Parse comments from multiple post URLs"""
        
        for i, url in enumerate(post_urls, 1):
            print(f"\n[{i}/{len(post_urls)}] Parsing post...")
            
            try:
                # Extract permalink
                if 'reddit.com' in url:
                    permalink = url.split('reddit.com')[1]
                else:
                    permalink = url
                
                # Scrape post details
                post_details = self.miner.scrape_post_details(permalink)
                
                if not post_details:
                    print(f"Failed to scrape post: {url}")
                    continue
                
                post_author = post_details.get('author', None)
                post_time = post_details.get('created_utc', time.time())
                post_title = post_details.get('title', '')
                
                comments = post_details.get('comments', [])
                print(f"  Found {len(comments)} comments")
                
                # Process comments
                should_continue = self.process_comments(
                    comments, post_author, post_time,
                    post_title, url, target_comments
                )
                
                if not should_continue:
                    df = pd.DataFrame(self.collected_data)
                    return df
                
                # Update target if we continued
                if len(self.collected_data) >= target_comments:
                    target_comments += 700
            
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
                continue
        
        df = pd.DataFrame(self.collected_data)
        print("\nParsing complete!")
        self.save_to_csv()
        return df


if __name__ == "__main__":
    # Initialize parser
    parser = RedditParser()
    
    # Example 1: Parse comments from a subreddit
    df = parser.parse_subreddit_comments(
        "AskReddit", 
        posts_limit=10, 
        category='hot',
        time_filter='week',
        target_comments=700
    )
    
    # Example 2: Parse comments from a single post
    # df = parser.parse_post_comments(
    #     "https://www.reddit.com/r/getdisciplined/comments/1frb5ib/what_single_health_test_or_practice_has/",
    #     target_comments=700
    # )
    
    # Example 3: Parse comments from multiple posts
    # post_urls = [
    #     "https://www.reddit.com/r/AskReddit/comments/...",
    #     "https://www.reddit.com/r/AskReddit/comments/...",
    # ]
    # df = parser.parse_multiple_posts(post_urls, target_comments=700)
    
    # Display results
    if not df.empty:
        print("\nFirst rows:")
        print(df.head())
        print("\nStatistics:")
        print(df.describe())
        print(f"\nTotal comments collected: {len(df)}")
