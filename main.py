import pandas as pd
import time
import json
from datetime import datetime
import requests
from typing import List, Dict, Optional, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class RedditParser:
    """
    Reddit comment parser using public JSON endpoints
    Collects ONE comment per unique user with reply_delay, karma, and account_age
    """

    def __init__(self, user_agent: str = "RedditParser/1.0"):
        self.collected_data = []
        self.processed_users = set()  # Track unique usernames
        self.target_comments = None
        # Arrays for storing data (commented out as requested)
        # self.reply_delays = []
        # self.user_karmas = []
        # self.account_ages = []
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent
        })
        # Rate limiting  
        self.last_request_time = 0
        self.min_request_interval = 5  # seconds between requests (increased to 5 to avoid 429 errors)

    def _rate_limit(self):
        """Ensure we don't make requests too quickly"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        self._rate_limit()
    
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Request error for {url}: {e}")
                    return None
                wait = 2 ** attempt  # экспоненциальная задержка
                print(f"Request failed, retrying in {wait}s...")
                time.sleep(wait)
        return None

    def get_user_info(self, username: str) -> Dict[str, Any]:
        """Fetch user information from Reddit"""
        url = f"https://www.reddit.com/user/{username}/about.json"
        
        try:
            data = self._make_request(url)
            
            if data and 'data' in data:
                user_data = data['data']
                account_created = user_data.get('created_utc')
                if account_created is not None:
                    try:
                        account_created = float(account_created)
                        account_age_days = (time.time() - account_created) / 86400
                    except (TypeError, ValueError):
                        account_age_days = None
                else:
                    account_age_days = None

                link_karma = int(user_data.get('link_karma', 0) or 0)
                comment_karma = int(user_data.get('comment_karma', 0) or 0)
                total_karma = link_karma + comment_karma
                
                return {
                    'account_age_days': round(account_age_days, 2) if account_age_days else None,
                    'user_karma': total_karma
                }
        except Exception as e:
            print(f"Warning: Could not fetch user info for {username}: {e}")
        
        return {'account_age_days': None, 'user_karma': 0}

    def fetch_subreddit_posts(self, 
                             subreddit_name: str,
                             limit: int = 25,
                             category: str = 'hot',
                             time_filter: str = 'week') -> List[Dict]:
        """
        Fetch posts from a subreddit
        
        Args:
            subreddit_name: Name of subreddit (without r/)
            limit: Number of posts to fetch (max 100 per request)
            category: 'hot', 'new', 'top', 'rising'
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all' (for 'top')
        """
        url = f"https://www.reddit.com/r/{subreddit_name}/{category}.json"
        params = {'limit': min(limit, 100)}
        
        if category == 'top':
            params['t'] = time_filter
        
        data = self._make_request(url, params)
        
        if not data or 'data' not in data:
            return []
        
        posts = []
        for child in data['data'].get('children', []):
            if child['kind'] == 't3':  # t3 is a post
                post_data = child['data']
                posts.append({
                    'title': post_data.get('title', ''),
                    'author': post_data.get('author', ''),
                    'permalink': post_data.get('permalink', ''),
                    'created_utc': post_data.get('created_utc', 0),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'id': post_data.get('id', '')
                })
        
        return posts

    def _extract_comments(self, comments_data: List, all_comments: List = None) -> List[Dict]:
        """Recursively extract all comments from Reddit's nested structure"""
        if all_comments is None:
            all_comments = []
        
        for item in comments_data:
            if isinstance(item, dict):
                kind = item.get('kind', '')
                data = item.get('data', {})
                
                if kind == 't1':  # t1 is a comment
                    comment = {
                        'id': data.get('id', ''),
                        'author': data.get('author', ''),
                        'body': data.get('body', ''),
                        'score': data.get('score', 0),
                        'created_utc': data.get('created_utc', 0),
                    }
                    all_comments.append(comment)
                    
                    # Recursively get replies
                    replies = data.get('replies', '')
                    if isinstance(replies, dict) and 'data' in replies:
                        children = replies['data'].get('children', [])
                        self._extract_comments(children, all_comments)
                
                elif kind == 'Listing':
                    # This is a listing of comments
                    children = data.get('children', [])
                    self._extract_comments(children, all_comments)
        
        return all_comments

    def scrape_post_details(self, permalink: str) -> Optional[Dict]:
        """
        Scrape post details including all comments
        
        Args:
            permalink: Post permalink (e.g., /r/subreddit/comments/id/title/)
        """
        # Ensure permalink starts with /
        if not permalink.startswith('/'):
            permalink = '/' + permalink
        
        url = f"https://www.reddit.com{permalink}.json"
        
        data = self._make_request(url)
        
        if not data or len(data) < 2:
            return None
        
        # First element is the post, second is comments
        post_listing = data[0]
        comments_listing = data[1]
        
        # Extract post data
        if 'data' not in post_listing or 'children' not in post_listing['data']:
            return None
        
        post_data = post_listing['data']['children'][0]['data']
        
        # Extract all comments
        comments = []
        if 'data' in comments_listing:
            children = comments_listing['data'].get('children', [])
            comments = self._extract_comments(children)
        
        return {
            'title': post_data.get('title', ''),
            'author': post_data.get('author', ''),
            'created_utc': post_data.get('created_utc', 0),
            'score': post_data.get('score', 0),
            'num_comments': post_data.get('num_comments', 0),
            'selftext': post_data.get('selftext', ''),
            'url': post_data.get('url', ''),
            'permalink': post_data.get('permalink', ''),
            'comments': comments
        }

    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """Save collected data to CSV"""
        if filename is None:
            filename = f"reddit_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
        df = pd.DataFrame(self.collected_data)
        df.to_csv(filename, index=False)
        print(f"\nData saved to '{filename}'")
        return filename

    def ask_continue(self) -> bool:
        """Ask user if they want to continue parsing"""
        response = input("\nContinue parsing? (yes/no): ").strip().lower()
        return response in ['yes', 'y']

    def process_comments(self, 
                        comments: List[Dict],
                        post_author: str,
                        post_time: float,
                        post_title: Optional[str] = None,
                        post_url: Optional[str] = None,
                        target_comments: int = 700) -> bool:
        """
        Process comments and add to collected data
        ONLY ONE COMMENT PER USER - subsequent comments from same user are skipped
        
        Returns True if should continue, False if should stop
        """
        for comment in comments:
            author = comment.get('author', None)
            
            # Skip deleted comments or post author's comments
            if not author or author == post_author or author == '[deleted]':
                continue
            
            # CRITICAL: Skip if we already processed this user
            if author in self.processed_users:
                continue
            
            try:
                # Get user information
                user_info = self.get_user_info(author)
                
                # Calculate reply delay (time between post and comment)
                comment_created = comment.get('created_utc', 0)
                reply_delay_seconds = int(comment_created - post_time) if comment_created else 0
                
                # Collect the three required values
                account_age_days = user_info['account_age_days']
                user_karma = user_info['user_karma']
                
                # Skip if we couldn't get user data
                if account_age_days is None:
                    print(f"  Skipping {author}: no account data")
                    continue
                
                # Create data entry with the three main fields
                comment_data = {
                    'reply_delay_seconds': reply_delay_seconds,
                    'user_karma': user_karma,
                    'account_age_days': account_age_days,
                }
                
                # Add optional metadata
                comment_data['username'] = author
                comment_data['comment_id'] = comment.get('id', '')
                comment_data['comment_score'] = comment.get('score', 0)
                
                if post_title:
                    comment_data['post_title'] = post_title
                if post_url:
                    comment_data['post_url'] = post_url
                
                # Add to collected data
                self.collected_data.append(comment_data)
                
                # Mark user as processed
                self.processed_users.add(author)
                
                # Add to arrays (commented out)
                # self.reply_delays.append(reply_delay_seconds)
                # self.user_karmas.append(user_karma)
                # self.account_ages.append(account_age_days)
                
                total_collected = len(self.collected_data)

                if total_collected >= self.target_comments:
                    print(f"\n{'='*60}")
                    print(f"Reached target: {total_collected} unique users")
                    print(f"{'='*60}")
                    self.save_to_csv()
                    if not self.ask_continue():
                        return False
                    else:
                        print("Continuing parsing...")
                        self.target_comments += 700   # увеличиваем атрибут
                        return True
            
            except Exception as e:
                print(f"Warning: Error processing comment from {author}: {e}")
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
            post_details = self.scrape_post_details(permalink)
            
            if not post_details:
                print("Failed to scrape post details")
                return pd.DataFrame()
            
            post_author = post_details.get('author', None)
            post_time = post_details.get('created_utc', time.time())
            post_title = post_details.get('title', '')
            
            print(f"Post title: {post_title}")
            print(f"Post author: {post_author}")
            
            comments = post_details.get('comments', [])
            print(f"Found {len(comments)} total comments")
            
            # Process comments
            self.process_comments(comments, post_author, post_time,
                                post_title, post_url, target_comments)
            
            df = pd.DataFrame(self.collected_data)
            print(f"Collected {len(df)} unique users\n")
            return df
        
        except Exception as e:
            print(f"Error parsing post: {e}\n")
            return pd.DataFrame()

    def parse_subreddit_comments(self, subreddit_name: str, posts_limit: int = 10,
                             category: str = 'hot', time_filter: str = 'week',
                             target_comments: int = 700) -> pd.DataFrame:
        
        self.target_comments = target_comments   # сохраняем
        try:
            print(f"Parsing subreddit: r/{subreddit_name}")
            print(f"Category: {category}, Time filter: {time_filter}")
            print(f"Target: {target_comments} unique users")
            print("-" * 60)
            
            # Fetch subreddit posts
            posts = self.fetch_subreddit_posts(
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
            
            for post in posts:
                post_count += 1
                title = post.get('title', '')[:60]
                print(f"\n[{post_count}/{len(posts)}] {title}...")
                
                # Get post details with comments
                permalink = post.get('permalink', '')
                post_details = self.scrape_post_details(permalink)
                
                if not post_details:
                    print("  Could not fetch post details")
                    continue
                
                post_author = post_details.get('author', None)
                post_time = post_details.get('created_utc', time.time())
                post_title = post_details.get('title', '')
                post_url = post.get('permalink', '')
                
                comments = post_details.get('comments', [])
                print(f"  Found {len(comments)} total comments")
                
                # Process comments
                initial_count = len(self.collected_data)
                should_continue = self.process_comments(
                    comments, post_author, post_time,
                    post_title, post_url)
                
                new_users = len(self.collected_data) - initial_count
                print(f"  Added: {new_users} new unique users | Total: {len(self.collected_data)}")
                
                if not should_continue:
                    df = pd.DataFrame(self.collected_data)
                    return df
                
                # Update target if we continued
                if len(self.collected_data) >= target_comments:
                    target_comments += 700
            
            df = pd.DataFrame(self.collected_data)
            print("\n" + "=" * 60)
            print(f"Parsing complete: {len(df)} unique users")
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
                post_details = self.scrape_post_details(permalink)
                
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

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        if not self.collected_data:
            return {}
        
        df = pd.DataFrame(self.collected_data)
        
        stats = {
            'total_unique_users': len(df),
            'avg_reply_delay': df['reply_delay_seconds'].mean(),
            'avg_user_karma': df['user_karma'].mean(),
            'avg_account_age': df['account_age_days'].mean(),
            'median_reply_delay': df['reply_delay_seconds'].median(),
            'median_user_karma': df['user_karma'].median(),
            'median_account_age': df['account_age_days'].median(),
        }
        
        return stats

class Model:  # Переименовал с маленькой буквы на заглавную (PEP 8)
    @staticmethod
    def prepare_data(filepath: str = 'reddit_dead_internet_analysis_2026.csv'):
        """Загружает и подготавливает обучающие данные"""
        
        data = pd.read_csv(filepath)
        print(f"✓ Загружено {len(data)} строк для обучения")
        
        # Безопасное удаление колонок
        columns_to_drop = ['subreddit', 'comment_id', 'bot_type_label', 
                          'bot_probability', 'contains_links', 'avg_word_length']
        data = data.drop(columns=[c for c in columns_to_drop if c in data.columns], errors='ignore')
        
        # Создаём новый признак, если есть нужные колонки
        required_cols = ['account_age_days', 'user_karma', 'reply_delay_seconds']
        if all(col in data.columns for col in required_cols):
            data['hum_val'] = (data['account_age_days'] + data['user_karma']) * data['reply_delay_seconds']
            data = data.drop(columns=required_cols)
        
        return data
    
    @staticmethod
    def prepare_parser_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготавливает данные из парсера для предсказания.
        Применяет те же преобразования, что и при обучении.
        """
        data = df.copy()
        
        # Проверяем наличие нужных колонок
        required_cols = ['account_age_days', 'user_karma', 'reply_delay_seconds']
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(f"❌ В данных парсера нет колонок: {missing}")
        
        # Создаём тот же признак hum_val
        data['hum_val'] = (data['account_age_days'] + data['user_karma']) * data['reply_delay_seconds']
        
        # Сохраняем username для вывода
        usernames = data['username'].copy() if 'username' in data.columns else None
        
        # Оставляем только признаки (как в обучении)
        X = data[['hum_val']].fillna(0)
        
        return X, usernames
    
    @staticmethod
    def train_model(data):
        """Обучает модель и возвращает классификатор"""
        
        if 'is_bot_flag' not in data.columns:
            raise ValueError("❌ В данных нет колонки 'is_bot_flag'")
        
        y = data['is_bot_flag']
        X = data.drop(columns=['is_bot_flag', 'sentiment_score']).fillna(0)
        
        stratify_param = y if len(y.unique()) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_param
        )
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print("\n" + "="*60)
        print("📊 ОТЧЁТ ОБ ОБУЧЕНИИ")
        print("="*60)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return clf
    
    @staticmethod
    def predict(clf, df: pd.DataFrame):
        """
        Делает предсказания на данных парсера.
        Возвращает DataFrame с username и prediction.
        """
        # Подготавливаем данные (те же преобразования что при обучении)
        X, usernames = Model.prepare_parser_data(df)
        
        predictions = clf.predict(X)
        probabilities = clf.predict_proba(X) if hasattr(clf, 'predict_proba') else None
        
        # Создаём результат с username
        result = pd.DataFrame({
            'username': usernames if usernames is not None else range(len(predictions)),
            'is_bot_flag': predictions,
        })
        
        if probabilities is not None:
            result['bot_probability'] = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        print(f"\n✓ Предсказаний: {len(predictions)}")
        print(f"  Боты: {sum(predictions)}, Люди: {len(predictions) - sum(predictions)}")
        
        return result


if __name__ == "__main__":
    # === ЧАСТЬ 1: Парсинг ===
    parse = input("Start parse? y/n: ").strip().lower()
    df_raw = None
    
    if parse in ["y", "yes"]:
        parser = RedditParser(user_agent="RedditDataCollector/1.0 (Educational)")
        df_raw = parser.parse_subreddit_comments(
            "AskReddit",
            posts_limit=10,
            category='hot',
            time_filter='week',
            target_comments=100
        )
        if not df_raw.empty:
            print(f"\n✓ Собрано {len(df_raw)} уникальных пользователей")
            parser.save_to_csv("collected_reddit_data.csv")
    
    # === ЧАСТЬ 2: Модель ===
    run_model = input("\nRun ML model? y/n: ").strip().lower()
    
    if run_model in ["y", "yes"]:
        try:
            print("\n[1/3] Загрузка обучающих данных...")
            train_data = Model.prepare_data('reddit_dead_internet_analysis_2026.csv')
            
            print("\n[2/3] Обучение модели...")
            clf = Model.train_model(train_data)
            
            print("\n[3/3] Предсказание...")
            
            # Выбираем источник данных для предсказания
            if df_raw is not None and not df_raw.empty:
                print("  → Используем данные из парсера")
                predict_source = df_raw
            else:
                # Если нет данных парсера, пробуем загрузить из CSV
                csv_file = input("  → Введите путь к CSV с данными для предсказания (Enter для collected_reddit_data.csv): ").strip()
                if not csv_file:
                    csv_file = 'collected_reddit_data.csv'
                predict_source = pd.read_csv(csv_file)
                print(f"  → Загружено {len(predict_source)} записей из {csv_file}")
            
            # Делаем предсказание
            predictions_df = Model.predict(clf, predict_source)
            
            # Сохраняем результат
            predictions_df.to_csv('model_predictions.csv', index=False)
            print("\n" + "="*60)
            print("📋 РЕЗУЛЬТАТЫ ПРЕДСКАЗАНИЯ")
            print("="*60)
            print(predictions_df.head(10))
            print(f"\n✓ Результаты сохранены в 'model_predictions.csv'")
            print(f"  Всего пользователей: {len(predictions_df)}")
            print(f"  Ботов обнаружено: {predictions_df['is_bot_flag'].sum()}")
            print(f"  Людей: {len(predictions_df) - predictions_df['is_bot_flag'].sum()}")
            
        except FileNotFoundError as e:
            print(f"❌ Файл не найден: {e}")
        except KeyError as e:
            print(f"❌ Ошибка: в данных нет колонки {e}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()