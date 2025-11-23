from atproto import Client
from datetime import datetime, timezone
import json, time, math, logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from httpx/httpcore
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc) # Jan 1st
END   = datetime(2025, 2, 7, 0, 0, tzinfo=timezone.utc) # Feb 7th

POST_FILE  = '../data/bluesky/ca_fire_20250101_20250207.jsonl'
INTERACT_FILE= '../data/bluesky/interactions_20250101_20250207.jsonl'
LOG_FILE   = '../data/bluesky/all_read_dates.txt'

def safe_get(model, path, default=None):
    """walk dot-notation path safely:  safe_get(post, 'record.reply.parent.uri')"""
    for part in path.split('.'):
        model = getattr(model, part, None)
        if model is None:
            return default
    return model or default

def fetch_all_pages(endpoint, params):
    """generic cursor paginator that yields every item in endpoint(params)."""
    cursor = None
    while True:
        try:
            params['cursor'] = cursor
            rsp = endpoint(params)
            yield from getattr(rsp, 'feed', []) or getattr(rsp, 'likes', []) or getattr(rsp, 'reposted_by', [])
            cursor = getattr(rsp, 'cursor', None)
            if not cursor:
                break
            time.sleep(0.2)
        except Exception as e:
            logger.error(f'Paginator error: {e}')
            time.sleep(2)

def fetch_posts():
    """Fetch posts with #california + keyword 'fire' from START to END dates, storing posts and interactions"""
    total_posts = 0
    log         = open(LOG_FILE, 'w', encoding='utf8')

    # Initialize client
    logger.info('Initializing Bluesky client...')
    client = Client()
    client.login("email", "password")
    logger.info('Successfully logged in to Bluesky')

    # Base query terms
    query_terms = 'wildfire | california | evacuation'

    # Iterate day by day
    from datetime import timedelta
    current_day = START
    while current_day < END:
        next_day = current_day + timedelta(days=1)
        day_str = current_day.strftime('%Y-%m-%d')
        next_str = next_day.strftime('%Y-%m-%d')

        logger.info(f'Fetching posts for {day_str}...')
        cursor = None
        day_posts = 0

        # Paginate through all posts for this day
        while True:
            try:
                query = f'{query_terms} since:{day_str} until:{next_str}'
                rsp = client.app.bsky.feed.search_posts({'q': query, 'limit': 100, 'cursor': cursor})
            except Exception as e:
                logger.error(f'Search error: {e}')
                time.sleep(2)
                continue
            if not rsp.posts:
                break

            for p in rsp.posts:
                ts = datetime.fromisoformat(p.record.created_at.replace('Z','+00:00'))
                print(ts.date(), ts.time(), file=log)

                # ---------- core post row ----------
                row = {
                    "uri"          : p.uri,
                    "cid"          : p.cid,
                    "created_at"   : p.record.created_at,
                    "indexed_at"   : p.indexed_at,
                    "author_did"   : p.author.did,
                    "author_handle": p.author.handle,
                    "text"         : p.record.text,
                    "langs"        : getattr(p.record, 'langs', []),
                    "reply_to"     : safe_get(p, 'record.reply.parent.uri'),
                    "quoted"       : safe_get(p, 'record.embed.record.uri'),
                    "like_count"   : p.like_count or 0,
                    "repost_count" : p.repost_count or 0
                }
                with open(POST_FILE, 'a', encoding='utf8') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                total_posts += 1
                day_posts += 1

                # Update progress on same line
                print(f'\r{day_str}: {day_posts} posts downloaded', end='', flush=True)

                # ---------- interaction rows ----------
                # 1) likes
                for like in fetch_all_pages(client.app.bsky.feed.get_likes,
                                            {'uri': p.uri, 'cid': p.cid, 'limit': 100}):
                    with open(INTERACT_FILE, 'a', encoding='utf8') as f:
                        f.write(json.dumps({
                            'src_did': like.actor.did,
                            'dst_uri': p.uri,
                            'edge'   : 'like'
                        }) + '\n')

                # 2) reposts
                for rep in fetch_all_pages(client.app.bsky.feed.get_reposted_by,
                                        {'uri': p.uri, 'cid': p.cid, 'limit': 100}):
                    with open(INTERACT_FILE, 'a', encoding='utf8') as f:
                        f.write(json.dumps({
                            'src_did': rep.did,
                            'dst_uri': p.uri,
                            'edge'   : 'repost'
                        }) + '\n')

                # 3) replies (store the comment text too)
                try:
                    thread = client.app.bsky.feed.get_post_thread({'uri': p.uri, 'cid': p.cid, 'depth': 1})
                    for kid in (thread.thread.replies or []):
                        row = {
                            'src_did': kid.post.author.did,
                            'dst_uri': p.uri,
                            'edge': 'reply',
                            'reply_text': kid.post.record.text   # <-- children text
                        }
                        with open(INTERACT_FILE, 'a', encoding='utf8') as f:
                            f.write(json.dumps(row, ensure_ascii=False) + '\n')
                except Exception:
                    pass   # no thread / no replies

            cursor = getattr(rsp, 'cursor', None)
            if not cursor:
                break

        print()  # New line after day completes
        logger.info(f'Completed {day_str}: {day_posts} posts')
        current_day = next_day

    log.close()
    logger.info(f'Finished fetching data - Total posts: {total_posts}')
    logger.info(f'Interactions saved to: {INTERACT_FILE}')
    return

if __name__ == '__main__':
    fetch_posts()

