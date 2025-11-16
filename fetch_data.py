from atproto import Client
from datetime import datetime, timezone
import json, time, math

START = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc) # Jan 1st
END   = datetime(2025, 2, 7, 0, 0, tzinfo=timezone.utc) # Feb 7th

POST_FILE  = 'ca_fire_20250101_20250207.jsonl'
INTERACT_FILE= 'interactions_20250101_20250207.jsonl'
LOG_FILE   = 'all_read_dates.txt'

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
            print('paginator error:', e)
            time.sleep(2)

def fetch_posts():
    """Fetch posts with #california + keyword 'fire' from START to END dates, storing posts and interactions"""
    total_posts = 0
    cursor      = None
    log         = open(LOG_FILE, 'w', encoding='utf8')

    # Initialize client
    client = Client()
    client.login("your-username", "your-account-password/app-password")

    # Main search loop
    while True:
        try:
            rsp = client.app.bsky.feed.search_posts({'q':'#california fire', 'limit':100, 'cursor':cursor})
        except Exception as e:
            print('search error:', e); time.sleep(2); continue
        if not rsp.posts:
            break

        for p in rsp.posts:
            ts = datetime.fromisoformat(p.record.created_at.replace('Z','+00:00'))
            print(ts.date(), ts.time(), file=log)

            if ts >= END:          # too new
                continue
            if ts < START:         # too old → stop
                cursor = None
                break

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

    log.close()
    print("finished -> posts:", total_posts,
        "  (interactions appended to", INTERACT_FILE + ")")
    return
