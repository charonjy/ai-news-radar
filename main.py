"""
AI 资讯雷达：抓取 → 去重 → AI 打分总结 → 推送飞书
由 GitHub Actions 定时运行，一般不需要改这个文件，改 config.yaml 即可。
"""
import base64
import hashlib
import hmac
import html
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")
SEEN_PATH = os.path.join(ROOT, "data", "seen.json")
SEEN_KEEP_DAYS = 14
UA = "Mozilla/5.0 (compatible; AINewsRadar/1.0; +https://github.com)"
MODELS_URL = "https://models.github.ai/inference/chat/completions"
BJT = timezone(timedelta(hours=8))


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 抓取
def _text(el, *names):
    """在元素下按顺序找第一个存在的子节点文本（忽略命名空间）"""
    if el is None:
        return ""
    for name in names:
        for child in el:
            if child.tag.split("}")[-1] == name:
                if name == "link" and child.get("href"):
                    if child.get("rel", "alternate") == "alternate":
                        return child.get("href")
                    continue
                return (child.text or "").strip()
    return ""


def _parse_date(s):
    if not s:
        return None
    try:
        d = parsedate_to_datetime(s)
    except Exception:
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


def _clean(s, n=400):
    s = re.sub(r"<[^>]+>", " ", html.unescape(s or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:n]


def parse_feed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    items = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        link = _text(el, "link")
        items.append({
            "title": _clean(_text(el, "title"), 200),
            "link": link,
            "guid": _text(el, "guid", "id") or link,
            "summary": _clean(_text(el, "description", "summary", "content")),
            "published": _parse_date(_text(el, "pubDate", "published", "updated", "date")),
        })
    return items


def fetch_source(src):
    r = requests.get(src["url"], headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return parse_feed(r.content)


def collect(cfg, seen):
    max_age = timedelta(hours=cfg["settings"].get("max_age_hours", 48))
    now = datetime.now(timezone.utc)
    fresh, all_ids, errors = [], [], []
    for src in cfg["sources"]:
        try:
            items = fetch_source(src)
        except Exception as e:
            errors.append(f"{src['name']}: {e}")
            log(f"  ✗ {src['name']}: {e}")
            continue
        exclude = src.get("exclude")
        kept = 0
        for it in items:
            if not it["title"]:
                continue
            if exclude and re.search(exclude, it["title"], re.I):
                continue
            if it["published"] and now - it["published"] > max_age:
                continue
            iid = hashlib.sha1((it["guid"] or it["title"]).encode()).hexdigest()[:16]
            all_ids.append(iid)
            if iid in seen:
                continue
            it["id"], it["source"] = iid, src["name"]
            fresh.append(it)
            kept += 1
        log(f"  ✓ {src['name']}: {len(items)} 条，新增 {kept}")
    # 同一批次内按标题粗去重
    uniq, titles = [], set()
    for it in fresh:
        key = re.sub(r"\W+", "", it["title"].lower())[:60]
        if key in titles:
            continue
        titles.add(key)
        uniq.append(it)
    return uniq, all_ids, errors


# ---------------------------------------------------------------- AI 分析
PROMPT = """你是一名 AI 行业资讯编辑。下面是一批新抓到的资讯（JSON），请：
1. 合并讲同一件事的条目（把它们的 id 都放进 ids）；
2. 按读者关注点给每件事打 1~5 分（5=重磅一手消息，如新模型发布/额度重置/关键人物重要表态；3=值得一看；1=无关或噪音）；
3. 用中文写一个简短标题（≤30字）和摘要（≤80字，说清楚“发生了什么、为什么重要”）。

读者关注点：
{interests}

只输出 JSON，格式：
{{"items":[{{"ids":["..."],"score":4,"title":"...","summary":"..."}}]}}

资讯：
{data}"""


def analyze(cfg, items):
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("没有 GITHUB_TOKEN，无法调用模型")
    by_id = {it["id"]: it for it in items}
    results = []
    for i in range(0, len(items), 25):
        batch = items[i:i + 25]
        data = json.dumps([{"id": it["id"], "source": it["source"], "title": it["title"],
                            "summary": it["summary"][:300]} for it in batch], ensure_ascii=False)
        body = {
            "model": cfg["settings"].get("model", "openai/gpt-4.1-mini"),
            "messages": [{"role": "user", "content": PROMPT.format(interests=cfg.get("interests", ""), data=data)}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        r = requests.post(MODELS_URL, json=body, timeout=90,
                          headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
        for x in json.loads(text).get("items", []):
            ids = [k for k in x.get("ids", []) if k in by_id]
            if not ids:
                continue
            src = by_id[ids[0]]
            results.append({
                "score": int(x.get("score", 1)),
                "title": x.get("title") or src["title"],
                "summary": x.get("summary", ""),
                "links": [(by_id[k]["source"], by_id[k]["link"]) for k in ids],
            })
    return results


def fallback(items):
    """模型不可用时：不打分，直接推原标题"""
    return [{"score": 3, "title": it["title"], "summary": "",
             "links": [(it["source"], it["link"])]} for it in items[:15]]


# ---------------------------------------------------------------- 飞书推送
def feishu_send(card):
    url = os.environ.get("FEISHU_WEBHOOK")
    if not url:
        log("（未配置 FEISHU_WEBHOOK，以下为将要推送的内容）")
        log(json.dumps(card, ensure_ascii=False, indent=1)[:3000])
        return
    payload = {"msg_type": "interactive", "card": card}
    secret = os.environ.get("FEISHU_SECRET")
    if secret:
        ts = str(int(time.time()))
        sign = base64.b64encode(hmac.new(f"{ts}\n{secret}".encode(), digestmod=hashlib.sha256).digest()).decode()
        payload.update(timestamp=ts, sign=sign)
    r = requests.post(url, json=payload, timeout=20)
    res = r.json()
    if res.get("code", 0) != 0:
        raise RuntimeError(f"飞书返回错误：{res}")


def make_card(title, lines, color="blue"):
    return {
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": title}, "template": color},
        "elements": [{"tag": "markdown", "content": "\n\n".join(lines)}],
    }


def push(cfg, results, note=""):
    per = cfg["settings"].get("items_per_card", 10)
    results.sort(key=lambda x: -x["score"])
    stamp = datetime.now(BJT).strftime("%m-%d %H:%M")
    for i in range(0, len(results), per):
        chunk = results[i:i + per]
        lines = []
        for r in chunk:
            fire = "🔥" * max(1, r["score"] - 2)
            links = " · ".join(f"[{s}]({u})" for s, u in r["links"][:3])
            line = f"**{fire} {r['title']}**"
            if r["summary"]:
                line += f"\n{r['summary']}"
            lines.append(line + f"\n{links}")
        if note and i == 0:
            lines.append(f"<font color='grey'>{note}</font>")
        color = "red" if chunk[0]["score"] >= 5 else "blue"
        feishu_send(make_card(f"AI 资讯雷达 · {stamp}（{len(chunk)} 条）", lines, color))


# ---------------------------------------------------------------- 主流程
def load_seen():
    try:
        with open(SEEN_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_seen(seen):
    cutoff = time.time() - SEEN_KEEP_DAYS * 86400
    seen = {k: v for k, v in seen.items() if v > cutoff}
    os.makedirs(os.path.dirname(SEEN_PATH), exist_ok=True)
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=0, sort_keys=True)


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seen = load_seen()
    first_run = not seen
    log(f"开始抓取 {len(cfg['sources'])} 个信息源…")
    items, all_ids, errors = collect(cfg, seen)
    now = time.time()

    if first_run:
        # 首次运行：只记录现有内容，不推送旧消息，避免刷屏
        for iid in all_ids:
            seen[iid] = now
        save_seen(seen)
        ok = len(cfg["sources"]) - len(errors)
        lines = [f"✅ 已上线，{ok}/{len(cfg['sources'])} 个信息源可用，记录了 {len(all_ids)} 条现有内容。",
                 "之后有新消息会自动推送到这里。"]
        if errors:
            lines.append("⚠️ 暂时抓不到：\n" + "\n".join(f"- {e[:80]}" for e in errors))
        feishu_send(make_card("AI 资讯雷达 · 启动成功", lines, "green"))
        return

    log(f"新内容 {len(items)} 条")
    if not items:
        save_seen(seen)
        return

    note = ""
    try:
        results = analyze(cfg, items)
    except Exception as e:
        log(f"AI 分析失败，改为直接推送标题：{e}")
        results, note = fallback(items), "（AI 分析暂不可用，以上为原始标题）"

    min_score = cfg["settings"].get("min_score", 3)
    picked = [r for r in results if r["score"] >= min_score]
    log(f"达到推送标准 {len(picked)} 条")
    if picked:
        push(cfg, picked, note)

    for it in items:  # 推送成功后才标记为已读
        seen[it["id"]] = now
    save_seen(seen)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"运行失败：{e}")
        sys.exit(1)
