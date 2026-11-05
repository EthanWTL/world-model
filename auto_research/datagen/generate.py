# Shell Game dataset (arXiv 2608.30692, Sec. 3.3 / Appendix D), rendered by render.html in headless Chrome.
#   train      : 30,000 five-swap episodes (8 chunks, 157 frames)
#   test/nXX   : 64 episodes for every N = 5..30 (N+3 chunks, 20N+57 frames)
# 256x192, 16 fps. One action per chunk: 0-9 swap the cups in slots (i, j), i < j, in PAIRS order; 10 lift, 11 lower, 12 no-op.
# An episode with N swaps has actions [lift, lower, swap_1 .. swap_N, lift].
# Usage: python shellgame/generate.py            (re-running resumes: finished videos are skipped)
#        sbatch shellgame/render.slurm          (same, split over a Slurm job array)
# Needs only numpy and a Chrome/Chromium/Edge install (WebGL + WebCodecs VP9). On a Linux box without a display
# GPU, try --chrome-args="--no-sandbox --use-angle=swiftshader --enable-unsafe-swiftshader".
import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = Path(__file__).resolve().parent
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
LIFT, LOWER, NOOP = 10, 11, 12


def find_chrome():
    names = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome", "msedge"]
    paths = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
    for c in [os.environ.get("CHROME"), *map(shutil.which, names), *paths]:
        if c and Path(c).exists():
            return c
    raise SystemExit("Chrome not found: pass --chrome /path/to/chrome")


def sample(rng, num_swaps, count, taken):
    # unique episodes; `taken` is shared across splits so test N=5 never repeats a training episode
    out = []
    while len(out) < count:
        ep = (int(rng.integers(5)), *rng.integers(10, size=num_swaps).tolist())
        if ep not in taken:
            taken.add(ep)
            out.append(ep)
    return out


def episode(start, swaps):
    ball = [start, start]  # slot of the ball at the end of each chunk
    for a in swaps:
        i, j = PAIRS[a]
        ball.append(j if ball[-1] == i else i if ball[-1] == j else ball[-1])
    ball.append(ball[-1])
    return {"start": start, "swaps": [list(PAIRS[a]) for a in swaps], "actions": [LIFT, LOWER, *swaps, LIFT],
            "ball": ball, "final": ball[-1], "num_frames": 20 * len(swaps) + 57}


def build_splits(seed):
    rng = np.random.default_rng(seed)
    taken = set()
    splits = {"train": [episode(e[0], e[1:]) for e in sample(rng, 5, 30000, taken)]}
    for n in range(5, 31):
        splits[f"test/n{n:02d}"] = [episode(e[0], e[1:]) for e in sample(rng, n, 64, taken)]
    return splits


class Jobs:
    def __init__(self, tasks):
        self.pending = list(reversed(tasks))
        self.inflight = {}  # id -> (task, time handed out)
        self.total, self.done = len(tasks), 0
        self.lock = threading.Lock()
        self.finished = threading.Event()
        if not tasks:
            self.finished.set()

    def next(self):
        with self.lock:
            if self.done == self.total:
                return {"done": True}
            if self.pending:
                task = self.pending.pop()
            else:  # hand out again anything a dead tab never returned
                stale = [t for t, t0 in self.inflight.values() if time.time() - t0 > 120]
                if not stale:
                    return {"wait": True}
                task = stale[0]
            self.inflight[task["id"]] = (task, time.time())
            return {k: v for k, v in task.items() if k != "out"}  # auto_research: extra fields (starts) go to the page too

    def finish(self, task_id, mp4):
        with self.lock:
            if task_id not in self.inflight:
                return
            task, _ = self.inflight.pop(task_id)
        out = Path(task["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(mp4)
        os.replace(tmp, out)
        with self.lock:
            self.done += 1
            if self.done == self.total:
                self.finished.set()


def make_handler(jobs):
    class Handler(SimpleHTTPRequestHandler):
        extensions_map = {".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript"}

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(HERE), **kw)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if urlparse(self.path).path == "/task":
                return self.reply(jobs.next())
            super().do_GET()

        def do_POST(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if url.path == "/result":
                jobs.finish(int(q["id"][0]), body)
            elif url.path == "/log":
                print(f"[tab {q.get('w', ['?'])[0]}] {body.decode(errors='replace')}", flush=True)
            self.reply({})

        def reply(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def render(tasks, workers, chrome, chrome_args, page="render.html"):
    jobs = Jobs(tasks)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(jobs))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile_dir = Path(tempfile.mkdtemp(prefix="shellgame_chrome_"))  # per run: Chrome hands off to any instance sharing a profile
    browsers = [subprocess.Popen([
        chrome, "--headless=new", f"--user-data-dir={profile_dir / f'w{w}'}", "--no-first-run",
        "--no-default-browser-check", "--disable-extensions", "--mute-audio",
        "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows", *chrome_args, f"http://127.0.0.1:{port}/{page}?w={w}",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for w in range(workers)]
    t0 = time.time()
    try:
        while not jobs.finished.wait(30):
            if all(b.poll() is not None for b in browsers):
                raise SystemExit("all browsers exited before the episodes were done")
            done = jobs.done
            rate = done / (time.time() - t0)
            eta = (jobs.total - done) / rate / 60 if rate else float("nan")
            print(f"{done}/{jobs.total} episodes, {rate:.1f}/s, ETA {eta:.0f} min", flush=True)
    finally:
        for b in browsers:
            b.kill()
            b.wait()
        server.shutdown()
        shutil.rmtree(profile_dir, ignore_errors=True)
    print(f"rendered {jobs.total} episodes in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/sds-rise/ethan/Dataset/ShellGame")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="render only the first LIMIT episodes of each split")
    ap.add_argument("--shard", type=int, default=0, help="render only every NUM_SHARDS-th episode, from SHARD on")
    ap.add_argument("--num-shards", type=int, default=1, help="e.g. one shard per Slurm array task (render.slurm)")
    ap.add_argument("--episode", type=int, nargs="+", default=None,
                    help="debug: render one episode 'start a1 a2 ...' (action ids) to <out>/debug.mp4")
    ap.add_argument("--chrome", default=None, help="browser executable (default: auto-detect)")
    ap.add_argument("--chrome-args", default="", help="extra browser flags, e.g. '--no-sandbox'")
    args = ap.parse_args()
    root = Path(args.out)
    chrome, chrome_args = args.chrome or find_chrome(), shlex.split(args.chrome_args)

    if args.episode:
        ep = episode(args.episode[0], args.episode[1:])
        return render([{"id": 0, "start": ep["start"], "swaps": ep["swaps"], "out": str(root / "debug.mp4")}],
                      1, chrome, chrome_args)

    todo = []
    for name, eps in build_splits(args.seed).items():
        d = root / name
        for i, ep in enumerate(eps):
            ep["video"] = f"videos/{i // 1000:02d}/{i:06d}.mp4"
        if args.shard == 0:  # every shard builds the same splits from the seed; one of them writes the metadata
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "meta.jsonl", "w") as f:
                for i, ep in enumerate(eps):
                    f.write(json.dumps({"id": i, **ep}) + "\n")
            np.save(d / "actions.npy", np.array([ep["actions"] for ep in eps], dtype=np.int64))
            np.save(d / "ball.npy", np.array([ep["ball"] for ep in eps], dtype=np.int64))
        todo += [(d, ep) for ep in eps[:args.limit]]
    # strided over all splits, so every shard gets the same mix of short train and long test episodes
    tasks = [{"id": k, "start": ep["start"], "swaps": ep["swaps"], "out": str(d / ep["video"])}
             for k, (d, ep) in enumerate(todo[args.shard::args.num_shards]) if not (d / ep["video"]).exists()]

    info = {
        "source": "arXiv 2608.30692 Shell Game (App. D); renderer adapted from github.com/liutiedong/shellgame",
        "height": 192, "width": 256, "fps": 16, "num_cups": 5, "seed": args.seed, "codec": "VP9 in mp4, 2 Mbps",
        "actions": [f"swap {i}-{j}" for i, j in PAIRS] + ["lift", "lower", "no-op"],
        "chunks": "N+3 per episode: reveal(lift), lower, N swaps, reveal(lift); chunk 0 = 17 frames, later chunks = 20 frames (5 latent frames each)",
        "ball": "slot of the ball (0 = leftmost) at the end of each chunk",
        "latents": "encode_latents.py: latents/XX/NNNNNN.npy, float16 [16, 5(N+3), 24, 32], Wan2.1 VAE mean, "
                   "normalized (mu - latents_mean) / latents_std",
    }
    if args.shard == 0:
        (root / "info.json").write_text(json.dumps(info, indent=2))
    print(f"shard {args.shard}/{args.num_shards}: {len(tasks)} episodes to render", flush=True)
    render(tasks, args.workers, chrome, chrome_args)


if __name__ == "__main__":
    main()
