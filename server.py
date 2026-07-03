"""手机问答网页服务（局域网版）。

在 Mac 上跑：  python server.py
然后手机(连同一 WiFi)浏览器打开启动时打印的那个 http://<Mac的IP>:8000

- "大脑"(嵌入/重排/索引/论文)都在 Mac，手机只是个瘦客户端(输入框+答案区)。
- 复用 chat.py 里已有的 condense_question + map_reduce，不重复造检索逻辑。
- 手机上"说话提问"直接用输入法自带的语音输入(中文识别比桌面 Whisper 好)，无需额外代码。
- 只监听局域网、无鉴权 —— 仅限家里可信 WiFi 使用；别把 8000 端口暴露到公网。

依赖(可选)：  pip install fastapi uvicorn
"""
import socket
import logging
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

import chat
from config import MAX_HISTORY_TURNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

app = FastAPI()

# 单用户个人工具：服务端保存一份全局对话历史(和终端 REPL 行为一致)。
_history = []

# 后台任务表：一次问答很慢(可能一分钟+)，若让手机一直挂着等，iOS fetch ~60s 会超时断掉。
# 所以改成：提问 → 立刻返回 job_id → 手机每隔几秒轮询 /result。每次轮询都是瞬时返回，不会超时。
_jobs = {}                 # job_id -> {status, answer, sources, standalone, error, started}
_job_lock = threading.Lock()   # 模型/检索不保证线程安全 → 同一时刻只跑一个任务


class Ask(BaseModel):
    question: str


def _lan_ip():
    """取本机局域网 IP(用于打印手机该访问的地址)。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))      # 不真的发包，只为让系统选出出口网卡的 IP
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>docsense</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px;
         max-width: 760px; margin: 0 auto; line-height: 1.6; }
  h1 { font-size: 20px; margin: 4px 0 12px; }
  #q { width: 100%; font-size: 17px; padding: 12px; border: 1px solid #bbb;
       border-radius: 10px; resize: vertical; min-height: 64px; }
  .row { display: flex; gap: 8px; margin-top: 10px; }
  button { font-size: 16px; padding: 10px 16px; border: 0; border-radius: 10px;
           background: #2563eb; color: #fff; }
  button.ghost { background: #6b7280; }
  button:disabled { opacity: .5; }
  #hint { color: #888; font-size: 13px; margin: 6px 2px; }
  #asked { font-weight: 600; margin: 16px 2px 2px; }
  #rewrite { color: #888; font-size: 13px; margin: 4px 2px 0; }
  #answer { white-space: pre-wrap; margin-top: 14px; padding: 14px;
            border: 1px solid #ddd; border-radius: 10px; min-height: 40px; }
  #sources { margin-top: 14px; }
  .src { font-size: 13px; color: #555; border-top: 1px solid #eee; padding: 8px 2px; }
  .tag { display: inline-block; background: #eef; color: #446; border-radius: 6px;
         padding: 0 6px; font-size: 12px; margin-left: 4px; }
  .spin { color: #2563eb; }
</style>
</head>
<body>
  <h1>📚 docsense — ask your library</h1>
  <textarea id="q" placeholder="Type, or tap 🎤 on the keyboard to speak (any language)…"></textarea>
  <div id="hint">Tip: use the keyboard's dictation for voice. Ask in any language — answers are in English.</div>
  <div class="row">
    <button id="ask" onclick="ask()">Ask</button>
    <button class="ghost" onclick="reset()">New topic</button>
  </div>
  <div id="asked"></div>
  <div id="rewrite"></div>
  <div id="answer"></div>
  <div id="sources"></div>

<script>
const sleep = ms => new Promise(r => setTimeout(r, ms));

function render(d, q) {
  if (d.standalone && d.standalone !== q)
    document.getElementById('rewrite').textContent = 'Search query → ' + d.standalone;
  document.getElementById('answer').textContent = d.answer || '(nothing relevant found)';
  const box = document.getElementById('sources');
  box.innerHTML = '';
  (d.sources || []).forEach((s, i) => {
    const div = document.createElement('div');
    div.className = 'src';
    let tag = s.type === 'figure' ? '<span class="tag">figure</span>'
            : s.type === 'table'  ? '<span class="tag">table</span>' : '';
    div.innerHTML = '[' + (i+1) + '] ' + s.paper + ' p.' + s.page +
                    (s.section ? ' · ' + s.section : '') + tag +
                    ' <span style="color:#999">(rerank=' + s.rerank.toFixed(3) + ')</span><br>' +
                    '<span style="color:#888">' + s.snippet + '…</span>';
    box.appendChild(div);
  });
}

async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('ask');
  const ans = document.getElementById('answer');
  btn.disabled = true;
  document.getElementById('asked').textContent = 'Q: ' + q;   // 问题留在答案上方，问完不消失
  document.getElementById('rewrite').textContent = '';
  document.getElementById('sources').innerHTML = '';
  ans.innerHTML = '<span class="spin">🤔 Searching & answering…</span>';
  try {
    // 1) 提交任务，立刻拿到 job_id（这一步很快，不会超时）
    const r = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const j = await r.json();
    if (j.error || !j.job_id) { ans.textContent = 'Error: ' + (j.error || 'no job id'); return; }
    // 2) 每隔 N 毫秒轮询一次，直到 done/error（每次轮询都是瞬时返回，永不超时）
    while (true) {
      await sleep(3000);       // 轮询间隔（毫秒）：3000 = 每 3 秒问一次
      const rr = await fetch('/result/' + j.job_id);
      const d = await rr.json();
      if (d.status === 'running') {
        ans.innerHTML = '<span class="spin">🤔 Searching & answering… ' + d.elapsed + 's</span>';
        continue;
      }
      if (d.status === 'error') { ans.textContent = 'Error: ' + d.error; return; }
      render(d, q);                       // done
      document.getElementById('q').value = '';
      return;
    }
  } catch (e) {
    ans.textContent = 'Request failed: ' + e;
  } finally {
    btn.disabled = false;
  }
}
async function reset() {
  await fetch('/reset', {method: 'POST'});
  document.getElementById('asked').textContent = '';
  document.getElementById('rewrite').textContent = '';
  document.getElementById('answer').textContent = '';
  document.getElementById('sources').innerHTML = '';
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    # no-store：让手机浏览器每次都拿最新页面，免得改了代码手机还跑旧缓存
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)      # 浏览器要小图标；给个空响应，省得日志刷 404


@app.post("/reset")
def reset():
    _history.clear()
    return {"ok": True}


def _run_job(job_id, question):
    """后台线程：真正的耗时活(condense + map_reduce)，结果写回 _jobs[job_id]。"""
    with _job_lock:                       # 一次只跑一个：模型/检索非线程安全
        try:
            standalone = chat.condense_question(question, _history)
            logging.info("收到问题: %r  →  规整为英文: %r", question, standalone)
            answer, sources = chat.map_reduce(standalone)

            _history.append({"role": "user", "content": question})
            _history.append({"role": "assistant", "content": answer})
            if len(_history) > MAX_HISTORY_TURNS * 2:
                del _history[:-(MAX_HISTORY_TURNS * 2)]

            _jobs[job_id].update(
                status="done",
                standalone=standalone,
                answer=answer,
                sources=[
                    {
                        "paper": s["paper"], "page": s["page"],
                        "section": (s.get("section") or "").strip(),
                        "type": s.get("type", "text"),
                        "rerank": float(s.get("rerank_score", 0)),
                        "snippet": s["text"][:120].replace("\n", " ").strip(),
                    }
                    for s in sources
                ],
            )
        except Exception as e:
            logging.exception("job failed")
            _jobs[job_id].update(status="error", error=str(e))


@app.post("/ask")
def ask(req: Ask):
    """立刻返回 job_id；活儿丢到后台线程，手机去轮询 /result/<job_id>。"""
    question = req.question.strip()
    if not question:
        return JSONResponse({"error": "empty question"})
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "started": time.time()}
    threading.Thread(target=_run_job, args=(job_id, question), daemon=True).start()
    return {"job_id": job_id}


@app.get("/result/{job_id}")
def result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        return JSONResponse({"status": "error", "error": "unknown job"})
    out = {"status": job["status"], "elapsed": round(time.time() - job["started"])}
    if job["status"] == "done":
        out.update(answer=job["answer"], standalone=job["standalone"], sources=job["sources"])
        _jobs.pop(job_id, None)           # 取走即清，别攒内存
    elif job["status"] == "error":
        out["error"] = job.get("error", "unknown")
        _jobs.pop(job_id, None)
    return out


if __name__ == "__main__":
    ip = _lan_ip()
    print("\n" + "=" * 56)
    print("  docsense 手机问答已启动。手机(连同一 WiFi)打开：")
    print(f"      http://{ip}:8000")
    print("  停止：Ctrl-C")
    print("=" * 56 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
