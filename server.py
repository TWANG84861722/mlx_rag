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

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

import chat
from config import MAX_HISTORY_TURNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

app = FastAPI()

# 单用户个人工具：服务端保存一份全局对话历史(和终端 REPL 行为一致)。
_history = []


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
  #rewrite { color: #888; font-size: 13px; margin: 12px 2px 0; }
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
  <h1>📚 docsense — 问你的论文库</h1>
  <textarea id="q" placeholder="打字，或点键盘上的🎤用语音输入（中文/英文都行）…"></textarea>
  <div id="hint">提示：手机键盘的语音输入对中文更准。答案用英文，问题什么语言都行。</div>
  <div class="row">
    <button id="ask" onclick="ask()">提问</button>
    <button class="ghost" onclick="reset()">清空对话</button>
  </div>
  <div id="rewrite"></div>
  <div id="answer"></div>
  <div id="sources"></div>

<script>
async function ask() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  const btn = document.getElementById('ask');
  btn.disabled = true;
  document.getElementById('rewrite').textContent = '';
  document.getElementById('sources').innerHTML = '';
  document.getElementById('answer').innerHTML = '<span class="spin">🤔 检索+作答中，可能要十几秒到一分钟…</span>';
  try {
    const r = await fetch('/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    });
    const d = await r.json();
    if (d.error) { document.getElementById('answer').textContent = '出错：' + d.error; return; }
    if (d.standalone && d.standalone !== q)
      document.getElementById('rewrite').textContent = '规整为英文检索 → ' + d.standalone;
    document.getElementById('answer').textContent = d.answer || '(没找到相关内容)';
    const box = document.getElementById('sources');
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
    document.getElementById('q').value = '';
  } catch (e) {
    document.getElementById('answer').textContent = '请求失败：' + e;
  } finally {
    btn.disabled = false;
  }
}
async function reset() {
  await fetch('/reset', {method: 'POST'});
  document.getElementById('rewrite').textContent = '';
  document.getElementById('answer').textContent = '';
  document.getElementById('sources').innerHTML = '';
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.post("/reset")
def reset():
    _history.clear()
    return {"ok": True}


@app.post("/ask")
def ask(req: Ask):
    question = req.question.strip()
    if not question:
        return JSONResponse({"error": "empty question"})
    try:
        standalone = chat.condense_question(question, _history)
        answer, sources = chat.map_reduce(standalone)
    except Exception as e:
        logging.exception("ask failed")
        return JSONResponse({"error": str(e)})

    _history.append({"role": "user", "content": question})
    _history.append({"role": "assistant", "content": answer})
    if len(_history) > MAX_HISTORY_TURNS * 2:
        del _history[:-(MAX_HISTORY_TURNS * 2)]

    return {
        "answer": answer,
        "standalone": standalone,
        "sources": [
            {
                "paper": s["paper"], "page": s["page"],
                "section": (s.get("section") or "").strip(),
                "type": s.get("type", "text"),
                "rerank": float(s.get("rerank_score", 0)),
                "snippet": s["text"][:120].replace("\n", " ").strip(),
            }
            for s in sources
        ],
    }


if __name__ == "__main__":
    ip = _lan_ip()
    print("\n" + "=" * 56)
    print("  docsense 手机问答已启动。手机(连同一 WiFi)打开：")
    print(f"      http://{ip}:8000")
    print("  停止：Ctrl-C")
    print("=" * 56 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
