"""PDF 解析：parse_pdf（主入口，在最上面）+ 它用到的零件（在下面）。

从上往下读（高层在前、细节在后，即 stepdown / 报纸式结构）：
  parse_pdf(一篇PDF)        ← 主入口：逐页拆段 + 抽图注识图，汇成 elements
     ↓ 用到
  page_segments(一页)       ← 把一页按版面从上到下切成 (text/table, 章节, 文字)
  parse_scanned_page        ← 扫描页(无文字层)：整页渲染 → VL 转 markdown
  extract_figure_captions   ← 找出 "Figure N" 图注
  describe_figure           ← 渲染图片区域 + 调 VL 识图（Tier 0：裁切图注上方）
  describe_figure_fullpage  ← 裁切失败时的兜底：整页渲染，让 VL 找出该图注对应的图
  table_to_markdown         ← 表格 → Markdown
  bbox_overlap              ← 判断两个矩形框重不重叠
"""
from collections import Counter
import logging
import re

import fitz

import config
import model_client
from .common import is_table_caption, table_label

logger = logging.getLogger(__name__)

fitz.TOOLS.mupdf_display_errors(False)   # 关掉 PyMuPDF 的报错刷屏（解析坏页时不吵）

# 章节标题正则：匹配 "Introduction" / "2. Methods" / "Results and Discussion" 等常见论文小节名
SECTION_RE = re.compile(
    r"^\d{0,2}\.?\s*"
    r"(abstract|introduction|background|related\s+work|"
    r"methods?|materials?\s*(and\s+)?methods?|experimental|"
    r"results?(\s+and\s+discussion)?|discussion|conclusions?|"
    r"acknowledgements?|references?|supplementary|appendix|"
    r"funding|ethics|data\s+availability)\b",
    re.IGNORECASE
)

# 图注正则：匹配 "Figure 1" / "Fig. 2A" / "Extended Data Fig 3" 等开头
FIGURE_RE = re.compile(
    r'^((?:Extended\s+Data\s+|Supplementary\s+)?Fig(?:ure)?\.?\s*\d+[A-Za-z]?)\b',
    re.IGNORECASE
)

# 参考文献小节标题：匹配 "References" / "1. References" 等
REFERENCES_RE = re.compile(r'^\d{0,2}\.?\s*references?\b', re.IGNORECASE)

# 扫描页判定：文字层字符数 < 这个数 → 认为这页没有可抽的文字（扫描/图片版），走整页 VL。
# 正常文字页动辄几百上千字符；纯图/扫描页通常为 0，用小阈值即可安全区分。
_SCANNED_PAGE_MAX_CHARS = 50




# ════════════════════════════════════════════════════════════
#  主入口
# ════════════════════════════════════════════════════════════

def parse_pdf(path):
    """解析单篇 PDF → elements。

    接收: PDF 文件路径
    输出: 元件列表，每项 {"page","section","type"∈{text,table,figure},"text"}
    """
    try:
        doc = fitz.open(path)                  # 打开 PDF（失败就记日志、返回空）
    except Exception as e:
        logger.error(f"Cannot open {path}: {e}")
        return []

    elements = []
    current_section = ""          # 跨页延续的"当前章节"
    in_references = False         # 是否已进入参考文献段（之后的内容都不要）

    for page_num, page in enumerate(doc):       # 逐页：page_num 从 0 开始（用时 +1 变页码）
        try:
            # 扫描页(无文字层)：get_text 抽不到字 → 整页交给 VL 转 markdown，跳过基于文字层的裁切流程。
            # 一次整页 VL 同时拿到正文+表+图描述，正是 NotebookLM 处理扫描版的路子。
            if len(page.get_text().strip()) < _SCANNED_PAGE_MAX_CHARS:
                md = parse_scanned_page(page)
                if md:
                    elements.append({
                        "page": page_num + 1,
                        "section": current_section,   # 扫描页不做章节跟踪，沿用上一页的
                        "type": "text",
                        "text": md,
                    })
                continue                              # 这页处理完了，不再走下面的裁切流程

            # 本页的图注、表注、有边框表（后面都要用；bordered 复用、不重复跑 find_tables）
            fig_caps = extract_figure_captions(page)
            table_caps = extract_table_captions(page)
            bordered_tables, bordered_bboxes = _bordered_tables(page)

            # 无边框表（find_tables 抓不到）：先用 VL 抽成干净 Markdown，并记下其区域。
            # 记区域是为了待会儿让 page_segments 把这块从正文排除 → 避免"表被摊进正文"和 VL 版重复。
            vl_tables = []          # 待加入 elements 的 VL 表
            table_excludes = []     # 无边框表区域（含表注），传给 page_segments 从正文排除
            for label, caption, cap_bbox in table_caps:
                cap_top, cap_bottom = cap_bbox[1], cap_bbox[3]
                # 表区域上/下边界 = 最近的 表/图 注（下方取其上边、上方取其下边），否则到页顶/页底
                belows = ([c[2][1] for c in table_caps if c[2][1] > cap_bottom]
                          + [f[2][1] for f in fig_caps if f[2][1] > cap_bottom])
                bottom_y = min(belows) if belows else page.rect.y1
                aboves = ([c[2][3] for c in table_caps if c[2][3] < cap_top]
                          + [f[2][3] for f in fig_caps if f[2][3] < cap_top])
                top_y = max(aboves) if aboves else page.rect.y0
                # find_tables 已抓到（有边框）→ page_segments 会处理，跳过（表注上/下任一侧命中即算）
                below_r = _table_region(page, cap_bottom, bottom_y)
                above_r = _table_region(page, top_y, cap_top)
                if any(bbox_overlap(tuple(below_r), bb) or bbox_overlap(tuple(above_r), bb)
                       for bb in bordered_bboxes):
                    continue
                md, region = extract_table_via_vl(page, cap_bbox, bottom_y, top_y)
                if md:
                    vl_tables.append({
                        "page": page_num + 1,
                        "section": label,
                        "type": "table",
                        "text": f"{caption}\n{md}",   # 原表注 + VL 抽出的 Markdown 表
                    })
                    table_excludes.append(tuple(region | fitz.Rect(cap_bbox)))  # 连表注一起排除

            # 正文拆段：把有边框表 + 无边框表(已被 VL 抽走)的区域都从正文里排除
            segs = page_segments(page, current_section,
                                 bordered=(bordered_tables, bordered_bboxes),
                                 exclude_bboxes=table_excludes)
            if segs:
                current_section = segs[-1][1]             # 用本页最后一段的章节，延续到下一页

            for seg_type, section, seg_text in segs:
                # 进入 References 段后就不再收正文/表格（参考文献不进索引）
                if is_references(section):
                    in_references = True
                elif section:
                    in_references = False
                if in_references:
                    continue

                elements.append({
                    "page": page_num + 1,
                    "section": section,
                    "type": "table" if seg_type == "table" else "text",
                    "text": seg_text,
                })

            # 图注 + 识图：每条 "Figure N" 图注 → 渲染对应图片 → VL 描述 → 一个 figure 元件。
            # 裁切失败(VL说没看到图)时逐级升级兜底：整页当前页(救侧注/区域取错) → 上一页(救跨页) → 认栽只留图注。
            # 阶梯自终结：救得了的在对应档停下，怎么都救不了的自动就是"坏图/无此图"。
            for fig_label, caption, cap_bbox in fig_caps:
                description = describe_figure(page, cap_bbox)                   # Tier 0：裁切图注上方那块
                if description and model_client.vl_found_nothing(description):
                    description = None
                if description is None:                                         # Tier 2a：整页当前页
                    logger.info(f"  {fig_label} p{page_num+1}: 裁切没抓到图 → 升级整页(Tier 2a)")
                    description = describe_figure_fullpage(page, caption)
                if description is None and page_num > 0:                        # Tier 2b：上一页(跨页)
                    logger.info(f"  {fig_label} p{page_num+1}: 整页仍没有 → 找上一页(Tier 2b)")
                    description = describe_figure_fullpage(doc[page_num - 1], caption)
                if description is None:                                         # 各档都失败
                    logger.info(f"  {fig_label} p{page_num+1}: 各档都没抓到图 → 只留图注(疑似坏图)")
                text = f"{description}\n{caption}" if description else caption  # 都失败 → 只留图注
                elements.append({
                    "page": page_num + 1,
                    "section": fig_label,        # figure 的 section 用图标签，如 "Figure 1"
                    "type": "figure",
                    "text": text,
                })

            # 无边框表：加入前面已 VL 抽好的干净 Markdown 表（正文里那份已被排除，不再重复）
            elements.extend(vl_tables)
        except Exception as e:
            logger.warning(f"  Page {page_num + 1} skipped: {e}")   # 单页出错只跳过这页，不中断整篇

    doc.close()
    return elements


# ════════════════════════════════════════════════════════════
#  parse_pdf 用到的零件
# ════════════════════════════════════════════════════════════

def page_segments(page, prev_section="", bordered=None, exclude_bboxes=()):
    """把一页拆成 (类型, 章节, 文字) 的小段，按版面从上到下。

    接收: 页对象 page、进入本页时所处的章节 prev_section、
          bordered（可选 (表对象list, bbox list)，parse_pdf 已算好就传进来、省得重跑 find_tables）、
          exclude_bboxes（可选，要从正文里额外排除的区域 —— 即无边框表已被 VL 抽走的那块）
    输出: 列表，每项是 (kind, section, text)，kind ∈ {"text","table"}
          （figure 不在这里，由 parse_pdf 另行处理）

    有边框的表格 → 转 Markdown 单独成段，并从正文里排除；
    无边框的表格 → parse_pdf 用 VL 抽取、并把其区域经 exclude_bboxes 传进来从正文排除
                   （否则表格文字会被摊进正文，和 VL 干净版重复）。
    """
    # ── 1. 有边框表格（find_tables）：parse_pdf 已算好就复用，否则自己算 ──
    if bordered is None:
        bordered_tables, bordered_bboxes = _bordered_tables(page)
    else:
        bordered_tables, bordered_bboxes = bordered
    # 正文要排除的所有区域 = 有边框表 + 无边框表(已被 VL 抽走)
    skip_bboxes = list(bordered_bboxes) + list(exclude_bboxes)

    blocks = page.get_text("dict")["blocks"]   # 本页所有"块"（文字块 type=0 / 图片块 type=1）

    # ── 2. 统计正文字号：出现最多的字号就是"正文大小"，比它大的行可能是标题 ──
    sizes = [
        round(s["size"], 1)
        for b in blocks if b.get("type") == 0                       # 只看文字块
        if not any(bbox_overlap(b["bbox"], tb) for tb in skip_bboxes)  # 且不在表格区域内
        for line in b.get("lines", [])
        for s in line.get("spans", [])
        if s["text"].strip()
    ]

    if not sizes:                       # 整页没正文文字（可能纯图）→ 把整页文本当一段返回
        raw = page.get_text()
        return [("text", prev_section, raw)] if raw.strip() else []

    body_size = Counter(sizes).most_common(1)[0][0]   # 出现次数最多的字号 = 正文字号

    # ── 3. 把"文字块"和"表格"放进同一个列表 items，并按垂直位置(y)从上到下排序 ──
    items = []
    for b in blocks:
        if b.get("type") != 0:                                          # 不是文字块，跳过
            continue
        if any(bbox_overlap(b["bbox"], tb) for tb in skip_bboxes):  # 落在表格区域内，跳过（表格单独处理）
            continue
        items.append(("text", b["bbox"][1], b))     # ("text", 这块的上边y, 块内容)
    for t in bordered_tables:
        items.append(("table", t.bbox[1], t))       # ("table", 表的上边y, 表对象)
    items.sort(key=lambda x: x[1])                  # 按 y（上边坐标）排序 → 还原"从上到下"的阅读顺序

    # ── 4. 从上到下走一遍，边走边归类：标题→新章节；正文→攒着；表格→单独成段 ──
    segs = []
    cur_section = prev_section
    cur_text = ""                 # 暂存"还没归档的正文"
    pending_caption = ""          # 上一个识别到的 "Table N..." 表注，等下一张表来绑定（表注在表上方）
    last_table = None             # 最近生成的表格段，用于"表注在表下方"时把表注回填上去

    for kind, _, content in items:

        if kind == "table":
            if cur_text.strip():                                   # 表格前，先把攒的正文归档
                segs.append(("text", cur_section, cur_text.strip()))
                cur_text = ""
            md = table_to_markdown(content)
            if md:
                # 表注通常在表上方：把刚存的表注拼到表格前面，同进一个 chunk
                text = f"{pending_caption}\n{md}" if pending_caption else md
                segs.append(("table", cur_section, text))
                last_table = {"idx": len(segs) - 1, "bbox": content.bbox,
                              "captioned": bool(pending_caption)}   # 记下这张表，"表注在下方"时回填
            pending_caption = ""

        else:   # 文字块
            lines = content.get("lines", [])
            if lines:
                first_spans = [s["text"] for s in lines[0].get("spans", []) if s["text"].strip()]
                first_line = " ".join(first_spans).strip()         # 这一块的第一行
                if FIGURE_RE.match(first_line):                    # 是图注 → 跳过（图注由 parse_pdf 那边处理）
                    continue
                # 这一块本身是表注（"Table N ..."）？
                if is_table_caption(first_line):
                    cap_text = " ".join(                            # 整块表注文字
                        " ".join(s["text"] for s in ln.get("spans", []))
                        for ln in lines
                    ).strip()
                    cap_bbox = content["bbox"]
                    # 表注在表【下方】：紧邻上方若有张还没带表注的表 → 把表注回填到那张表上（不丢进正文）
                    if (last_table and not last_table["captioned"]
                            and 0 <= cap_bbox[1] - last_table["bbox"][3] < 30):
                        i = last_table["idx"]
                        _, sec, tbl_text = segs[i]
                        segs[i] = ("table", sec, f"{cap_text}\n{tbl_text}")
                        last_table["captioned"] = True
                        continue
                    # 否则表注在表【上方】（常见）：存起来，等下一张表绑定
                    if cur_text.strip():
                        segs.append(("text", cur_section, cur_text.strip()))
                        cur_text = ""
                    pending_caption = cap_text
                    continue

            # 普通文字块：若之前存了表注却没等到表，把它放回正文（别丢）
            if pending_caption:
                cur_text += pending_caption + " "
                pending_caption = ""

            # 逐行处理：判断是不是"小节标题"，否则就累加进正文
            for line in lines:
                spans = [s for s in line.get("spans", []) if s["text"].strip()]
                if not spans:
                    continue
                line_text = " ".join(s["text"] for s in spans).strip()
                avg_size = sum(s["size"] for s in spans) / len(spans)        # 这行平均字号
                is_bold = any(bool(s["flags"] & 16) for s in spans)         # 这行有没有加粗（flags 的第16位=粗体）
                is_larger = avg_size > body_size * 1.1                      # 比正文大 10% 以上

                # 短 + (加粗或更大) + 命中章节名 → 判定为"小节标题"
                if (
                    len(line_text) < 80
                    and (is_bold or is_larger)
                    and SECTION_RE.match(line_text)
                ):
                    if cur_text.strip():                            # 开新节前，先把上一节正文归档
                        segs.append(("text", cur_section, cur_text.strip()))
                    cur_section = line_text                         # 切换到新章节
                    cur_text = ""
                else:
                    cur_text += line_text + " "                     # 普通正文 → 累加

    if pending_caption:           # 循环结束还存着表注（没等到表）→ 放回正文别丢
        cur_text += pending_caption + " "
    if cur_text.strip():          # 把最后攒的正文也归档
        segs.append(("text", cur_section, cur_text.strip()))

    return segs


def _bordered_tables(page):
    """本页"有边框的表格"：返回 (表对象list, bbox list)。

    find_tables(strategy="lines") 只认有竖线的表；过滤掉算不出 bbox 的坏表
    （空表会让 t.bbox 崩 → 整页被跳过、正文丢失）。page_segments 与 parse_pdf 共用，避免重复检测。
    """
    try:
        found = page.find_tables(strategy="lines").tables
    except Exception:
        found = []
    tables, bboxes = [], []
    for t in found:
        try:
            bbox = t.bbox              # 试算一次：能算出来才保留
        except Exception:
            continue
        tables.append(t)
        bboxes.append(bbox)
    return tables, bboxes


def extract_figure_captions(page):
    # 接收: 一页(page)对象
    # 输出: 列表，每项是 (图标签, 完整图注文字, 图注块的边界框)
    #       例: ("Figure 1", "Figure 1. Editing efficiency...", (x0,y0,x1,y1))
    captions = []
    for block in page.get_text("dict")["blocks"]:   # 遍历整页的每个"块"
        if block.get("type") != 0:                  # type!=0 不是文字块(可能是图片)，跳过
            continue
        lines = block.get("lines", [])
        if not lines:
            continue
        # 取这一块的"第一行"文字，看是不是 "Figure N..." 开头
        first_line = " ".join(
            s["text"] for s in lines[0].get("spans", [])
        ).strip()
        m = FIGURE_RE.match(first_line)
        if not m:                                   # 不是图注 → 跳过
            continue
        # 是图注：把这一块所有行拼成完整图注文字
        full_text = " ".join(
            " ".join(s["text"] for s in line.get("spans", []))
            for line in lines
        ).strip()
        captions.append((m.group(1), full_text, block["bbox"]))   # m.group(1) = "Figure 1" 这个标签
    return captions


def describe_figure(page, caption_bbox):
    """Render the figure image above the caption and return a VL description.

    接收: 页对象 page、图注的边界框 caption_bbox
    输出: 这张图的文字描述（字符串），失败返回 None

    识图预算由 describe_image 统一给足（封顶 4000）：按实际生成计费、模型写完自停，
    所以一处给足即可；"逐 panel 描述"靠 prompt 驱动，不靠 max_tokens。
    """
    cap_top = caption_bbox[1]   # 图注框的"上边"y 坐标（图在它上方，所以要找底部在这之上的东西）

    # Find image blocks sitting above the caption
    image_blocks = [
        b for b in page.get_text("dict")["blocks"]  # get_text("dict")["blocks"] 把整页拆成一个个"块",每块有 type:0=文字块,1=图片块。
        if b.get("type") == 1 and b["bbox"][3] <= cap_top + 10  # 挑出:①图片块(type==1) 且 ②下边(bbox[3])≤ 图注上边(cap_top)——坐落在图注上方的图片。+10 是容差。
    ]

    if image_blocks:
        best = max(image_blocks, key=lambda b: b["bbox"][3])
        clip = fitz.Rect(best["bbox"])  # 选下边最大的那张——最贴近图注、正上方那张（最可能就是这条图注的图）。clip = 它的边界框。
    else:
        # Fall back to the page region above the caption
        page_rect = page.rect
        clip = fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, cap_top)
        # 兜底:矢量图不算"图片块",找不到。就把"页面顶部→图注上边"整条圈出来，反正图在图注上方。

    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    # get_pixmap = 把 PDF 某块区域渲染成像素图。clip=只渲这块；matrix=2倍放大，更清晰，VL 看得更准。

    prompt = (
        "You are reading a figure from a molecular-biology research paper. "
        "Describe it thoroughly so it can be retrieved later. "
        "If the figure has multiple panels (A, B, C, ...), describe EACH panel "
        "separately: its purpose, what is plotted (axes, conditions/groups), and the "
        "key quantitative result or trend. Identify the figure type (e.g., western blot, "
        "bar graph, microscopy, survival curve). Include numbers, units, and statistical "
        "significance where visible. Do not omit any panel."
    )
    try:
        # pix.tobytes("png") → 图片的 PNG 字节；交给可切换的 VL（当前 qwen-vl-max）识图
        return model_client.describe_image(pix.tobytes("png"), prompt)
    except Exception as e:
        logger.warning(f"VL description failed: {e}")
        return None


def parse_scanned_page(page):
    """扫描页(无文字层)：整页渲染 → VL 转结构化 Markdown（正文 + 表格 + 图描述）。

    接收: 页对象   输出: markdown 字符串（失败返回 None）
    扫描页 get_text 抽不到字，只能"看图"。分辨率调高(2.5x)利于 VL 识字。
    prompt 明确"只有真实可见图片才写 [FIGURE]，别凭正文臆造"——防止在纯文字页幻觉出假图块。
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
    prompt = (
        "You are transcribing a scanned page from a scientific journal article into clean Markdown "
        "so it can be indexed for retrieval:\n"
        "- Transcribe all body text in reading order (handle multi-column layout correctly).\n"
        "- Render any table as a Markdown table with all rows and columns.\n"
        "- For a figure/photo/gel/diagram that is ACTUALLY VISIBLE on this page, add a line starting "
        "with '[FIGURE]' followed by its caption (if any) and a thorough description of what it shows "
        "(figure type, panels, axes/labels, key findings).\n"
        "- Do NOT invent a [FIGURE] block from the text alone — only when a figure is truly present.\n"
        "- Preserve section headings with ##. Do not add commentary that isn't on the page."
    )
    try:
        # 扫描页是"读字"活 → 用 OCR 专用模型（qwen-vl-ocr，准且省；没配则回退 vl_model）
        return model_client.describe_image(pix.tobytes("png"), prompt, model=config.ocr_model())
    except Exception as e:
        logger.warning(f"VL scanned-page parse failed: {e}")
        return None


def describe_figure_fullpage(page, caption):
    """兜底：整页渲染，让 VL 在整页里找出"图注是 caption 的那张图"并描述。

    接收: 页对象、该图注全文 caption   输出: 图描述（找不到/失败返回 None）
    用于 describe_figure 裁切失败后的升级——给 VL 整页视野 + 图注文字，
    它能自己定位对应的图（救侧注/区域取错），换上一页对象调用则救跨页。
    prompt 里给了明确的"没有就回 NO FIGURE HERE"，配合 vl_found_nothing 触发下一档兜底。
    """
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    prompt = (
        "This is a full page from a research paper. On this page there should be a figure whose caption is:\n"
        f"\"{caption}\"\n"
        "Find THAT specific figure on the page and describe it thoroughly for retrieval: figure type, "
        "each panel (A, B, C, ...), axes/conditions/groups, and the key quantitative results or trends. "
        "If that figure is NOT visually present anywhere on this page, reply with exactly: NO FIGURE HERE."
    )
    try:
        desc = model_client.describe_image(pix.tobytes("png"), prompt)
    except Exception as e:
        logger.warning(f"VL full-page figure failed: {e}")
        return None
    if desc and model_client.vl_found_nothing(desc):
        return None      # VL 说这页没有这张图 → 交给上层升级到下一档（上一页）
    return desc


def extract_table_captions(page):
    # 接收: 页对象
    # 输出: 列表，每项 (表标签, 完整表注, 表注块bbox)，如 ("Table 2", "Table 2. ...", (x0,y0,x1,y1))
    caps = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines = block.get("lines", [])
        if not lines:
            continue
        first_line = " ".join(s["text"] for s in lines[0].get("spans", [])).strip()
        label = table_label(first_line)        # 第一行是不是 "Table N..."
        if not label:
            continue
        full = " ".join(
            " ".join(s["text"] for s in ln.get("spans", []))
            for ln in lines
        ).strip()
        caps.append((label, full, block["bbox"]))
    return caps


def _table_region(page, top_y, bottom_y):
    """[top_y, bottom_y] 这段纵向区间里那张表的区域(fitz.Rect)。

    左右边界按"落在该区间内的文字块"算（表注常比表窄，用表注宽会裁掉右侧列）。
    extract_table_via_vl 用它当渲染 clip；parse_pdf 用它把这块从正文里排除，两处共用同一算法。
    """
    xs0, xs1 = [], []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = b["bbox"]
        if top_y <= by0 < bottom_y:
            xs0.append(bx0)
            xs1.append(bx1)
    left = min(xs0) if xs0 else page.rect.x0     # 区间内没文字块 → 兜底用整页宽
    right = max(xs1) if xs1 else page.rect.x1
    return fitz.Rect(left, top_y, right, bottom_y)


def _vl_read_table(page, clip):
    """渲染一块区域 → OCR 读成 Markdown 表。区域里没有表就返回 None。"""
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)
    prompt = (
        "This image is a region from a scientific paper. If it contains a table, extract it as a "
        "clean GitHub-flavored Markdown table — preserve all rows, columns, headers, numbers and units "
        "exactly, and output ONLY the markdown table (no caption or commentary). "
        "If the region does NOT contain a table (e.g. it is body text or a figure), reply with exactly: NO TABLE."
    )
    try:
        # 表格是"读字"活 → 走 OCR 专用模型（转录三线表更准、更省）
        md = model_client.describe_image(pix.tobytes("png"), prompt, model=config.ocr_model())
    except Exception as e:
        logger.warning(f"VL table extraction failed: {e}")
        return None
    if md and model_client.vl_found_nothing(md):
        return None      # VL 说这块没表（假 Table 标题 / 取错方向 / 是正文）→ 跳过
    return md


def extract_table_via_vl(page, caption_bbox, bottom_y, top_y):
    """三线表/无边框表：表注可能在表【下方】(常见)或【上方】，两个方向都试。

    接收: 页、表注bbox、下边界 bottom_y(下一个表/图注或页底)、上边界 top_y(上一个表/图注或页顶)
    输出: (markdown, 用到的区域Rect)；都读不到表则 (None, None)
    先试"表注下方"(绝大多数)；OCR 说没表(NO TABLE)再试"表注上方"(表注在表下的情况)。
    用 VL 是因为三线表没竖线，find_tables 抓不到；让 VL"看图读表"绕开几何检测。
    """
    cap_top, cap_bottom = caption_bbox[1], caption_bbox[3]
    for top, bot in [(cap_bottom, bottom_y), (top_y, cap_top)]:   # 先下、后上
        if bot - top < 10:                       # 这侧区间太薄 → 没东西，跳过
            continue
        region = _table_region(page, top, bot)
        md = _vl_read_table(page, region)
        if md:
            return md, region
    return None, None


def table_to_markdown(table):
    # 接收: 一个 PyMuPDF 检测到的表格对象   输出: Markdown 表格字符串
    rows = table.extract()                     # .extract() → 二维列表（每行的每个格子）
    if not rows:
        return ""
    lines = []
    header = [str(c or "").strip() for c in rows[0]]      # 第一行当表头（None→""）
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")   # Markdown 必需的分隔线
    for row in rows[1:]:                       # 其余数据行
        cells = [str(c or "").strip() for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bbox_overlap(a, b):
    # 接收: 两个矩形框 a、b，每个是 (x0左, y0上, x1右, y1下)
    # 输出: True/False —— 两个框是否重叠
    # 思路：先列出"绝对不重叠"的 4 种情况，取反就是"重叠"
    #   a[2] <= b[0]: a 的右边 ≤ b 的左边 → a 完全在 b 左侧
    #   a[0] >= b[2]: a 在 b 右侧；  a[3] <= b[1]: a 在 b 上方；  a[1] >= b[3]: a 在 b 下方
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def is_references(section):
    """接收: 章节名(字符串)  输出: True/False —— 它是不是"参考文献"小节标题。"""
    return bool(REFERENCES_RE.match(section or ""))