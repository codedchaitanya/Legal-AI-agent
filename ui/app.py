# ui/app.py
import uuid
import json
import httpx
import gradio as gr

API_URL = "http://api:8000"

EXAMPLE_RESEARCH = [
    ["What BNS sections apply for murder with common intention?"],
    ["Explain bail conditions under BNSS for kidnapping charges"],
    ["What are the key elements to prove cheating under BNS 318?"],
    ["Summarise landmark Supreme Court judgments on anticipatory bail"],
    ["What constitutes dowry death under BNS and its punishment?"],
    ["How does the court assess compensation in land acquisition disputes?"],
]

EXAMPLE_ARGUMENTS = [
    ["Accused was found with stolen goods — generate defense arguments"],
    ["FIR filed under POCSO — generate prosecution arguments"],
    ["Bail application for cyber fraud case — generate both sides"],
    ["Murder case with circumstantial evidence only — generate defense"],
]


def _headers(api_key: str) -> dict:
    return {"x-api-key": api_key.strip() or "set-your-api-key"}


def new_session() -> str:
    return str(uuid.uuid4())


# ── Research tab ─────────────────────────────────────────────────────────────

def research_chat(message: str, history: list, case_id: str, session_id: str, api_key: str):
    if not message.strip():
        return history, session_id, ""
    if not case_id.strip():
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please enter a Case ID in the sidebar first."},
        ]
        return history, session_id, ""

    sid = session_id or new_session()
    try:
        r = httpx.post(
            f"{API_URL}/cases/{case_id}/query",
            json={"session_id": sid, "query": message, "query_type": "research"},
            headers=_headers(api_key),
            timeout=90,
        )
        r.raise_for_status()
        data = r.json()
        answer = data.get("answer", "No answer returned.")
        adapters = ", ".join(data.get("adapters_used", []))
        latency = data.get("latency_ms", 0)
        answer_with_meta = f"{answer}\n\n---\n*Adapters: {adapters} | {latency:.0f} ms*"
    except Exception as e:
        answer_with_meta = f"❌ Error: {e}"

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer_with_meta},
    ]
    return history, sid, ""


# ── Arguments tab ────────────────────────────────────────────────────────────

def generate_args(query: str, side: str, images: list, case_id: str, session_id: str, api_key: str):
    if not query.strip():
        return "Please describe the case scenario.", session_id
    if not case_id.strip():
        return "⚠️ Please enter a Case ID in the sidebar first.", session_id

    sid = session_id or new_session()
    image_paths = [f.name for f in (images or [])]
    try:
        r = httpx.post(
            f"{API_URL}/cases/{case_id}/query",
            json={
                "session_id": sid,
                "query": query,
                "query_type": "arguments",
                "side": side.lower(),
                "image_paths": image_paths,
            },
            headers=_headers(api_key),
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        args = data.get("arguments") or {}

        if args.get("parse_error"):
            return args.get("raw_arguments", "No arguments generated."), sid

        lines = []
        if "prosecution_arguments" in args:
            lines.append("### ⚖️ Prosecution Arguments")
            for a in args["prosecution_arguments"]:
                strength_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(a.get("strength", ""), "•")
                lines.append(f"**{strength_icon} {a['point']}**\n> {a['legal_basis']}")
        if "defense_arguments" in args:
            lines.append("\n### 🛡️ Defense Arguments")
            for a in args["defense_arguments"]:
                strength_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(a.get("strength", ""), "•")
                lines.append(f"**{strength_icon} {a['point']}**\n> {a['legal_basis']}")
        if "key_precedents" in args:
            lines.append("\n### 📚 Key Precedents")
            for p in args["key_precedents"]:
                lines.append(f"- {p}")
        if "strategic_recommendations" in args:
            lines.append("\n### 💡 Strategic Recommendations")
            for s in args["strategic_recommendations"]:
                lines.append(f"- {s}")
        if "risk_assessment" in args:
            lines.append(f"\n### 🎯 Risk Assessment\n{args['risk_assessment']}")

        return "\n\n".join(lines), sid

    except Exception as e:
        return f"❌ Error: {e}", sid


# ── Document upload tab ──────────────────────────────────────────────────────

def upload_document(files: list, case_id: str, api_key: str):
    if not files:
        return "No files selected."
    if not case_id.strip():
        return "⚠️ Please enter a Case ID in the sidebar first."

    results = []
    for f in files:
        try:
            with open(f.name, "rb") as fp:
                r = httpx.post(
                    f"{API_URL}/cases/{case_id}/documents",
                    files={"file": (f.name.split("/")[-1], fp)},
                    headers=_headers(api_key),
                    timeout=120,
                )
                r.raise_for_status()
                results.append(f"✅ {f.name.split('/')[-1]} — uploaded successfully")
        except Exception as e:
            results.append(f"❌ {f.name.split('/')[-1]} — {e}")

    return "\n".join(results)


# ── Create case ──────────────────────────────────────────────────────────────

def create_case(title: str, api_key: str):
    if not title.strip():
        return "Please enter a case title.", ""
    try:
        r = httpx.post(
            f"{API_URL}/cases",
            json={"title": title},
            headers=_headers(api_key),
            timeout=30,
        )
        r.raise_for_status()
        case_id = r.json().get("case_id", "")
        return f"✅ Case created! ID: **{case_id}**", case_id
    except Exception as e:
        return f"❌ Error: {e}", ""


# ── UI ───────────────────────────────────────────────────────────────────────

THEME = gr.themes.Monochrome(
    primary_hue="orange",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

CSS = """
.sidebar { border-right: 1px solid var(--border-color-primary); padding: 16px; }
.badge { font-size: 11px; color: var(--body-text-color-subdued); margin-top: 4px; }
.section-title { font-weight: 700; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
footer { display: none !important; }
blockquote { border-left: 3px solid var(--color-accent); padding-left: 8px; }
"""

with gr.Blocks(title="Indian Legal AI") as demo:

    session_state = gr.State("")

    with gr.Row():
        # ── Sidebar ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260, elem_classes="sidebar"):
            gr.Markdown("## ⚖️ Legal AI\n*Indian Case Research*")
            gr.Markdown("---")

            gr.Markdown("**Configuration**", elem_classes="section-title")
            api_key_input = gr.Textbox(
                label="API Key", type="password",
                placeholder="your-api-key", container=True,
            )
            case_id_input = gr.Textbox(
                label="Case ID", placeholder="CASE-2024-001",
            )

            gr.Markdown("---")
            gr.Markdown("**Create New Case**", elem_classes="section-title")
            case_title_input = gr.Textbox(label="Case Title", placeholder="State vs. Sharma — IPC 302")
            create_btn = gr.Button("Create Case", size="sm", variant="secondary")
            create_status = gr.Markdown("")

            gr.Markdown("---")
            new_session_btn = gr.Button("🔄 New Session", size="sm", variant="secondary")
            session_display = gr.Markdown("*No session started*", elem_classes="badge")

        # ── Main content ──────────────────────────────────────────────────
        with gr.Column(scale=4):
            gr.Markdown("# Indian Legal AI Assistant")
            gr.Markdown(
                "Domain-aware legal research powered by QLoRA-trained adapters on Indian law "
                "(BNS/IPC/BNSS) + Claude Sonnet for argument generation."
            )

            with gr.Tabs():
                # ── Tab 1: Research ───────────────────────────────────────
                with gr.Tab("🔍 Legal Research"):
                    research_chatbot = gr.Chatbot(
                        height=480,
                        placeholder="Ask anything about your case or Indian law...",
                        show_label=False,
                    )
                    with gr.Row():
                        research_input = gr.Textbox(
                            placeholder="What BNS sections apply for this FIR?",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        research_btn = gr.Button("Send", variant="primary", scale=1)

                    gr.Examples(
                        examples=EXAMPLE_RESEARCH,
                        inputs=research_input,
                        label="Example queries — click to fill",
                    )

                # ── Tab 2: Arguments ──────────────────────────────────────
                with gr.Tab("⚖️ Generate Arguments"):
                    gr.Markdown(
                        "Describe the case scenario. Claude Sonnet will generate structured "
                        "legal arguments with citations, precedents, and strategy."
                    )
                    with gr.Row():
                        arg_side = gr.Radio(
                            ["Both", "Prosecution", "Defense"],
                            value="Both",
                            label="Generate arguments for",
                        )
                    arg_input = gr.Textbox(
                        label="Case scenario / query",
                        placeholder="Accused found with stolen vehicle. FIR under BNS 303. Generate arguments.",
                        lines=4,
                    )
                    arg_images = gr.File(
                        label="Court exhibits / images (optional)",
                        file_types=["image", ".pdf"],
                        file_count="multiple",
                    )
                    arg_btn = gr.Button("Generate Arguments", variant="primary")
                    arg_output = gr.Markdown(label="Generated Arguments")

                    gr.Examples(
                        examples=EXAMPLE_ARGUMENTS,
                        inputs=arg_input,
                        label="Example scenarios — click to fill",
                    )

                # ── Tab 3: Documents ──────────────────────────────────────
                with gr.Tab("📄 Upload Documents"):
                    gr.Markdown(
                        "Upload case documents (FIR, chargesheet, affidavits, judgments). "
                        "The system will OCR, summarise, and index them for research."
                    )
                    doc_upload = gr.File(
                        label="Select files",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".txt", ".docx"],
                        file_count="multiple",
                    )
                    doc_upload_btn = gr.Button("Upload & Process", variant="primary")
                    doc_status = gr.Markdown("")

                # ── Tab 4: About ──────────────────────────────────────────
                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
## Architecture

```
Document (PDF/Image/Text)
        ↓
   OCR (GLM-OCR) + PageIndex
        ↓
   Summarizer (Legal-BERT + PEGASUS + Claude fallback)
        ↓
   Domain Router (Claude Haiku → 12 legal domains)
        ↓
   QLoRA Adapter (Qwen2.5-7B fine-tuned per domain)
        ↓
   Citation Validator → Research Answer
        ↓ (if arguments enabled)
   Claude Sonnet → Prosecution/Defense Arguments
```

## 12 Legal Domains
`criminal_violent` · `criminal_property` · `kidnapping_trafficking` · `sexual_offences`
`land_property` · `family_matrimonial` · `constitutional` · `corporate_commercial`
`labour_employment` · `cyber_digital` · `tax_fiscal` · `civil_general`

## Models Used
| Task | Model |
|------|-------|
| OCR | GLM-OCR |
| Summarization | Legal-BERT + PEGASUS |
| Domain routing | Claude Haiku |
| Legal QA | Qwen2.5-7B + QLoRA adapter |
| Argument generation | Claude Sonnet (CoT + Vision) |
                    """)

    # ── Event wiring ──────────────────────────────────────────────────────────

    def update_session_display(sid):
        return f"Session: `{sid[:8]}…`" if sid else "*No session started*"

    new_session_btn.click(new_session, outputs=session_state).then(
        update_session_display, inputs=session_state, outputs=session_display
    )

    create_btn.click(
        create_case,
        inputs=[case_title_input, api_key_input],
        outputs=[create_status, case_id_input],
    )

    research_btn.click(
        research_chat,
        inputs=[research_input, research_chatbot, case_id_input, session_state, api_key_input],
        outputs=[research_chatbot, session_state, research_input],
    ).then(update_session_display, inputs=session_state, outputs=session_display)

    research_input.submit(
        research_chat,
        inputs=[research_input, research_chatbot, case_id_input, session_state, api_key_input],
        outputs=[research_chatbot, session_state, research_input],
    ).then(update_session_display, inputs=session_state, outputs=session_display)

    arg_btn.click(
        generate_args,
        inputs=[arg_input, arg_side, arg_images, case_id_input, session_state, api_key_input],
        outputs=[arg_output, session_state],
    )

    doc_upload_btn.click(
        upload_document,
        inputs=[doc_upload, case_id_input, api_key_input],
        outputs=doc_status,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=THEME, css=CSS)
