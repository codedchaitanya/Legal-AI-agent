# ui/app.py
import gradio as gr
import httpx
import uuid

API_URL = "http://api:8000"
API_KEY_PLACEHOLDER = "set-your-api-key"


def chat(message: str, history: list, case_id: str, session_id: str, api_key: str):
    if not case_id.strip():
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Please enter a Case ID first."})
        return history, session_id

    if not session_id:
        session_id = str(uuid.uuid4())

    key = api_key.strip() or API_KEY_PLACEHOLDER

    try:
        response = httpx.post(
            f"{API_URL}/cases/{case_id}/query",
            json={"session_id": session_id, "query": message, "query_type": "research"},
            headers={"x-api-key": key},
            timeout=60,
        )
        answer = response.json().get("answer", "No answer returned.")
    except Exception as e:
        answer = f"Error: {e}"

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history, session_id


with gr.Blocks(title="Indian Legal AI Assistant") as demo:
    gr.Markdown("# Indian Legal AI Assistant")
    gr.Markdown("Enter your Case ID and ask questions about your case or Indian law.")

    with gr.Row():
        case_id_box = gr.Textbox(label="Case ID", placeholder="CASE-2024-001")
        api_key_box = gr.Textbox(label="API Key", placeholder="your-api-key", type="password")

    chatbot = gr.Chatbot(height=500, type="messages")
    session_state = gr.State("")
    msg_box = gr.Textbox(label="Your query", placeholder="What sections apply for cheating in this FIR?")
    send_btn = gr.Button("Send", variant="primary")

    send_btn.click(
        chat,
        inputs=[msg_box, chatbot, case_id_box, session_state, api_key_box],
        outputs=[chatbot, session_state],
    )
    msg_box.submit(
        chat,
        inputs=[msg_box, chatbot, case_id_box, session_state, api_key_box],
        outputs=[chatbot, session_state],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
