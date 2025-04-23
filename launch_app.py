import threading
import os
import webview
import pathlib

def start_streamlit():
    # Get absolute path of the app file
    app_path = pathlib.Path(__file__).parent / "main_app.py"
    os.system(f"streamlit run \"{app_path}\" --server.headless true")

if __name__ == '__main__':
    threading.Thread(target=start_streamlit, daemon=True).start()
    webview.create_window("Offline PDF Q&A", "http://localhost:8501", width=1200, height=800)
    webview.start()
