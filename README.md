# Streamlit interface

Experimental research tool. It estimates whether audio *sounds synthetic*; it cannot prove who is
speaking, does not monitor phone calls, and never disconnects or blocks anything.

## Files
| File | Role |
|---|---|
| `app.py` | Page layout and widget flow only |
| `ui_components.py` | Drawing helpers (status card, tables, charts, alert); no model code |
| `services.py` | Everything that touches the model, streaming detector, challenge session and decision logic; no Streamlit imports |
| `audio_io.py` | Decodes uploaded / recorded audio via the project's single loader |
| `beep.py` | Generates the optional alert tone |

## Run (from the repository root)
```bash
streamlit run streamlit_app/app.py
```

