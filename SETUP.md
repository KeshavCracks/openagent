# MedHarness — Setup Guide (start here)

You need: **Python 3.10+** on your computer, a **Google account** (for the free Colab GPU),
and the two files you already downloaded. A **Groq API key is optional** — see the box below.

---

## Step 1 — Start the desktop app (2 minutes)

**Windows:** double-click `run.bat`
**Linux / macOS:** open a terminal in the folder and run `./run.sh` (or `bash run.sh`)

First run installs 5 small packages into a private `.venv` (about 30 seconds), then a
browser window opens at `http://127.0.0.1:8787`. Keep this window open — this is your app.

Prefer manual commands instead of the launcher?

```bash
cd medical-harness
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python -m app.main                   # Windows: .venv\Scripts\python -m app.main
```

## Step 2 — Start the free cloud brain in Colab (3–8 minutes)

1. Open **colab.research.google.com** → **File → Upload notebook** → pick
   `MedHarness_CloudBrain.ipynb` (or use the one-click "Open In Colab" badge in the repo README).
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run **cell 1** (▶ button). It prints a long **session token** — copy it.
4. Run **cell 2**. It downloads both models and starts the engines (3–8 min).
   When the banner prints a **`https://….trycloudflare.com`** URL — copy that too.
5. Keep the Colab tab open while you use the app (free VMs sleep when closed).

## Step 3 — Connect and chat

Back in the desktop app: **Settings → Cloud Brain** →
paste the **session token** and the **tunnel URL** → **Connect**.
The status chip turns **healthy** — chat away. The app keeps the VM awake automatically.

---

## Where does the Groq API key go?

Nowhere, unless you want it. The notebook path above never touches Groq.
Paste it in **Settings → Groq API key** only if you want the in-app *Deploy* generator
(Groq acts as a headless DevOps assistant that drafts deployment cells — it never
sees clinical traffic).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: externally-managed-environment` | Use the launcher (`run.sh` / `run.bat`) — it builds a private venv. Manual: `pip install --user -r requirements.txt`. |
| `No module named venv` (Debian/Ubuntu) | `sudo apt install python3-venv python3-pip` |
| Colab cell 2 fails on a model download | A public GPTQ mirror moved. Set `MEDR1_GPTQ_REPO` / `MEDGEMMA_GPTQ_REPO` in cell 1 to a GPTQ W4A16 repo id, or follow the quantize-once recipe printed in the error. |
| "Cloud brain not connected" in chat | Tunnel URL missing/stale. Re-run cell 2's last lines or restart the notebook, paste the fresh URL, Connect. |
| Colab VM went to sleep | Free VMs idle out after ~90 min. Re-run both cells and reconnect with the new URL + token. |
| Port 8787 busy | Another instance is running. Close it, or run with a different `--port` if you edited config. |

> Decision support for trained clinicians — not a medical device. Verify outputs
> against institutional protocols.
