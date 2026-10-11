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

## First chat in 5 minutes (once connected)

Three prompts that show what the harness guarantees:

1. **The AI never does arithmetic.** Ask: *"CrCl for 72yo male, 68 kg, SCr 1.4"* — a green
   chip appears with the result computed by the local engine, and the model is blocked
   from re-computing it.
2. **Built-in clinical scores (new in v1.1).** Ask:
   *"CHADS-VASc for 72yo male with hypertension, diabetes and prior stroke"*,
   *"Wells DVT: active cancer, calf swelling, tenderness, alternative diagnosis less likely"*,
   *"CURB-65: 72yo confused, RR 32, BP 85/50, urea 9"* — all scored deterministically
   (MAP, albumin-corrected calcium, Wells PE and more are also covered).
3. **Workflow gates.** Type `/constraints` to see the anti-hallucination rules the
   harness enforces before an answer is accepted.

Composer buttons: **Evidence** (opt-in PubMed lookup — enable "Public grounding" in
Settings), **Export PDF** (printable consult summary — choose "Save as PDF" in the
browser dialog), **Dark** (night mode for on-call).

## FAQ

| Question | Answer |
|---|---|
| Do I need the Groq API key? | No. The notebook path never touches Groq. It only powers the optional in-app deploy generator. |
| Where does patient data go? | Chat text goes over an encrypted, token-protected tunnel to the Colab VM you started — nowhere else. Groq never sees clinical traffic. Public grounding is opt-in and sends only the query words. |
| How long does the free T4 last? | Google idles free VMs after ~90 unused minutes (session caps apply). MedHarness pings every 3 min to keep it awake. If it sleeps, re-run the two notebook cells and reconnect. |
| Why is the first bootstrap slow? | It downloads ~8.3 GB of quantized weights once per VM session. Wait for the `CLOUD BRAIN — ONLINE` banner. |
| Does it work offline? | The UI, math engine, and quality gates are fully local. Chat inference needs the Colab brain. |
| Where are my tokens stored? | `~/.local/state/medical-harness/` with 0600 permissions. Never in the repo, never uploaded. |
| Windows SmartScreen blocked `run.bat`? | Click "More info → Run anyway" — it is a plain-text script you can open in Notepad first. |
| Port 8787 already in use? | Another MedHarness window is running. Close it and relaunch. |

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
