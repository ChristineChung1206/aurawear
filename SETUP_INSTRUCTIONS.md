# AuraWear Analysis — Setup Instructions

This guide walks you through running the AuraWear Analysis app on your own computer from scratch. No prior coding experience is required.

By default, the app runs in **synthetic demo mode** using a bundled sample catalogue and placeholder images. This lets you explore all features without downloading any external datasets.

---

## What You Will Need

- A computer running **macOS, Windows, or Linux**
- About **4 GB of free disk space** (for Miniconda, Python packages, and the first-run CLIP model download)
- A stable internet connection (for the one-time setup)
- An **OpenAI API Key** (optional — the app works without one, but AI explanation features will be disabled)

---

## Step 1 — Install Miniconda

Miniconda is a lightweight tool that manages Python environments. It keeps everything self-contained and avoids conflicts with other software on your computer.

1. Go to: https://docs.conda.io/en/latest/miniconda.html
2. Download the installer for your operating system:
   - **macOS (Apple Silicon / M1/M2/M3)** → choose *Miniconda3 macOS Apple Silicon*
   - **macOS (Intel)** → choose *Miniconda3 macOS Intel x86*
   - **Windows** → choose *Miniconda3 Windows 64-bit*
   - **Linux** → choose *Miniconda3 Linux 64-bit*
3. Run the installer and follow the on-screen instructions. When asked, accept the defaults.
4. **Restart your terminal** (or open a new terminal window) after installation.

To verify it worked, open a terminal and run:
```
conda --version
```
You should see something like `conda 24.x.x`.

---

## Step 2 — Open a Terminal

- **macOS**: Press `Cmd + Space`, type `Terminal`, press Enter.
- **Windows**: Search for `Anaconda Prompt` in the Start menu and open it.
- **Linux**: Open your system terminal application.

---

## Step 3 — Create a Python Environment

In the terminal, type the following commands **one at a time**, pressing Enter after each:

```bash
conda create -n aurawear python=3.10
```

When asked `Proceed ([y]/n)?`, type `y` and press Enter. This takes about 1–2 minutes.

Then activate the environment:
```bash
conda activate aurawear
```

You should now see `(aurawear)` at the beginning of your terminal prompt. **You need to do this activation step every time you open a new terminal.**

---

## Step 4 — Navigate to the Project Folder

The folder you cloned or downloaded (named `aurawear-public`) contains all the project files. In the terminal, navigate into it.

For example, if you cloned or downloaded and unzipped it to your Desktop:

- **macOS / Linux**:
  ```bash
  cd ~/Desktop/aurawear-public
  ```
- **Windows**:
  ```bash
  cd C:\Users\YourName\Desktop\aurawear-public
  ```

Replace the path above with the actual location on your computer.

To confirm you are in the right place, run:
```bash
ls
```
You should see files like `app_gradio.py` and `requirements.txt` listed.

---

## Step 5 — Set Up the Configuration File

The repository includes a template configuration file called `.env.example`. You need to copy it to `.env` before running the app. The `.env` file is **not committed to GitHub** (it is listed in `.gitignore`), so each user creates their own local copy.

Run:

```bash
cp .env.example .env
```

> **Windows users:** Use `copy .env.example .env` instead.

The default settings in `.env` are already configured for **synthetic demo mode** — no additional data downloads are required. You can open `.env` in any text editor to review or change the settings.

---

## Step 6 — Install Required Packages

Run the following command. This installs all the software libraries the app depends on. It may take **5–10 minutes**.

```bash
pip install -r requirements.txt
```

Wait until it finishes and you see the terminal prompt again.

> **Note for Windows users:** If you see an error related to `torch`, try installing it separately first:
> ```
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> Then re-run `pip install -r requirements.txt`.

---

## Step 7 — Add Your OpenAI API Key (Optional)

The app works without this. Skipping this step means the AI-powered explanation and suggestion features will be turned off, but everything else will still work normally.

If you have an OpenAI API key:

1. Open the `.env` file you created in Step 5 using any text editor (Notepad on Windows, TextEdit on macOS, or VS Code).
2. Find the line:
   ```
   OPENAI_API_KEY=
   ```
3. Paste your key directly after the `=` sign, so it looks like:
   ```
   OPENAI_API_KEY=sk-...your-key-here...
   ```
4. Save and close the file.

> **How to get an OpenAI API Key:** Go to https://platform.openai.com/api-keys, sign in, and create a new key. Note that usage incurs costs billed to your OpenAI account.

---

## Step 8 — Run the App

In the terminal (make sure you still see `(aurawear)` at the start of the prompt), run:

```bash
python app_gradio.py
```

The first time you run this, it will download the CLIP model (~300 MB) automatically. This only happens once. Wait until you see a message like:

```
Running on local URL:  http://127.0.0.1:7860
```

---

## Step 9 — Open the App in Your Browser

Open any web browser (Chrome, Safari, Firefox, Edge) and go to:

```
http://127.0.0.1:7860
```

The AuraWear Analysis interface will appear. You are ready to go.

---

## Stopping the App

To stop the app, go back to the terminal and press `Ctrl + C`.

---

## Starting the App Again Next Time

Every time you want to use the app, open a terminal and run these two commands:

```bash
conda activate aurawear
```
```bash
cd /path/to/aurawear-public
python app_gradio.py
```

Then open `http://127.0.0.1:7860` in your browser.

---

## Troubleshooting

**`conda: command not found` after installing Miniconda**
→ Close the terminal completely and open a fresh one. On macOS, you may need to run `source ~/.zshrc` or `source ~/.bash_profile`.

**`ModuleNotFoundError` when starting the app**
→ Make sure you activated the environment first: `conda activate aurawear`. Then re-run `pip install -r requirements.txt`.

**The page at `127.0.0.1:7860` does not load**
→ The app is still starting up. Wait 30 seconds and refresh. If it still does not load, check the terminal for any error messages.

**Items or images are not showing**
→ Make sure `data/sample_catalog.csv` and the `assets/synthetic_demo_images/` directory are present. They are required for the app to work in demo mode.

---

## Folder Structure Overview

```
aurawear-public/
├── app_gradio.py               ← main app (run this)
├── requirements.txt            ← list of required packages
├── .env.example                ← configuration template (committed to git)
├── .env                        ← your local configuration (copy from .env.example; not in git)
├── data/
│   └── sample_catalog.csv      ← bundled synthetic product catalogue
├── assets/
│   └── synthetic_demo_images/  ← placeholder images for demo mode
└── aurawear_analysis/          ← core analysis library
    ├── assets/
    │   ├── palette18.json
    │   └── models/             ← optional face parsing model (not bundled; see models/README.md)
    ├── color_analysis/         ← skin/eye/hair colour detection
    └── recommend/              ← recommendation engine
```

---

### Full Reproduction (Advanced)

The synthetic demo uses a small bundled catalogue. To reproduce results with the full DeepFashion-MultiModal dataset, you must obtain the data directly from the original source and run the preprocessing scripts in `tools/` locally. This is subject to the original dataset licenses and is not supported in this public release.

See [DATA_POLICY.md](DATA_POLICY.md) for details.
