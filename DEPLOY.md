# How to get this online

Follow these in order. Nothing costs money.

---

## Part 1 — Run it on your own computer first (10 minutes)

**1. Install Python 3.10 or newer.** Check what you have:

```bash
python3 --version
```

If that errors, download it from [python.org](https://www.python.org/downloads/).

**2. Open a terminal in this folder** and install what the app needs:

```bash
pip install -r requirements.txt
```

**3. Start the app:**

```bash
streamlit run app.py
```

Your browser opens to `http://localhost:8501`. That's it — the data for
2019–2025 is already included, so it works immediately.

Press `Ctrl+C` in the terminal to stop it.

---

## Part 2 — Put it on the internet (15 minutes)

Streamlit Community Cloud hosts this free and gives you a public link.
It reads your code from GitHub, so GitHub comes first.

### Step 1 — Make a GitHub account

Go to [github.com](https://github.com) and sign up if you don't have one.

### Step 2 — Install Git

Check if you already have it:

```bash
git --version
```

If not, download from [git-scm.com](https://git-scm.com/downloads).

### Step 3 — Create an empty repository

On GitHub, click **+** (top right) → **New repository**.

- Name: `nfl-contract-value`
- Set it to **Public** (required for the free Streamlit tier)
- **Do not** check "Add a README" — this folder already has one
- Click **Create repository**

Leave that page open. You'll need the URL it shows you.

### Step 4 — Upload your code

In your terminal, in this folder:

```bash
git init
git add .
git commit -m "NFL contract value app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/nfl-contract-value.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your actual GitHub username.

If it asks for a password, GitHub wants a **personal access token**, not your
account password. Go to GitHub → Settings → Developer settings → Personal access
tokens → Tokens (classic) → Generate new token, tick the `repo` box, and paste
the generated token as the password.

> The `.gitignore` file already excludes `data/raw/` (about 60 MB of downloaded
> source files). The two small `.parquet` files in `data/` **do** get uploaded —
> that's intentional, they're what the app reads.

### Step 5 — Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **Sign in with GitHub** and authorize it
3. Click **Create app** → **Deploy a public app from GitHub**
4. Fill in:
   - Repository: `YOUR-USERNAME/nfl-contract-value`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**

First build takes 2–5 minutes. You get a URL like
`https://nfl-contract-value.streamlit.app` that anyone can open.

### Step 6 — Add your link to the README

Open `README.md`, put your new URL at the top, then:

```bash
git add README.md
git commit -m "Add live link"
git push
```

Streamlit redeploys automatically every time you push. That's the whole update
loop from here on: change something, `git push`, it's live in a minute.

---

## Part 3 — Refreshing the data

The app ships with 2019–2025 already built. To pull in a new season, or refresh
mid-season:

```bash
python src/build_data.py --seasons 2019 2026
python src/value_model.py
git add data/*.parquet
git commit -m "Refresh data"
git push
```

The first command downloads and caches source files in `data/raw/`, so re-runs
are quick. Delete that folder to force a fresh download.

---

## If something breaks

**`streamlit: command not found`** — the install didn't finish or isn't on your
PATH. Try `python3 -m streamlit run app.py`.

**`ModuleNotFoundError: No module named 'pyarrow'`** — run
`pip install -r requirements.txt` again and read the output for errors.

**App deploys but shows a file-not-found error** — the `.parquet` files in
`data/` didn't get committed. Run `git status` to check, and make sure your
`.gitignore` only excludes `data/raw/`, not all of `data/`.

**Deploy button is greyed out on Streamlit** — the repo is private. Free tier
needs public. Change it in the repo's Settings → General → Danger Zone.
