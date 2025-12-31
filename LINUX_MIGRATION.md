# 🐧 Linux Migration Guide

Welcome to the Linux environment! This guide will help you migrate your workflow from Windows to Linux (Ubuntu/Debian) seamlessly.

---

## 🚀 1. System Setup (First Time Only)

Open your terminal and install the necessary system tools:

```bash
# Update package list
sudo apt update

# Install Git, Python, and Pip
sudo apt install -y git python3 python3-pip python3-venv
```

---

## 📥 2. Clone & Setup Repository

Get the latest code and set up your fresh environment.

```bash
# 1. Clone your repository
git clone https://github.com/farhann-saleem/Alaska-Wildfire-prediction-MVP.git
cd Alaska-Wildfire-prediction-MVP

# 2. Create a clean virtual environment
python3 -m venv venv

# 3. Activate it (This is different from Windows!)
source venv/bin/activate

# 4. Install project dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🌍 3. Google Earth Engine Authentication (Critical)

Since this is a new machine, you **MUST** re-authenticate with Google Earth Engine.

```bash
# Run the authentication flow
earthengine authenticate
```

*   **Headless Server?** If you are on a remote server (no browser), follow the link provided in the terminal, copy the code, and paste it back.
*   **Desktop Linux?** A browser window will open automatically.

**Verify Project ID:**
Make sure strictly that `scripts/era5_analysis.py` is using your correct Cloud Project ID:
```python
# Check line 22 in scripts/era5_analysis.py
GEE_PROJECT = 'alaska-dataset'  # Update this if yours is different!
```

---

## ▶️ 4. Running the Phase 2 Analysis

To reproduce your "Alaska Anomaly" results on Linux:

```bash
# Ensure venv is active
source venv/bin/activate

# Run the analysis script
python3 scripts/era5_analysis.py
```

---

## 🛠️ Common Linux Commands You'll Need

| Action | Windows Command | Linux Command |
| :--- | :--- | :--- |
| **Activate Env** | `.\venv\Scripts\Activate` | `source venv/bin/activate` |
| **List Files** | `dir` | `ls -la` |
| **Clear Screen** | `cls` | `clear` |
| **Delete File** | `del filename` | `rm filename` |
| **Delete Folder** | `rmdir /s folder` | `rm -rf folder` |

---

### ✅ Checklist for Success
- [ ] `git clone` successful
- [ ] `source venv/bin/activate` shows `(venv)` in prompt
- [ ] `pip install` completed without errors
- [ ] `earthengine authenticate` successfully saved credentials
- [ ] `python3 scripts/era5_analysis.py` runs and prints "Fetching ERA5 data..."
