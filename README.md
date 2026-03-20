# Find-your-plankton

## Run The App (Windows PowerShell)

### 1. Open the project folder

```powershell
cd "D:\college notes\6th sem\ML\ml project"
```

### 2. Activate the virtual environment

```powershell
.\.venv312\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this once in the same terminal and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Start the Streamlit app

Option A (recommended, uses project launcher script):

```powershell
.\run_app.ps1
```

Option B (direct command):

```powershell
python -m streamlit run app.py
```

### 4. Open in browser

Streamlit usually opens automatically. If not, open:

http://localhost:8501
