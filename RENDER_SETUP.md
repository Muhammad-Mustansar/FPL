# Render Deployment Checklist

## What You Have
✅ 2 deployments running (wonderful-adaptation & protective-love)

## Environment Variables to Add in Render Dashboard

### For Backend Service (fpl-backend):
| Variable | Value | Notes |
|----------|-------|-------|
| `PYTHON_VERSION` | `3.11` | Python version |
| `PORT` | `8000` | Backend port (auto-set by Render) |
| `LOG_LEVEL` | `INFO` | Optional: logging level |

### For Frontend Service (fpl-frontend):
| Variable | Value | Notes |
|----------|-------|-------|
| `PYTHON_VERSION` | `3.11` | Python version |
| `PORT` | `8501` | Streamlit port |
| `FPL_API_BASE_URL` | From backend service | Automatically linked via render.yaml |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Required for Render |

## Start Commands

### Backend:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend (Streamlit):
```bash
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.enableXsrfProtection false
```

## Build Command (Both Services):
```bash
pip install -r requirements.txt
```

## Steps to Set Up on Render

### Option 1: Using Blueprint (Recommended)
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New +** → **Blueprint**
3. Select this GitHub repo
4. Render will auto-detect `render.yaml`
5. Click **Create New Services**

### Option 2: Manual Setup (If Blueprint Fails)

#### Backend Service:
1. Click **New +** → **Web Service**
2. Connect GitHub repo → Select this repo
3. Fill in:
   - **Name**: `fpl-backend`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`
   - **Plan**: Free
4. Add Environment Variables (from table above)
5. Click **Create Web Service**

#### Frontend Service:
1. Click **New +** → **Web Service**
2. Connect GitHub repo → Select this repo
3. Fill in:
   - **Name**: `fpl-frontend`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0 --server.enableXsrfProtection false`
   - **Plan**: Free
4. Add Environment Variables:
   - `PYTHON_VERSION`: `3.11`
   - `FPL_API_BASE_URL`: (get from backend service URL)
5. Click **Create Web Service**

## Connecting Frontend to Backend

### After Deployment:
1. Copy backend service URL (e.g., `https://fpl-backend-xxx.onrender.com`)
2. Go to frontend service → **Environment**
3. Add/Edit: `FPL_API_BASE_URL` = `https://fpl-backend-xxx.onrender.com`
4. Click **Save Changes** → Service redeploys

## Testing Deployment

Once both services are running:

### Test Backend Health:
```bash
curl https://<backend-url>/health
```

### Test Frontend:
Open: `https://<frontend-url>`

### Check Backend Docs:
Visit: `https://<backend-url>/docs`

## Troubleshooting

### Frontend shows "Connection Error"
- Check `FPL_API_BASE_URL` env var is set correctly
- Verify backend service is running (check Logs)
- Ensure no trailing slash in API URL

### Models not loading
- Check `models/` directory is in git
- Verify file paths in code use relative paths
- Check Render logs for file not found errors

### Streamlit shows blank page
- Check browser console for errors (F12)
- Verify all API calls succeed in Network tab
- Check Streamlit logs in Render dashboard

### Service crashes during build
- Check build logs in Render dashboard
- Ensure all dependencies in `requirements.txt`
- Verify Python version compatibility

## File Structure Render Expects:
```
├── app/
│   ├── main.py          ✅ Backend
│   ├── services/
│   └── schemas.py
├── frontend/
│   └── app.py           ✅ Streamlit frontend
├── ml/                  ✅ ML models
├── models/              ✅ Trained models (joblib files)
├── data/                ✅ CSV data files
├── requirements.txt     ✅ Dependencies
├── render.yaml          ✅ Deployment config
└── .streamlit/
    └── config.toml      ✅ Streamlit config
```

## Important Notes

- **Render Free Tier**: Services spin down after 15 min of inactivity
- **Models**: Ensure all model files are in `models/` directory (max 500MB)
- **Data**: Keep CSV files small; consider lazy loading
- **Logs**: Check Render dashboard → Service → Logs for debugging
