# Deploying to Render

This project is configured to deploy on Render with both FastAPI backend and Streamlit frontend.

## Prerequisites
- GitHub account with this repository
- Render account (https://render.com)

## Deployment Steps

### 1. Push Code to GitHub
```bash
git add .
git commit -m "Configure Render deployment"
git push origin main
```

### 2. Connect Repository to Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New +** → **Blueprint**
3. Select **Connect GitHub Account** and authorize Render
4. Find and select this repository
5. Click **Connect**

### 3. Deploy from Blueprint

1. Render will automatically detect `render.yaml`
2. Review the configuration:
   - **fpl-backend**: FastAPI service on port 8000
   - **fpl-frontend**: Streamlit app on port 8501
3. Click **Create New Services**
4. Wait for both services to deploy (5-10 minutes)

### 4. Access Your Application

Once deployed:
- **Frontend (Streamlit)**: Click the URL for `fpl-frontend` service
- **Backend (FastAPI)**: Click the URL for `fpl-backend` service
  - Health check: `https://<backend-url>/health`
  - Docs: `https://<backend-url>/docs`

### 5. Update Environment Variables (if needed)

If the frontend can't connect to the backend:
1. Go to `fpl-frontend` service settings
2. Add/Update environment variable:
   - Key: `FPL_API_BASE_URL`
   - Value: `https://<backend-url>` (without trailing slash)
3. Redeploy the frontend service

## Local Development

### Run Backend
```bash
python -m uvicorn app.main:app --reload
```

### Run Frontend (in another terminal)
```bash
streamlit run frontend/app.py
```

Then open `http://localhost:8501`

## Architecture

```
┌─────────────────────────────────────┐
│   Streamlit Frontend (8501)          │
│   - Interactive Dashboard           │
│   - Squad Optimization UI           │
│   - KPI Visualizations              │
└────────────┬────────────────────────┘
             │ HTTP Requests
             ↓
┌─────────────────────────────────────┐
│   FastAPI Backend (8000)             │
│   - ML Model Services                │
│   - Data Processing                  │
│   - Squad Optimization Logic         │
└──────────────────────────────────────┘
             │
             ↓
        CSV Data Files
       & ML Models (joblib)
```

## Troubleshooting

### Frontend can't connect to backend
- Check backend service is running: `https://<backend-url>/health`
- Verify `FPL_API_BASE_URL` env var is set correctly
- Check CORS is enabled in FastAPI (it is by default)

### Models/Data not loading
- Ensure `models/` and `data/` folders are included in deployment
- Check Streamlit logs on Render dashboard

### Build fails
- Verify all dependencies are in `requirements.txt`
- Check Python version compatibility (3.11)
