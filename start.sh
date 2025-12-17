#!/bin/bash

# 1. Start the Backend
# I am assuming your main.py is inside the 'app' folder based on your screenshot.
# If main.py is in the root, remove 'app.' and use 'uvicorn main:app ...'
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# 2. Start the Frontend
# We now point to 'frontend/app.py' because the script is in the root
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0