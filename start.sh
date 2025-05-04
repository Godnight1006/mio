#!/bin/bash

# Start the health check app in the background using the PORT environment variable provided by Render
echo "Starting health check app on port $PORT..."
uvicorn health_check_app:app --host 0.0.0.0 --port ${PORT:-10000} & # Use Render's PORT, default to 10000 if unset (Render uses various ports)

# Wait a few seconds for the health check app to start
sleep 5

# Start the Discord bot
echo "Starting Discord bot..."
python minionbot.py
