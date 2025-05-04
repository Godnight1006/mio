# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.10

# Create a non-root user and switch to it
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
# Set PYTHONUNBUFFERED to ensure logs are visible in Hugging Face Space
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the requirements first to leverage Docker cache
COPY --chown=user ./requirements.txt requirements.txt
# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the essential application files and directories
COPY --chown=user ./minionbot.py minionbot.py
COPY --chown=user ./tools.py tools.py
COPY --chown=user ./data.json data.json
COPY --chown=user ./date.json date.json
COPY --chown=user ./said_words.json said_words.json
COPY --chown=user ./said_words.txt said_words.txt
COPY --chown=user ./python/ ./python/
COPY --chown=user ./health_check_app.py health_check_app.py
COPY --chown=user ./start.sh start.sh

# Make start script executable
RUN chmod +x start.sh

# Command to run the startup script
CMD ["./start.sh"]
