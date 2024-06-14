#!/bin/bash

# Set the Python command or script to check
PYTHON_COMMAND="python ./scrap_comments.py"
PYTHON_PID=""

terminate_script() {
    echo "Terminating the monitoring script..."
    echo "Bash scrip received SIGINT signal. Sending signal to Python script..."
    if [ -n "$PYTHON_PID" ]; then
        kill -SIGINT "$PYTHON_PID"
    fi
    exit 0
}

trap 'terminate_script' SIGINT SIGTERM

while true; do
    # Check if the Python command is running
    if ! pgrep -f "$PYTHON_COMMAND" > /dev/null; then
        echo "$(date) - Python command not running, starting it now."
        $PYTHON_COMMAND &
        PYTHON_PID=$!
    else
        echo "$(date) - Python command is running."
    fi

    # Wait for 1 minute
    sleep 60
done
