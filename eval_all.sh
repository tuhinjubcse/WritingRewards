#!/bin/bash

# Set maximum number of parallel processes
MAX_PARALLEL=4

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to cleanup on exit
cleanup() {
    echo -e "\nCleaning up..."
    # Kill the progress monitor
    kill $FOLLOW_PID 2>/dev/null
    # Kill all background python processes
    pkill -P $$ python
    exit
}

# Trap Ctrl+C (SIGINT) and script exit
trap cleanup EXIT SIGINT

# Function to sanitize model name for filename
sanitize_model_name() {
    echo "$1" | sed 's/[:\/ ]/_/g'
}

# Function to follow last 4 tqdm bars
follow_progress() {
    while true; do
        clear
        echo "Monitoring progress of running evaluations..."
        echo "----------------------------------------"
        # Find the 4 most recently modified log files and show their last tqdm lines
        find logs -type f -name "*.log" -mmin -10 | sort -r | head -n 4 | while read logfile; do
            model=$(basename "$logfile" .log)
            echo -e "\n=== $model ==="
            # Try to find the last line with a percentage
            tqdm_line=$(grep -a "%" "$logfile" | tail -n 1)
            if [ -n "$tqdm_line" ]; then
                echo "$tqdm_line"
            else
                echo "Waiting for progress..."
            fi
        done
        sleep 2
    done
}

# Start progress monitoring in background before starting evaluations
follow_progress &
FOLLOW_PID=$!

# Main experiment on LAMP split

# for model in gpt-4o-mini gemini-1.5-flash tunedModels/lamp-gem-1p5-flash-p-d; do # base models
# for model in tunedModels/lamp-gem-1p5-flash-p-e gpt-4o-2024-08-06 tunedModels/lamp-gem-1p5-flash-s-a tunedModels/lamp-gem-1p5-flash-s-b tunedModels/lamp-gem-1p5-flash-ps-a tunedModels/lamp-gem-1p5-flash-ps-b  tunedModels/lamp-gem-1p5-flash-s1000-a  tunedModels/lamp-gem-1p5-flash-s2000-a tunedModels/lamp-gem-1p5-flash-s4000-a  tunedModels/lamp-gem-1p5-flash-s8000-a  tunedModels/lamp-gem-1p5-flash-s16000-a; do # tuned Gemini models

#  ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-s1000:AnBiuKOa ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-s2000:AnD7pa0n ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-s4000:AnEcSYEK ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-s8000:AnGk0JUW ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-cot:Aqlv1wPq ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-detection:Aqna3sJ6
for model in ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-p:AYKM53Ac \
    ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-r-eval:AtJMxSx2 \
    ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-pr-eval:AtdmpEpc \
    ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-p-eval-b:AzdkWgJT \
    ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-r-eval-b:AzcFoRjX \
    ft:gpt-4o-2024-08-06:salesforce-research:lamp-4o-rsg2-eval:AzbVZNmb \
    gpt-4o; do # tuned GPT models
    
    # Create a safe filename
    safe_name=$(sanitize_model_name "$model")
    
    # Run each evaluation in background with output redirected to a log file
    PYTHONUNBUFFERED=1 python -u populate_eval.py --model "$model" --input_fn data/lamp_PRGSH_test.json > "logs/${safe_name}.log" 2>&1 &
    
    echo "Started evaluation for $model"
    
    # Limit number of parallel processes
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
done

# Wait for all evaluation processes to complete
wait

echo "All evaluations completed. Check logs directory for output."

# The trap will handle killing the follow_progress process

# Fill in for editor split
# python populate_eval.py --input_fn data/lamp_PR_editor_test.json --model gpt-4o-mini-2024-07-18
# python populate_eval.py --input_fn data/lamp_PR_editor_test.json --model ft:gpt-4o-mini-2024-07-18:salesforce-research:lamp-4o-mini-p-editor:AZpOgz62