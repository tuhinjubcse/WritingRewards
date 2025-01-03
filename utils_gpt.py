from datetime import datetime
from openai import OpenAI
import time, json, os

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def prepare_training_data(json_file_path, output_file_path):
    """Convert JSON data to OpenAI's fine-tuning format"""
    assert json_file_path.endswith(".json"), "JSON file must have a .json extension"
    assert output_file_path.endswith(".jsonl"), "Output file must have a .jsonl extension"
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Format data for fine-tuning
    formatted_data = [
        {
            "messages": [
                {"role": "user", "content": item["text_input"]},
                {"role": "assistant", "content": item["output"]}
            ]
        }
        for item in data
    ]
    
    # Save formatted data to a JSONL file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for entry in formatted_data:
            json.dump(entry, file)
            file.write('\n')

def create_fine_tune(training_file_path):
    """Create and monitor fine-tuning job"""
    try:
        # Upload the training file
        print("Uploading training file...")
        training_file = client.files.create(
            file=open(training_file_path, 'rb'),
            purpose='fine-tune'
        )

        # Create fine-tuning job
        print("Creating fine-tuning job...")
        job = client.fine_tuning.jobs.create(
            training_file=training_file.id,
            model="gpt-4-0125-preview"  # or specify another base model
        )

        # Monitor the fine-tuning progress
        while True:
            job_status = client.fine_tuning.jobs.retrieve(job.id)
            print(f"Status: {job_status.status}")
            
            if job_status.status in ['succeeded', 'failed']:
                break
                
            time.sleep(60)  # Check status every minute

        if job_status.status == 'succeeded':
            print(f"Fine-tuning completed! Fine-tuned model: {job_status.fine_tuned_model}")
        else:
            print("Fine-tuning failed. Please check the logs.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def track_job(job_id):
    while True:
        job_status = client.fine_tuning.jobs.retrieve(job_id)
        if job_status.status == "running":
            if not job_status.estimated_finish:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {job_status.status}, Estimated completion: N/A")
            else:
                estimated_completion = datetime.fromtimestamp(job_status.estimated_finish)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {job_status.status}, Estimated completion: {estimated_completion}")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Status: {job_status.status}")
        
        if job_status.status in ['succeeded', 'failed']:
            if job_status.status == "succeeded":
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fine-tuning job completed successfully")     
                print(f"Model: {job_status.fine_tuned_model}")       
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fine-tuning job failed")
            break
        time.sleep(60)  # Check status every minute


def main():
    # Replace with your JSON file path
    json_file_path = 'your_training_data.json'
    
    # Prepare training data
    training_file_path = prepare_training_data(json_file_path)
    
    # Start fine-tuning
    create_fine_tune(training_file_path)

if __name__ == "__main__":
    main()
