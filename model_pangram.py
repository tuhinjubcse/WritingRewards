from pangram import Pangram
import os

assert os.getenv("PANGRAM_API_KEY") is not None, "PANGRAM_API_KEY is not set"

class PangramModel:
    def __init__(self):
        self.pangram_client = Pangram()

    def predict(self, text):
        # {'text': 'The quick brown fox jumps over the lazy dog.', 'ai_likelihood': 0.0029010772705078125, 'prediction': 'Unlikely AI', 'llm_prediction': {'GPT35': 0.0, 'GPT4': 0.0, 'MISTRAL': 0.0, 'LLAMA': 0.0, 'GEMINI': 0.0, 'CLAUDE': 0.0, 'HUMAN': 0.0}, 'metadata': {}}
        keys_to_keep = ["ai_likelihood", "prediction", "llm_prediction"]
        result = self.pangram_client.predict(text)
        return {k: v for k, v in result.items() if k in keys_to_keep}

# pangram_client = Pangram()
# text = "The quick brown fox jumps over the lazy dog."
# result = pangram_client.predict(text)
# print(result)

if __name__ == "__main__":
    model = PangramModel()
    text = "The quick brown fox jumps over the lazy dog."
    result = model.predict(text)
    print(result)
