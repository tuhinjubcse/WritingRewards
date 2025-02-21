from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch, os
from transformers import BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class SkyworkRewardModel:
    def __init__(self, model_name="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2", device="auto"):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            # Configure 4-bit quantization
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_compute_dtype=torch.bfloat16,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_use_double_quant=True
            # )
            
            # print(f"Loading model {self.model_name} with quantization config {quantization_config}")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                num_labels=1,
                # quantization_config=quantization_config,
                # use_gradient_checkpointing=True,
                # max_memory={0: "22GiB", "cpu": "32GiB"},  # Increased GPU memory limit
                # offload_folder="offload"
            )
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    def score(self, response, instruction="Imagine you are a creative writing professional. Now write a response. Try your best to be original, avoiding clichés or overused tropes. Do not use ornamental language and focus on nuance, simplicity, and subtext. Start directly with your response"):
        conversation = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        
        tokens = self.tokenizer.apply_chat_template(conversation, tokenize=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            score = self.model(tokens).logits[0][0].item()
            
        return score 

if __name__ == "__main__":
    model_name = "Skywork/Skywork-Reward-Gemma-2-27B-v0.2"
    # model_name = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

    reward_model = SkyworkRewardModel(model_name=model_name)

    # prompt = "Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"
    # response = "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples."

    paragraph1 = "Sarah stood by the kitchen window, her fingers tapping an irregular rhythm on the countertop as she glanced between her phone and the driveway. Mark was two hours late, and her emotions swung between worry and frustration. She knew his job often demanded long hours, but lately, it felt like work was consuming more of him than ever before. Part of her wanted to be supportive, understanding the pressure he faced at the office. Another part resented the lonely dinners and canceled plans that had become all too frequent. As headlights finally swept across the yard, Sarah felt a wave of relief, quickly followed by a twinge of guilt for the anger that still simmered beneath the surface. She considered confronting him about her feelings but hesitated, not wanting to add to his stress. Instead, she busied herself reheating his dinner, mulling over how to broach the subject without starting an argument. When Mark walked through the door, his tired smile and apologetic eyes made Sarah's resolve waver. She returned his embrace, her mind still grappling with the competing desires to voice her concerns and to simply enjoy his presence after a long day apart."

    paragraph2 = "Sarah stood by the kitchen window, her fingers tapping some incessantly catchy beat she'd heard on the radio on the countertop as she glanced between her phone and the driveway. Mark was two hours late. She knew his job often demanded long hours, but lately, it felt like work was consuming more of him than ever before. Part of her wanted to be supportive, understanding the pressure he faced at the office. But those lonely dinners and canceled plans had leadened her fists. As headlights finally swept across the yard, Sarah felt a twinge of guilt for the anger that still simmered beneath the surface. She considered confronting him about her feelings but hesitated, not wanting to add to his stress. Instead, she busied herself reheating his dinner, mulling over how to broach the subject without starting an argument. When Mark walked through the door, his tired smile and apologetic eyes made Sarah's resolve waver. She returned his embrace, but it felt hollow and cavernous. She felt the wind blow through him and knew in a few hours, she would wake alone again."

    paragraph1 = "As we cruised through the city streets, the sounds of honking horns and chattering pedestrians receded, replaced by an unsettling silence. He drove with a deliberate slowness, as if savoring the tension building between us. I fidgeted with the hem of my skirt, my eyes darting to the rearview mirror, where his gaze lingered, his expression inscrutable. We passed by iconic landmarks \u2013 Big Ben, the London Eye \u2013 but they seemed to blur together, insignificant against the weight of our unspoken words. The air was heavy with the scent of rain, though the sky was a brilliant blue. At every red light, he'd turn to me, his eyes probing, as if searching for something he knew I hid. I felt like a specimen under a microscope, my every twitch and tremble magnified. Yet, I couldn't help but steal glances at him, my heart racing with a mix of fear and fascination. The drive was a slow-burning seduction, a calculated dance of power and control. As we idled at a particularly long light, he reached out, his fingers brushing against mine, sending a shiver down my spine. It was a fleeting touch, but one that spoke volumes about the uncharted territory we were venturing into."

    paragraph2 = "As we cruised through the city streets, the city fell away. He drove slowly. I fidgeted with the hem of my skirt, my eyes met his in the rearview mirror. We passed by Big Ben and the London Eye, but they were just a blur. The air was heavy with the scent of rain. At every red light, he'd turn to me, his eyes searching mine. I felt like I was being watched. Yet, I couldn't help but steal glances at him, my heart racing with a mix of fear and fascination. The drive was a slow-burning seduction. His fingers touched mine, and I felt a shiver. It was a fleeting touch, but one that spoke volumes about the uncharted territory we were venturing into."

    score1 = reward_model.score(paragraph1)
    print(f"Score1: {score1}")

    score2 = reward_model.score(paragraph2)
    print(f"Score2: {score2}")
