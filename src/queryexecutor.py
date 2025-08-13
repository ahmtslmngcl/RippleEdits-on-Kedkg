import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPTJForCausalLM, GPTNeoXForCausalLM, LlamaForCausalLM
from utils import call_openai, process_generation


class QueryExecutor:

    def __init__(self, model=None, tokenizer=None, device=None, send_to_device=True):
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device
        if send_to_device:
            self._model = model.to(self._device)
        else:
            self._model = model
        self._tokenizer = tokenizer
        self._prompt_context = ''
        # template support for kedkg prompts
        self._prompt_template = None
        self._prompt_placeholder = '<<<<QUESTION>>>>'

    def get_model(self):
        return self._model

    def set_model(self, model):
        self._model = model.to(self._device)

    def get_tokenizer(self):
        return self._tokenizer

    def get_device(self):
        return self._device

    def set_prompt_context(self, context):
        self._prompt_context = context

    def set_prompt_template(self, template: str, placeholder: str = '<<<<QUESTION>>>>'):
        self._prompt_template = template
        self._prompt_placeholder = placeholder

    def set_prompt_template_from_file(self, path: str, placeholder: str = '<<<<QUESTION>>>>'):
        with open(path, 'r', encoding='utf-8') as f:
            self._prompt_template = f.read()
        self._prompt_placeholder = placeholder

    @staticmethod
    def _verify_answer(model_answer, correct_answer):
        for answer in correct_answer:
            if True not in [possible_answer in model_answer for possible_answer in answer]:
                return False
        return True

    def execute_query(self, query, answer_length=30):
        question = query.get_query_prompt()
        core = self._prompt_template.replace(self._prompt_placeholder, question) if self._prompt_template else question
        prompt = self._prompt_context + core
        raw = self._generate_text(prompt, answer_length)
        model_answer = raw[len(prompt):].lstrip() if raw.startswith(prompt) else raw.replace(self._prompt_context, '', 1).lstrip()
        print(f'query: {query.to_dict()}\nmodel answer: {model_answer}')
        return self._verify_answer(model_answer, query.get_answers())

    def get_model_name(self):
        raise NotImplementedError()  # Override in concrete classes

    def _generate_text(self, prompt, length):
        raise NotImplementedError()  # Override in concrete classes


class HFQueryExecutor(QueryExecutor):

    def __init__(self, model=None, tokenizer=None, device=None, send_to_device=True):
        super().__init__(model, tokenizer, device, send_to_device)

    def get_model_name(self):
        raise NotImplementedError()  # Override in concrete classes

    def _generate_text(self, prompt, length):
        inputs = self._tokenizer.encode(prompt, return_tensors='pt').to(self._device)
        outputs = self._model.generate(
            inputs,
            do_sample=False,
            temperature=0,
            max_new_tokens=length,
            pad_token_id=getattr(self._tokenizer, "eos_token_id", None),
        )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)


class GPT2QueryExecutor(HFQueryExecutor):

    def __init__(self, model_size='xl', device=None, model=None, tokenizer=None):
        self._model_size = model_size
        self._model_name = f'gpt2-{self._model_size}'
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            tokenizer.pad_token = tokenizer.eos_token
        if model is None:
            model = GPT2LMHeadModel.from_pretrained(self._model_name, pad_token_id=tokenizer.eos_token_id)
        super().__init__(model, tokenizer, device)

    def get_model_name(self):
        return self._model_name


class GPTJQueryExecutor(HFQueryExecutor):

    def __init__(self, device=None, model=None, tokenizer=None):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
            tokenizer.pad_token = tokenizer.eos_token
        if model is None:
            model = GPTJForCausalLM.from_pretrained('EleutherAI/gpt-j-6B', pad_token_id=tokenizer.eos_token_id)
        super().__init__(model, tokenizer, device)

    def get_model_name(self):
        return 'EleutherAI_gpt-j-6B'


class GPTNeoXQueryExecutor(HFQueryExecutor):

    def __init__(self, device=None, model=None, tokenizer=None):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
            tokenizer.pad_token = tokenizer.eos_token
        if model is None:
            model = GPTNeoXForCausalLM.from_pretrained('EleutherAI/gpt-neox-20b', device_map="auto", offload_folder="offload", offload_state_dict=True, pad_token_id=tokenizer.eos_token_id)
        super().__init__(model, tokenizer, device, send_to_device=False)

    def get_model_name(self):
        return 'EleutherAI_gpt-neox-20b'


class LlamaQueryExecutor(HFQueryExecutor):

    def __init__(self, model_size='7b', device=None, model=None, tokenizer=None):
        self._model_size = model_size
        self._model_name = f'llama-{self._model_size}'
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(f'huggyllama/{self._model_name}', use_fast=False, add_bos_token=False)
            tokenizer.pad_token = tokenizer.eos_token
        if model is None:
            model = LlamaForCausalLM.from_pretrained(f'huggyllama/{self._model_name}', device_map="auto", offload_folder="offload", offload_state_dict=True)
        super().__init__(model, tokenizer, device, send_to_device=False)

    def get_model_name(self):
        return self._model_name


class GPT3QueryExecutor(QueryExecutor):

    def __init__(self, model_size='text-davinci-003'):
        self._model_size = model_size
        super().__init__(send_to_device=False)

    def get_model_name(self):
        return self._model_size

    def _generate_text(self, prompt, length):
        text, log_probs = call_openai(
            prompt=prompt,
            model=self._model_size,
            temperature=0,
            max_tokens=length,
        )
        text = f'{prompt} {process_generation(text)}'
        return text
    
class Gemma3_4BExecutor(QueryExecutor):

    def __init__(self, model_size='gemma3:4b'):
        self._model_size = model_size
        super().__init__(send_to_device=False)
    
    def get_model_name(self):
        return self._model_size
    
    def _generate_text(self, prompt, length):
        text, log_probs = call_openai(
            prompt=prompt,
            model=self._model_size,
            temperature=0,
            max_tokens=length,
        )
        text = f'{prompt} {process_generation(text)}'
        return text
