from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from collections import OrderedDict
import torch
import traceback

class GPT2_probabilistic_model_wrapper_batch(ProbabilisticModel):
    
    def __init__(self, max_seq_length: int, alphabet:Alphabet, device: str, model, tokenizer, prompt:Sequence = Sequence()):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.device = device
        self._alphabet = alphabet        
        self._prompt = prompt
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def name(self) -> str:
        return "GPT_2"
    
    @property
    def terminal_symbol(self) -> Symbol:
        return SymbolStr(self.tokenizer.eos_token)

    @property
    def alphabet(self) -> Alphabet:
        return self._alphabet

    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError
    
    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    
    def last_token_probability(self, sequence: Sequence, symbols = None) -> float:
        if symbols is None:
            symbols = set(self._alphabet.symbols)
            symbols.add(self.terminal_symbol)
        return self._get_probability(sequence, symbols)
    
    #TODO: Fix interface, this should be removed from the learners and pymodelextractor as a whole
    def get_last_token_weights(self, sequence, required_suffixes):
        print(f"[⚠️ NON-BATCH WARNING] Wrapper get_last_token_weights called directly on {sequence}")
        traceback.print_stack(limit=5)


        weights = list()
        alphabet_symbols_weights = self.last_token_probability(sequence, required_suffixes)
        for suffix in required_suffixes:
            assert suffix in alphabet_symbols_weights
        return [alphabet_symbols_weights[suffix] for suffix in required_suffixes]
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        
        print(f"Wrapper Batch CALL — Batch size: {len(sequences)} | Suffixes: {len(required_suffixes)}")

        # 1. Convertir secuencias en strings tokenizados (prompt + input)
        input_ids_batch = []
        for seq in sequences:
            token_strs = [self.tokenizer.tokenize(str(x)) for x in self._prompt + seq]
            tokens = [item for sublist in token_strs for item in sublist]
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = [self.tokenizer.bos_token_id] + token_ids
            input_ids_batch.append(torch.tensor(input_ids))

        # 2. Padding para batching
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)

        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()

        # 3. Forward del modelo en batch
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids_padded, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (B, L, V)
            last_logits = logits[range(len(sequences)), attention_mask.sum(1) - 1, :]  # último token real por secuencia
            probs = torch.softmax(last_logits, dim=-1)  # shape: (B, V)

        # 4. Tokenizar todos los símbolos una vez
        symbol_token_ids = {}
        for symbol in required_suffixes:
            key = str(symbol)
            tokens = self.tokenizer.tokenize(key)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            symbol_token_ids[key] = token_ids

        # 5. Calcular probabilidades
        batch_results = []
        for i in range(len(sequences)):
            seq_probs = []
            for symbol in required_suffixes:
                token_ids = symbol_token_ids[str(symbol)]
                if len(token_ids) == 1:
                    # Single-token: usar prob directa
                    prob = probs[i, token_ids[0]]
                else:
                    # Multi-token: hacer pasos adicionales
                    curr_input = input_ids_batch[i].clone().to(self.device)
                    prob = torch.tensor(1.0, device=self.device)
                    for j, token_id in enumerate(token_ids):
                        curr_input = torch.cat([curr_input, torch.tensor([token_id]).to(self.device)])
                        with torch.no_grad():
                            out = self.model(curr_input.unsqueeze(0))
                            next_logits = out.logits[:, -1, :]
                            next_probs = torch.softmax(next_logits, dim=-1)[0]
                        prob *= next_probs[token_ids[j+1]] if j+1 < len(token_ids) else next_probs[token_ids[j]]
                    prob = prob.item()
                seq_probs.append(prob if isinstance(prob, float) else prob.item())
            batch_results.append(seq_probs)

        return batch_results

    
    def build_ids_sequence_from_tokens(self, sequence):
        bos_token_id = [self.tokenizer.bos_token_id,]

        str_prompt = [self.tokenizer.tokenize(str(x)) for x in self._prompt]
        str_prompt = [item for tokens in str_prompt for item in tokens]

        prompt_ids = self.tokenizer.convert_tokens_to_ids(str_prompt)
        sequence_ids = self.tokenizer.convert_tokens_to_ids(sequence)
        return torch.tensor(bos_token_id + prompt_ids + sequence_ids).reshape(1, -1).to(self.device)

    def tokenize_empty(self):
        bos_token_id = [self.tokenizer.bos_token_id,]

        str_prompt = [self.tokenizer.tokenize(str(x)) for x in self._prompt]
        str_prompt = [item for tokens in str_prompt for item in tokens]

        prompt_ids = self.tokenizer.convert_tokens_to_ids(str_prompt)        
        return torch.tensor(bos_token_id + prompt_ids).reshape(1, -1).to(self.device)
    
    def _get_probability(self, sequence, symbols):
        if len(sequence) == 0:
            input_ids = self.tokenize_empty()
        else:
            str_seq = [self.tokenizer.tokenize(str(x)) for x in sequence]
            str_seq = [item for tokens in str_seq for item in tokens]
            input_ids = self.build_ids_sequence_from_tokens(str_seq)
        
        with torch.no_grad():
            output = self.model(input_ids)
            logits = output.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

        return self._get_symbols_probabilities_dict(input_ids, probs, symbols)

    # TODO: We should make sure that we are calculating the probabilities for the correct words
    # Since the tokenizer splits words in different ways, we should check that the probabilities
    # make sense
    def _get_symbols_probabilities_dict(self, input_ids, probs, symbols):
        #Accounting for a batch of one element:
        input_ids = input_ids[0]
        probs = probs[0]

        symbols_probabilities = {}
        for symbol in symbols:
            #tokenizer.encode = tokenizer.tokenize + tokenizer.convert_tokens_to_ids
            tokens = self.tokenizer.tokenize(str(symbol))
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            symbol_prob = probs[token_ids[0]]
            if len(token_ids) > 1: 
                input_ids_for_token = input_ids.clone().detach()
                # Extract probabilities for the specified word from the distribution of the next token
                for i,id in enumerate(token_ids[:-1]):
                    input_ids_for_token = torch.cat([input_ids_for_token, torch.tensor([id])])
                    with torch.no_grad():
                        output = self.model(input_ids_for_token.unsqueeze(0))
                        logits = output.logits[:, -1, :]
                        next_probs = torch.softmax(logits, dim=-1)[0]
                    symbol_prob *= next_probs[token_ids[i+1]]
            symbols_probabilities[symbol] = symbol_prob
        #symbols_probabilities = OrderedDict(symbols_probabilities)    
        return symbols_probabilities
            
        