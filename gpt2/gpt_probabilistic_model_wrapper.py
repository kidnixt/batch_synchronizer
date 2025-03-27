from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet
from collections import OrderedDict
import torch

class GPT2_probabilistic_model_wrapper(ProbabilisticModel):
    
    def __init__(self, max_seq_length: int, alphabet:Alphabet, device: str, model, tokenizer, prompt:Sequence = Sequence()):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.device = device
        self._alphabet = alphabet        
        self._prompt = prompt

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
        weights = list()
        alphabet_symbols_weights = self.last_token_probability(sequence, required_suffixes)
        for suffix in required_suffixes:
            assert suffix in alphabet_symbols_weights
        return [alphabet_symbols_weights[suffix] for suffix in required_suffixes]
    
    def get_last_token_weights_batch(self, sequences, required_suffixes):
        results = []
        for seq in sequences:
            results.append(self.get_last_token_weights(seq, required_suffixes))
        return results
    
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
            
        