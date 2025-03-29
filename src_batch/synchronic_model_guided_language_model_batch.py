import numpy as np

from pythautomata.abstract.probabilistic_model import ProbabilisticModel
from pythautomata.base_types.symbol import SymbolStr, Symbol
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.alphabet import Alphabet

class SynchronicModelGuidedLanguageModelBatch(ProbabilisticModel):

    def __init__(self, model:ProbabilisticModel, guiding_model:ProbabilisticModel, max_seq_length: int = None,  model_name: str = None, normalize_outputs = False, top_k = None, check_is_defined = False, undefined_output = None):
        super().__init__()
        self._model = model
        self._guiding_model = guiding_model
        self._normalize_outputs = normalize_outputs
        self._top_k = top_k
        self.check_is_defined = check_is_defined
        self.undefined_output = undefined_output

        if model_name is None:
            self._model_name = model.name + ("_" + guiding_model.name if guiding_model else "")
        else:
            self._model_name = model_name

        if max_seq_length is None:
            if guiding_model is not None:
                self._max_seq_length = min(model._max_seq_length, guiding_model._max_seq_length)
            else:
                self._max_seq_length = model._max_seq_length

        if guiding_model is not None:
            assert model.alphabet == guiding_model.alphabet
            assert model.terminal_symbol == guiding_model.terminal_symbol

        self._alphabet = model.alphabet
        self._terminal_symbol = model.terminal_symbol
        self.query_cache = dict()

        if self.check_is_defined:
            assert undefined_output is not None, "There should be an undefined output specified if checking if the sequence is defined"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def terminal_symbol(self) -> Symbol:
        return self._terminal_symbol

    @property
    def name(self) -> str:
        return self.model_name

    def sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def log_sequence_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def last_token_probability(self, sequence: Sequence) -> float:
        raise NotImplementedError

    def _compose_probas(self, p1, p2):
        assert len(p1) == len(p2)
        return list(np.array(p1) * np.array(p2))

    def normalize(self, probas):
        total = np.sum(probas)
        return list(np.array(probas) / total) if total > 0 else probas

    def _mask_elements_below_topk(self, arr, topk):
        sorted_indices = np.argsort(arr, axis=None)
        mask = np.zeros_like(arr, dtype=bool)
        topk_indices = np.unravel_index(sorted_indices[-topk:], arr.shape)
        mask[topk_indices] = True
        return np.where(mask, arr, 0)

    def _raw_last_token_weights(self, sequence, required_suffixes):
        if self._guiding_model is not None:
            guiding_results = self._guiding_model.get_last_token_weights(sequence, required_suffixes)
            projected = [required_suffixes[i] for i in range(len(required_suffixes)) if guiding_results[i]]
        else:
            guiding_results = [1] * len(required_suffixes)
            projected = required_suffixes

        model_results = self._model.get_last_token_weights(sequence, projected)
        full = []
        j = 0
        for i in range(len(required_suffixes)):
            if guiding_results[i]:
                full.append(model_results[j])
                j += 1
            else:
                full.append(0)

        final = self._compose_probas(full, guiding_results)
        if self._top_k is not None:
            final = self._mask_elements_below_topk(final, self._top_k)
        if self._normalize_outputs:
            final = self.normalize(final)
        print(f"[WRAPPER NON-BATCH CALL] Sequence: {sequence} | Allowed: {sum(guiding_results)}")
        return final

    def _raw_last_token_weights_batch(self, sequences, required_suffixes):
        return self.get_last_token_weights_batch(sequences, required_suffixes)

    def get_last_token_weights(self, sequence, required_suffixes):
        if self.check_is_defined and not self.check_sequence_is_defined(sequence):
            return self.undefined_output
        return self._raw_last_token_weights(sequence, required_suffixes)

    def get_last_token_weights_batch(self, sequences, required_suffixes):
        if self.check_is_defined:
            is_defined_mask = self.check_sequences_are_defined_batch(sequences)
            sequences_to_process = [seq for seq, valid in zip(sequences, is_defined_mask) if valid]
        else:
            is_defined_mask = [True] * len(sequences)
            sequences_to_process = sequences

        guiding_outputs = []
        projected_suffixes_batch = []
        for sequence in sequences_to_process:
            if self._guiding_model is not None:
                guiding_result = self._guiding_model.get_last_token_weights(sequence, required_suffixes)
                projected_suffixes = [required_suffixes[i] for i in range(len(required_suffixes)) if guiding_result[i]]
            else:
                guiding_result = [1] * len(required_suffixes)
                projected_suffixes = required_suffixes

            guiding_outputs.append(guiding_result)
            projected_suffixes_batch.append(projected_suffixes)

        # Agrupar por sufijos
        groups = {}
        indices = {}
        for i, (seq, sufs) in enumerate(zip(sequences_to_process, projected_suffixes_batch)):
            key = tuple(sufs)
            if key not in groups:
                groups[key] = []
                indices[key] = []
            groups[key].append(seq)
            indices[key].append(i)

        model_results = [None] * len(sequences_to_process)
        for suffixes, seqs in groups.items():
            print(f"[BATCH DEBUG] group with suffixes={len(suffixes)} → {len(seqs)} sequences")
            if len(suffixes) == 0:
                for idx in indices[suffixes]:
                    model_results[idx] = []
                continue
            batch_result = self._model.get_last_token_weights_batch(seqs, list(suffixes))
            for res, idx in zip(batch_result, indices[suffixes]):
                model_results[idx] = res

        final_results = []
        j = 0
        for i in range(len(sequences)):
            if not is_defined_mask[i]:
                final_results.append(self.undefined_output)
            else:
                g_out = guiding_outputs[j]
                model_out = model_results[j]
                if model_out is None or len(model_out) == 0:
                    final_results.append([0] * len(required_suffixes))
                else:
                    full = []
                    idx = 0
                    for k in range(len(required_suffixes)):
                        full.append(model_out[idx] if g_out[k] else 0)
                        idx += int(g_out[k])
                    composed = self._compose_probas(full, g_out)
                    if self._top_k is not None:
                        composed = self._mask_elements_below_topk(composed, self._top_k)
                    if self._normalize_outputs:
                        composed = self.normalize(composed)
                    final_results.append(composed)
                j += 1

        return final_results


    def last_token_probabilities_batch(self, sequences, required_suffixes):
        return self.get_last_token_weights_batch(sequences, required_suffixes)

    def check_sequence_is_defined(self, sequence):
        symbols = [self.terminal_symbol] + sorted(self.alphabet.symbols)
        for i in range(len(sequence)):
            probs = self._raw_last_token_weights(Sequence(sequence.value[:i]), symbols)
            idx = symbols.index(sequence.value[i])
            if probs[idx] == 0:
                return False
        return True

    def check_sequences_are_defined_batch(self, sequences):
        symbols = [self.terminal_symbol] + sorted(self.alphabet.symbols)
        prefix_batches = []
        indices = []
        for i, seq in enumerate(sequences):
            for j in range(len(seq)):
                prefix_batches.append(Sequence(seq.value[:j]))
                indices.append((i, seq.value[j]))

        results = self._raw_last_token_weights_batch(prefix_batches, symbols)
        defined = [True] * len(sequences)
        for k, (seq_idx, symbol) in enumerate(indices):
            symbol_idx = symbols.index(symbol)
            if results[k][symbol_idx] == 0:
                defined[seq_idx] = False
        return defined
