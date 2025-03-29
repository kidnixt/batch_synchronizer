from pythautomata.base_types.sequence import Sequence
from pythautomata.automata.wheighted_automaton_definition.weighted_automaton import WeightedAutomaton
from pymodelextractor.teachers.sample_batch_probabilistic_teacher import SampleBatchProbabilisticTeacher
from pythautomata.abstract.finite_automaton import FiniteAutomataComparator
from src_batch.sample_batch_probabilistic_teacher_fixed import SampleBatchProbabilisticTeacherFixed
from pythautomata.base_types.alphabet import Alphabet
from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import SymbolStr, Symbol



from typing import Union, Sized
import numpy as np

class HypothesisAwareSampleProbabilisticTeacherBatch(SampleBatchProbabilisticTeacherFixed):
    def __init__(self, model, comparator: FiniteAutomataComparator, sample_size: float = None,
                 max_seq_length: int = 128, parallel_cache=False, max_query_elements=1_000_000,
                 batch_size=10_000, cache_from_dataloader=None):
        super().__init__(model, comparator, sample_size, None, max_seq_length, False,
                         parallel_cache, max_query_elements, batch_size, cache_from_dataloader)
        self._max_seq_length = max_seq_length
        self.batch_size = batch_size

        # Debug log
        self._debug_log_path = "teacher_debug_log.txt"
        with open(self._debug_log_path, "w") as f:
            f.write("DEBUG LOG - HypothesisAwareSampleProbabilisticTeacherBatch (Optimized)\n")

    def equivalence_query(self, aut: WeightedAutomaton) -> Union[tuple[bool, Sized], tuple[bool, None]]:
        self._equivalence_queries_count += 1
        tried = set()
        suffixes = [self.terminal_symbol] + [Sequence((s,)) for s in self.alphabet.symbols]

        for batch in self.generate_batch_words():
            batch = batch[0]
            word_to_prefixes = {word: sorted(word.get_prefixes(), key=len) for word in batch}

            # Recolectar todos los prefixes no probados
            all_prefixes = []
            prefix_to_word_index = {}
            for idx, word in enumerate(batch):
                for prefix in word_to_prefixes[word]:
                    if prefix not in tried:
                        prefix_to_word_index[prefix] = idx
                        all_prefixes.append(prefix)

            # Evaluar en batch
            model_results = self.last_token_weights_batch(all_prefixes, suffixes)

            for prefix, result in zip(all_prefixes, model_results):
                if isinstance(result, (list, np.ndarray)) and all(p == 0 for p in result):
                    with open(self._debug_log_path, "a") as f:
                        f.write(f"[SKIPPED - ZERO PROB] prefix: {prefix}, obs1: {result}\n")
                    tried.add(prefix)
                    continue

                obs2 = aut.get_last_token_weights(prefix, suffixes)
                with open(self._debug_log_path, "a") as f:
                    f.write(f"prefix: {prefix}, obs1: {result}, obs2: {obs2}\n")

                if not self._comparator.next_tokens_equivalent_output(result, obs2):
                    return False, prefix

                tried.add(prefix)

        return True, None

    def generate_batch_words(self):
        if self._full_prefix_set:
            pre_computed_rand_words = len(self.__rand_words)
            pre_computed_rand_words_index = 0
            while not self._all_rand_words_precomputed or pre_computed_rand_words_index < pre_computed_rand_words:
                batch = []
                for _ in range(self.batch_size):
                    if pre_computed_rand_words > 0 and pre_computed_rand_words_index < pre_computed_rand_words:
                        batch.append(self.__rand_words[pre_computed_rand_words_index])
                        pre_computed_rand_words_index += 1
                    else:
                        if self._all_rand_words_precomputed:
                            break
                        next_element = next(self._rand_words_generator)
                        if len(next_element) > self._sequence_generator._max_seq_length:
                            self._all_rand_words_precomputed = True
                            break
                        self.__rand_words.append(next_element)
                        batch.append(next_element)
                yield [batch]
        else:
            total = 0
            while total < self._sample_size:
                to_go = self._sample_size - total
                words_to_generate = min(self.batch_size, to_go)
                rand_words = sorted(self._sequence_generator.generate_words(words_to_generate))
                total += words_to_generate
                yield [rand_words]
                

