from collections import OrderedDict
from src_batch.sample_batch_probabilistic_teacher import SampleBatchProbabilisticTeacher

from pythautomata.base_types.sequence import Sequence

class SampleBatchProbabilisticTeacherFixed(SampleBatchProbabilisticTeacher):
    """
    Esta clase hereda de SampleBatchProbabilisticTeacher y corrige el comportamiento
    para que se aproveche el batching real de GPT-2 (u otro modelo) en lugar de hacer
    llamadas de una secuencia por vez.
    """

    def last_token_weights(self, sequence, required_suffixes):
        """
        Versión que maneja un solo 'sequence'. Internamente llama al método
        batch con una lista de 1 elemento, y devuelve la primera respuesta.
        Esto sirve de compatibilidad si en algún lugar se llama uno a uno.
        """
        self._last_token_weight_queries_count += len(required_suffixes)
        results = self._target_model.get_last_token_weights_batch([sequence], required_suffixes)
        return results[0]

    def last_token_weights_batch(self, sequences, required_suffixes):
        """
        Versión real de batching: hace UNA llamada a 'get_last_token_weights_batch'
        del modelo subyacente, pasando la lista completa de secuencias.

        Este método evita la iteración sobre cada secuencia, usando la ventaja
        de PyTorch/Transformers para procesar todo de una sola vez.
        """
        # Contabilizamos la cantidad total de consultas: (num secuencias) * (num sufijos)
        self._last_token_weight_queries_count += len(sequences) * len(required_suffixes)

        # Llamada directa al modelo (ej: GPT2_probabilistic_model_wrapper_batch)
        return self._target_model.get_last_token_weights_batch(sequences, required_suffixes)

    @property
    def alphabet(self):
        return self._target_model.alphabet

    @property
    def terminal_symbol(self):
        return self._target_model.terminal_symbol

    def log_sequence_weight(self, sequence):
        return self._target_model.log_sequence_weight(sequence)
