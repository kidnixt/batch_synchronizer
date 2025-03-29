from typing import Tuple

from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton as PDFA, ProbabilisticDeterministicFiniteAutomaton
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pymodelextractor.learners.observation_tree_learners.pdfa_quantization_n_ary_tree_learner import \
     PDFAQuantizationNAryTreeLearner
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException
from pymodelextractor.utils.time_bound_utilities import timeout
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from src_batch.pdfa_quantization_n_ary_tree_learner_batch import PDFAQuantizationNAryTreeLearnerBatch
import numpy as np

class BoundedPDFAQuantizationNAryTreeLearnerBatch(PDFAQuantizationNAryTreeLearnerBatch):
    def __init__(self, partitioner, max_states, max_query_length, max_seconds_run=None, generate_partial_hipothesis = False, pre_cache_queries_for_building_hipothesis = False, check_probabilistic_hipothesis = True, exhaust_counterexample = False, mean_distribution_for_partial_hipothesis = False, omit_zero_transitions = False, check_max_states_in_tree = False):
        super().__init__(partitioner, pre_cache_queries_for_building_hipothesis, check_probabilistic_hipothesis, exhaust_counterexample, omit_zero_transitions)
        self._max_states = max_states
        self._max_query_length = max_query_length
        self._max_seconds_run = max_seconds_run
        self._exceeded_max_states = False
        self._exceeded_max_mq_length = False
        self._exceded_time_bound = False
        self._history = []        
        self._generate_partial_hipothesis = generate_partial_hipothesis
        self._compute_mean_distribution_for_partial_hipothesis = mean_distribution_for_partial_hipothesis
        self._check_max_states_in_tree = check_max_states_in_tree
        

    def _perform_equivalence_query(self, model):
        self._history.append(model)
        if len(model.weighted_states) > self._max_states:
            raise NumberOfStatesExceededException
        return super()._perform_equivalence_query(model)

    def run_learning_with_time_bound(self, teacher, verbose):
        try:
            with timeout(self._max_seconds_run):
                super().learn(teacher, verbose)
        except TimeoutError:
            if verbose: print("Time Bound Reached")
            self._exceded_time_bound = True

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        try:
            if self._max_seconds_run is not None:
                self.run_learning_with_time_bound(teacher, verbose)
            else:
                super().learn(teacher, verbose)
        except NumberOfStatesExceededException:
            if verbose: print("NumberOfStatesExceeded")
            self._exceeded_max_states = True
        except QueryLengthExceededException:
            if verbose: print("QueryLengthExceeded")
            self._exceeded_max_mq_length = True        

        if (self._exceeded_max_states and self._check_max_states_in_tree) or (not self._exceeded_max_states and self._generate_partial_hipothesis and (self._exceeded_max_mq_length or self._exceded_time_bound) and len(self._tree.leaves)>0):         
            partial_hipothesis = self.partial_hipothesis()
            self._history.append(partial_hipothesis)

        hist = list(self._history)
        result = self._learning_results_for(hist[-1] if len(hist) > 0 else None)
        result.info['NumberOfStatesExceeded'] = self._exceeded_max_states
        result.info['QueryLengthExceeded'] = self._exceeded_max_mq_length
        result.info['TimeExceeded'] = self._exceded_time_bound
        return result

    def initialization(self, verbose) -> tuple[bool, ProbabilisticDeterministicFiniteAutomaton]:
        try:
            ret = super().initialization(verbose)
            if not ret[0]:
                self._tree._max_query_length = self._max_query_length
                self._tree._max_states = self._max_states
                self._tree._check_max_states_in_tree = self._check_max_states_in_tree
            return ret
        except QueryLengthExceededException:
            print("QueryLengthExceeded")
            self._exceeded_max_mq_length = True
            return True, None

    def partial_hipothesis(self) -> PDFA:
        if self._pre_cache_queries_for_building_hipothesis:
            self._tree.cache_queries_for_building_hipothesis()

        states = {}
        symbols = list(self._alphabet.symbols)
        symbols.sort()

        # Construir un estado por cada leaf
        for leaf_str, leaf in self._tree.leaves.items():
            initial_weight = 1 if leaf_str == epsilon else 0
            terminal_symbol_probability = leaf.probabilities[self.terminal_symbol]
            state = WeightedState(leaf_str, initial_weight, terminal_symbol_probability,
                                  terminal_symbol=self.terminal_symbol)
            states[leaf_str] = state

        # Creamos un estado "UNKNOWN"
        unknown_state = WeightedState(self._tree.unknown_leaf, 0, -1, terminal_symbol=self.terminal_symbol)
        states[self._tree.unknown_leaf] = unknown_state

        partial_distributions = []
        accessed_states = set()
        states_to_visit = [epsilon]

        while states_to_visit:
            access_string = states_to_visit.pop()
            accessed_states.add(access_string)

            if access_string != self._tree.unknown_leaf:
                # 1) Reunir todas las secuencias (access_string + symbol) en esta ronda
                expansions = []
                expansions_symbols = []
                for symbol in symbols:
                    if self._tree.leaves[access_string].probabilities[symbol] > 0 or not self._omit_zero_transitions:
                        expansions.append(access_string + symbol)
                        expansions_symbols.append(symbol)

                # 2) Llamada en batch (update=False para no modificar el árbol)
                results_batch = self._tree.sift_batch(expansions, update=False)

                # 3) Recorremos cada secuencia expandida
                for (seq_expanded, symbol), (access_string_of_transition, _) in zip(
                        zip(expansions, expansions_symbols), results_batch):

                    # Agregamos transición
                    states[access_string].add_transition(
                        symbol,
                        states[access_string_of_transition] if access_string_of_transition in states
                        else unknown_state,  # En caso de que no exista, fallback al unknown
                        self._tree.leaves[access_string].probabilities[symbol]
                    )

                    # Si es la primera vez que llegamos a esa "access_string_of_transition"
                    if (access_string_of_transition not in accessed_states
                        and access_string_of_transition != self._tree.unknown_leaf):
                        states_to_visit.append(access_string_of_transition)

                    # Si llegó al unknown_leaf, recolectamos su distribución
                    if access_string_of_transition == self._tree.unknown_leaf:
                        partial_distributions.append(
                            self._tree._next_token_probabilities(seq_expanded, check_max_query_length=False)
                        )

        # Si quieres crear transiciones “promedio” en el unknown_leaf:
        if self._compute_mean_distribution_for_partial_hipothesis and len(partial_distributions) > 0:
            for symbol in symbols:
                mean_symbol_dist = np.mean([prob[symbol] for prob in partial_distributions])
                states[self._tree.unknown_leaf].add_transition(symbol, states[self._tree.unknown_leaf], mean_symbol_dist)
            states[self._tree.unknown_leaf].final_weight = np.mean(
                [prob[self.terminal_symbol] for prob in partial_distributions]
            )
        else:
            # Dejar -1 o 0 en todas las transiciones
            for symbol in symbols:
                states[self._tree.unknown_leaf].add_transition(symbol, states[self._tree.unknown_leaf], -1)

        # Manejo de HOLE si _omit_zero_transitions
        if self._omit_zero_transitions:
            hole = WeightedState("HOLE", 0, 1, terminal_symbol=self.terminal_symbol)
            for symbol in symbols:
                hole.add_transition(symbol, hole, 0)
            added_transitions = 0
            for s_name, st in states.items():
                for symbol in symbols:
                    if symbol not in st.transitions_set:
                        st.add_transition(symbol, hole, 0)
                        added_transitions += 1
            if added_transitions > 0:
                states[hole.name] = hole

        # Eliminar estados no accesibles
        for s_name in list(states.keys()):
            if s_name not in accessed_states and states[s_name].initial_weight != 1:
                del states[s_name]

        comparator = WFAToleranceComparator()
        states = set(states.values())
        return PDFA(self._alphabet, states, self.terminal_symbol, comparator=comparator, check_is_probabilistic=False)
