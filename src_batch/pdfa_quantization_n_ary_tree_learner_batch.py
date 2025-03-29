from pythautomata.base_types.sequence import Sequence
from pythautomata.base_types.symbol import Symbol
from pythautomata.automata.wheighted_automaton_definition.weighted_state import WeightedState
from pythautomata.model_comparators.wfa_tolerance_comparison_strategy import WFAToleranceComparator
from pythautomata.utilities import pdfa_utils
from pythautomata.utilities.probability_partitioner import ProbabilityPartitioner
from pymodelextractor.teachers.probabilistic_teacher import ProbabilisticTeacher
from pythautomata.automata.wheighted_automaton_definition.probabilistic_deterministic_finite_automaton import \
    ProbabilisticDeterministicFiniteAutomaton as PDFA, ProbabilisticDeterministicFiniteAutomaton
from pymodelextractor.learners.observation_table_learners.observation_table import epsilon
from pymodelextractor.learners.learning_result import LearningResult
from pymodelextractor.exceptions.query_length_exceeded_exception import QueryLengthExceededException
from pymodelextractor.exceptions.number_of_states_exceeded_exception import NumberOfStatesExceededException
from collections import OrderedDict
import math
import warnings


class PDFAQuantizationNAryTreeLearnerBatch:
    def __init__(self, probabilityPartitioner: ProbabilityPartitioner, pre_cache_queries_for_building_hipothesis = False, check_probabilistic_hipothesis = True, exhaust_counterexample = False, omit_zero_transitions = False):
        self.probability_partitioner = probabilityPartitioner
        self._pre_cache_queries_for_building_hipothesis = pre_cache_queries_for_building_hipothesis
        self._verbose = False
        self._tree = None
        self._check_probabilistic_hipothesis = check_probabilistic_hipothesis
        self._exhaust_counterexample = exhaust_counterexample
        self._omit_zero_transitions = omit_zero_transitions
        pass

    @property
    def _alphabet(self):
        return self._teacher.alphabet

    @property
    def _symbols(self):
        return self._teacher.alphabet.symbols

    @property
    def _all_symbols_sorted(self):
        symbols = sorted(list(self._alphabet.symbols))
        symbols = [self.terminal_symbol] + symbols
        return symbols

    def _perform_equivalence_query(self, model):
        return self._teacher.equivalence_query(model)

    def _perform_next_token_probabilities(self, value):
        return self._teacher.next_token_probabilities(value)
    
    def _is_counterexample(self, sequence, hypothesis):
        teacher_probs = list(self._perform_next_token_probabilities(sequence).values())
        hypothesis_probs = hypothesis.last_token_probabilities(sequence, self._all_symbols_sorted)
        return teacher_probs!=hypothesis_probs

    def _shorten_counterexample(self, counterexample, hypothesis):
        for prefix in counterexample.get_prefixes():
            if self._is_counterexample(prefix, hypothesis):
                return prefix

    def initialization(self, verbose) -> tuple[bool, ProbabilisticDeterministicFiniteAutomaton]:
        probabilities = self._perform_next_token_probabilities(epsilon)
        starting_pdfa = self.create_single_state_PDFA(probabilities)
        are_equivalent, counterexample = self._perform_equivalence_query(starting_pdfa)
        if are_equivalent:
            self._tree = None
            return True, starting_pdfa

        if self._omit_zero_transitions:
            counterexample = self._shorten_counterexample(counterexample, starting_pdfa)

        next_token_probabilities_epsilon = self._perform_next_token_probabilities(epsilon)
        next_token_probabilities_counterexample = self._perform_next_token_probabilities(counterexample)
        nodeRoot = ClassificationNode(epsilon)
        nodeEpsilon = ClassificationNode(epsilon, parent=nodeRoot, probabilities=next_token_probabilities_epsilon)
        nodeCounterexample = ClassificationNode(counterexample, parent=nodeRoot,
                                                probabilities=next_token_probabilities_counterexample)

        nodeRoot.childs[tuple(next_token_probabilities_epsilon.values())] = nodeEpsilon
        nodeRoot.childs[tuple(next_token_probabilities_counterexample.values())] = nodeCounterexample

        self._tree = ClassificationTree(nodeRoot, self._teacher, self.probability_partitioner, verbose=verbose)
        return False, starting_pdfa

    def learn(self, teacher: ProbabilisticTeacher, verbose: bool = False) -> LearningResult:
        self._verbose = verbose        
        if self._pre_cache_queries_for_building_hipothesis:
            assert hasattr(teacher, 'next_token_probabilities_batch')

        self.terminal_symbol = teacher.terminal_symbol
        self._teacher = teacher
        models = []
        if verbose: print('Starting learning process')
        is_target_DFA, model = self.initialization(verbose)
        symbols = list(self._alphabet.symbols)
        epsilon_probability = self._teacher.next_token_probabilities(Sequence())
        if not is_target_DFA:
            batch = []
            for symbol in symbols:
                if epsilon_probability[symbol] != 0 or not self._omit_zero_transitions:
                    batch.append(Sequence([symbol]))

            # 2) Llamamos a sift_batch (una sola vez), en lugar de sift(...) en bucle:
            if len(batch) > 0:
                self._tree.sift_batch(batch)  

            # Luego seguimos igual:
            model = self.tentative_hypothesis()
            models.append(model)
            last_size = len(model.weighted_states)
            if verbose: print('Running EQ')
            are_equivalent, counterexample = self._perform_equivalence_query(model)
            while not are_equivalent:
                if verbose: print('Size before update:', last_size)                
                self.update_tree(counterexample, model)
                model = self.tentative_hypothesis()
                models.append(model)
                size = len(model.weighted_states)
                if verbose: print('Size after update:', size)
                if size == last_size:
                    warnings.warn('Possible infinite loop, last hipothesis has the same size as current hipothesis.\nSize: '+str(size))
                last_size = size
                
                teacher_prob = list(self._teacher.next_token_probabilities(counterexample).values())                
                model_prob = model.last_token_probabilities(counterexample, self._all_symbols_sorted)
                ce_is_correct = self.probability_partitioner.are_in_same_partition(teacher_prob, model_prob)
                
                while self._exhaust_counterexample and not ce_is_correct:
                    self.update_tree(counterexample, model)
                    model = self.tentative_hypothesis()
                    models.append(model)
                    model_prob = model.last_token_probabilities(counterexample, self._all_symbols_sorted)
                    ce_is_correct = self.probability_partitioner.are_in_same_partition(teacher_prob, model_prob)
                if verbose: print('Running EQ')
                are_equivalent, counterexample = self._perform_equivalence_query(model)
        if verbose: print('Learning process finished')
        result = self._learning_results_for(model)
        return result

    def _learning_results_for(self, model, rename_states = False):
        numberOfStates = len(model.weighted_states) if model is not None else 0
        if rename_states:
            for count, state in enumerate(model.weighted_states):
                state.name = 'q' + str(count)

        info = {
            'equivalence_queries_count': self._teacher.equivalence_queries_count,
            'last_token_weight_queries_count': self._teacher.last_token_weight_queries_count,
            'observation_tree': self._tree
        }
        return LearningResult(model, numberOfStates, info)

    def tentative_hypothesis(self) -> PDFA:        
        states = {}
        symbols = list(self._alphabet.symbols)
        symbols.sort()
        #updated_tree = True
        #while updated_tree:
        if self._pre_cache_queries_for_building_hipothesis:
            self._tree.cache_queries_for_building_hipothesis()

        for leaf_str, leaf in self._tree.leaves.items():
            initial_weight = 1 if leaf_str == epsilon else 0
            terminal_symbol_probability = leaf.probabilities[self.terminal_symbol]
            state = WeightedState(leaf_str, initial_weight, terminal_symbol_probability, terminal_symbol=self.terminal_symbol)
            states[leaf_str] = state
        
        visited_states = set()
        states_to_visit = []
        states_to_visit.append(epsilon)
        while states_to_visit:
            access_string = states_to_visit.pop()
            visited_states.add(access_string)

            # 1) Reunir todas las secuencias (access_string + symbol) en esta ronda
            expansions = []
            expansions_symbols = []
            for symbol in symbols:
                if self._tree.leaves[access_string].probabilities[symbol] > 0 or not self._omit_zero_transitions:
                    expansions.append(access_string + symbol)
                    expansions_symbols.append(symbol)

            # 2) Llamamos a sift_batch en vez de un bucle con sift(...)
            results_batch = self._tree.sift_batch(expansions)  
            # results_batch[i] será una tupla: (access_string_of_transition, updated_tree)

            # 3) Recorremos los resultados y actualizamos
            for (seq_expanded, symbol), (access_string_of_transition, updated_tree) in zip(
                    zip(expansions, expansions_symbols), results_batch):

                # Creamos el estado si es nuevo
                if updated_tree:  # Indica que se agregó un leaf nuevo
                    new_leaf = self._tree.leaves[access_string_of_transition]
                    terminal_symbol_probability = new_leaf.probabilities[self.terminal_symbol]
                    new_state = WeightedState(
                        access_string_of_transition, 0, terminal_symbol_probability,
                        terminal_symbol=self.terminal_symbol
                    )
                    states[access_string_of_transition] = new_state

                # Añadimos la transición
                states[access_string].add_transition(
                    symbol, states[access_string_of_transition],
                    self._tree.leaves[access_string].probabilities[symbol]
                )

                # Si aún no visitamos ese estado, lo programamos para la siguiente ronda
                if access_string_of_transition not in visited_states:
                    states_to_visit.append(access_string_of_transition)

        if self._omit_zero_transitions:
            hole = WeightedState("HOLE", 0, 1, terminal_symbol=self.terminal_symbol)
            for symbol in symbols:
                hole.add_transition(symbol, hole, 0)
            added_transitions = 0
            for access_string, state in states.items():
                for symbol in symbols:
                    if symbol not in state.transitions_set:
                        state.add_transition(symbol, hole, 0)
                        added_transitions+=1                        
            if added_transitions > 0:
                states[hole.name] = hole
        
        for state in list(states.keys()).copy():
            if state not in visited_states and states[state].initial_weight != 1:
                del states[state]

        comparator = WFAToleranceComparator()
        states = set(states.values())
        return PDFA(self._alphabet, states, self.terminal_symbol, comparator=comparator, check_is_probabilistic=self._check_probabilistic_hipothesis)

    def get_accessing_string(self, model: PDFA, sequence: Sequence):
        state = model.get_first_state()
        if sequence == epsilon:
            return state.name

        while len(sequence) > 0:
            state = list(state.next_states_for(sequence[0]))[0]
            sequence = sequence[1:]
        return state.name

    def update_tree(self, counterexample: Sequence, model: PDFA) -> None:
        prefixes = list(counterexample.get_prefixes())

        # 1) Llamada en batch a 'sift'
        batch_results = self._tree.sift_batch(prefixes)
        # Cada elemento será (s_i, updated), en el mismo orden que 'prefixes'

        s_i = epsilon
        gamma_j_minus_1 = epsilon

        for prefix, (sift_result, _) in zip(prefixes, batch_results):
            s_i_minus_1 = s_i
            s_i = sift_result
            s_hat_i = self.get_accessing_string(model, prefix)

            if s_i != s_hat_i:
                internal_node_string = prefix[-1] + self._tree.lca(s_i, s_hat_i)
                self._tree.update_node(s_i_minus_1, gamma_j_minus_1, internal_node_string)
                break
            gamma_j_minus_1 = prefix

    def create_single_state_PDFA(self, probabilities: OrderedDict[Symbol, float]):
        final_weight = probabilities[self.terminal_symbol]
        probabilities.pop(self.terminal_symbol)
        initialState = WeightedState(epsilon, 1, final_weight=final_weight, terminal_symbol=self.terminal_symbol)
        states = {initialState}
        hole = None
        if self._omit_zero_transitions:
            hole = WeightedState("HOLE", 0, 1, terminal_symbol=self.terminal_symbol)
            for symbol in probabilities.keys():
                hole.add_transition(symbol, hole, 0)
            states.add(hole)            
        for symbol, probability in probabilities.items():
            if probability > 0 or not self._omit_zero_transitions:
                initialState.add_transition(symbol, initialState, probability)
            else:
                
                initialState.add_transition(symbol, hole, probability)            
        return PDFA(self._alphabet, states, self.terminal_symbol, comparator=WFAToleranceComparator(), check_is_probabilistic=self._check_probabilistic_hipothesis)


class ClassificationTree:
    unknown_leaf = "UNKNOWN"

    def __init__(self, root: 'ClassificationNode', teacher: ProbabilisticTeacher, probability_partitioner: ProbabilityPartitioner,
                 max_query_length: int = math.inf, verbose=False, max_states: int = math.inf,check_max_states_in_tree = False):
        self.leaves = dict()
        self._teacher = teacher
        self.root = root
        self.probability_partitioner = probability_partitioner
        self.inner_nodes = dict()
        self._add_leaves_and_inner_nodes()
        self._equivalence_dict = dict()
        self._next_token_probabilities_cache = dict()
        self._partitions_cache = dict()
        self._sift_cache = dict()
        self._max_query_length = max_query_length        
        self._verbose = verbose
        self._check_max_states_in_tree = check_max_states_in_tree
        self._max_states = max_states
        

    @property
    def depth(self) -> int:
        return max([x.depth for x in self.leaves.values()])

    def _add_leaves_and_inner_nodes(self):
        q = [self.root]        
        while q:
            node = q.pop()
            if node.is_leaf():
                self.leaves.update({node.string: node})
            else:
                self.inner_nodes.update({node.string: node})
                for child in node.childs.values():
                    q.append(child)

    def cache_queries_for_building_hipothesis(self):
        symbols = list(self._teacher.alphabet.symbols)
        symbols.sort()
        queries = set()
        for access_string in self.leaves.keys():
            for symbol in symbols:
                for distinguishing_string in self.inner_nodes:
                    query = access_string + symbol + distinguishing_string
                    if query not in self._next_token_probabilities_cache:
                        queries.add(query)
        if len(queries)>0:
            results = self._teacher.next_token_probabilities_batch(queries)
            self._next_token_probabilities_cache.update(results)
            
    def sift_batch(self, sequences: list[Sequence], update=True) -> list[tuple[Sequence, bool]]:
        """
        Procesa todas las 'sequences' simultáneamente en modo batch.
        Devuelve una lista de (access_string, updated_flag) alineada
        con el orden de 'sequences'.

        - 'access_string' es el "string" donde cae cada secuencia en el árbol,
          o ClassificationTree.unknown_leaf si no se pudo clasificar.
        - 'updated_flag' indica si el árbol se ha modificado para esa secuencia.
        """

        # Resultado final (uno por secuencia)
        results = [None] * len(sequences)

        # 1. Inicializamos una cola con las secuencias que *no* están en cache.
        #    Si alguna secuencia YA estaba en el _sift_cache, nos ahorramos procesarla.
        queue = []
        for i, seq in enumerate(sequences):
            if seq in self._sift_cache:
                # Ya tenemos clasificación cacheada
                results[i] = (self._sift_cache[seq], False)
            else:
                # Empezamos en la raíz, sin haber actualizado nada aún.
                queue.append((i, seq, self.root, False))

        # 2. Procesamos la cola en rondas, cada ronda agrupa ítems por el "node" en que se hallan.
        from collections import defaultdict
        while queue:
            # Agrupamos las secuencias (índice, secuencia, nodo, updated_flag) por nodo
            node2items = defaultdict(list)
            for item in queue:
                idx, seq, node, updated_flag = item
                node2items[node].append((idx, seq, updated_flag))

            # Vaciamos la cola y la reconstruimos en función de lo que pase
            queue = []

            # Recorremos cada nodo
            for node, triple_list in node2items.items():
                # Si el nodo es hoja, clasificamos directamente
                if node.is_leaf():
                    for (idx, seq, updated_flag) in triple_list:
                        self._sift_cache[seq] = node.string
                        results[idx] = (node.string, updated_flag)
                    continue

                # NO es hoja: toco descender al siguiente nivel
                #
                # Este "node.string" se comporta como 'd' en tu sift() actual.
                # Construimos las secuencias "sd = sequence + d" para TODAS las secuencias del grupo
                d = node.string  
                sds = [(idx, seq + d, seq, updated_flag) for (idx, seq, updated_flag) in triple_list]

                # Llamada *batch* para obtener las probabilidades de TODAS las "sd" del grupo
                # Haría falta un método `_next_token_probabilities_batch(...)`.
                # Por simplicidad, supón que existe y retorna una lista de diccionarios con las
                # probabilidades para cada "sd", en el mismo orden.
                sd_probs_batch = self._next_token_probabilities_batch([sd for (_, sd, _, _) in sds], update)

                # Para cada secuencia en el grupo, vemos si hay hijo o creamos uno nuevo
                for ((idx, sd, original_seq, was_updated), sd_probs) in zip(sds, sd_probs_batch):
                    # Convertimos las probabilidades en la lista que se usa en "_look_for_branch".
                    prob_list = list(sd_probs.values())
                    child_key = self._look_for_branch(node.childs, prob_list)
                    if child_key is not None:
                        # Bajamos al hijo existente
                        child_node = node.childs[tuple(child_key)]
                        # Agregamos de nuevo a la cola para la siguiente ronda
                        queue.append((idx, original_seq, child_node, was_updated))
                    else:
                        if update:
                            # Creamos nodo nuevo
                            # => Volvemos a consultar (en tu sift actual, se llama:
                            #    "node_probabilities = self._next_token_probabilities(sequence, update)")
                            #    para "sequence" a secas, no "sd".
                            node_prob = self._next_token_probabilities(original_seq, update)

                            new_node = ClassificationNode(original_seq, parent=node, probabilities=node_prob)
                            node.childs[tuple(prob_list)] = new_node
                            self.leaves.update({new_node.string: new_node})
                            updated_tree_flag = True

                            if self._check_max_states_in_tree and len(self.leaves) > self._max_states:
                                raise NumberOfStatesExceededException(
                                    f"The maximum number of states ({self._max_states}) has been exceeded. "
                                    f"Current number of leaves: {len(self.leaves)}."
                                )

                            queue.append((idx, original_seq, new_node, was_updated or updated_tree_flag))
                        else:
                            # Sin update => devolvemos "UNKNOWN"
                            results[idx] = (ClassificationTree.unknown_leaf, False)
                            # No seguimos bajando

        # 3. Terminamos: 'results' está alineado con 'sequences'
        return results


    def sift(self, sequence: Sequence, update = True) -> Sequence:
        if sequence in self._sift_cache:
                return self._sift_cache[sequence], False
        node = self.root
        updated_tree = False
        while not node.is_leaf():
            d = node.string
            sd = sequence + d
            sd_probabilities = self._next_token_probabilities(sd, update).values()
            child_key = self._look_for_branch(node.childs, list(sd_probabilities))
            if child_key is not None:
                node = node.childs[tuple(child_key)]
            else:
                if update:
                    node_probabilities = self._next_token_probabilities(sequence, update)
                    new_node = ClassificationNode(sequence, parent=node, probabilities=node_probabilities)
                    node.childs[tuple(sd_probabilities)] = new_node
                    self.leaves.update({new_node.string: new_node})
                    updated_tree = True
                    node = new_node
                    if self._check_max_states_in_tree:
                        if len(self.leaves) > self._max_states:
                            raise NumberOfStatesExceededException(
                                f"The maximum number of states ({self._max_states}) has been exceeded. "
                                f"Current number of leaves: {len(self.leaves)}."
                            )
                else:
                    return ClassificationTree.unknown_leaf, False
        self._sift_cache[sequence] = node.string
        return node.string, updated_tree

    def _look_for_branch(self, childs, probabilities):
        if tuple(probabilities) in childs:
            return probabilities
        for probs in childs.keys():
            probs = list(probs)
            if self.probability_partitioner.are_in_same_partition(probs, probabilities):
                return probs
        return None

    def _next_token_probabilities(self, sequence: Sequence, check_max_query_length = True):
        if check_max_query_length and len(sequence) > self._max_query_length:
            raise QueryLengthExceededException
        if sequence in self._next_token_probabilities_cache:
            return self._next_token_probabilities_cache[sequence]
        else:
            value = self._teacher.next_token_probabilities(sequence)
            self._next_token_probabilities_cache[sequence] = value
            return value
        
    def _next_token_probabilities_batch(self, sequences: list[Sequence], check_max_query_length=True):
        """
        Versión batch de _next_token_probabilities:
        - Revisa la cache (_next_token_probabilities_cache).
        - Para las secuencias faltantes, llama a 'self._teacher.next_token_probabilities_batch(...)'
        - Devuelve una lista de dicts en el mismo orden que 'sequences'.
        """
        # 1) Filtramos secuencias ya cacheadas
        missing = []
        missing_indices = []
        results = [None]*len(sequences)
        for i, seq in enumerate(sequences):
            if check_max_query_length and len(seq) > self._max_query_length:
                raise QueryLengthExceededException
            if seq in self._next_token_probabilities_cache:
                results[i] = self._next_token_probabilities_cache[seq]
            else:
                missing.append(seq)
                missing_indices.append(i)

        # 2) Para las que faltan, llamamos al teacher en batch
        if missing:
            # Este método existe en tu teacher: next_token_probabilities_batch
            # Devuelve un dict {secuencia -> {simbolo->prob,...}} o una lista
            # (depende de tu implementación).
            batch_result = self._teacher.next_token_probabilities_batch(missing)
            # Asignamos a 'results' en el orden correspondiente
            for seq, probs in batch_result.items():
                i = missing_indices[missing.index(seq)]
                results[i] = probs
                # Actualizamos cache
                self._next_token_probabilities_cache[seq] = probs

        return results


    def lca(self, a: Sequence, b: Sequence) -> Sequence:
        ''' lca: lowest common ancestor '''
        if not a in self.leaves:
            print('recorcholis batman')
        t1 = self.leaves[a]
        t2 = self.leaves[b]
        if t1.depth < t2.depth:
            t = t1
            t1 = t2
            t2 = t
        while t1.depth > t2.depth:
            t1 = t1.parent
        while not (t1 is t2):
            assert not ((t1 is self.root) or (t2 is self.root))
            t1 = t1.parent
            t2 = t2.parent
        return t1.string

    def get_leftmost_node(self):
        node = self.root
        while node.left is not None:
            node = node.left[0]
        return self.leaves[node.string]

    def update_node(self, node_to_be_replaced, leaf_1, distinguishing_string):
        old_node = self.leaves[node_to_be_replaced]
        old_string = old_node.string
        old_node.string = distinguishing_string
        self.inner_nodes.update({distinguishing_string: old_node})
        next_token_probabilities_node1 = self._next_token_probabilities(leaf_1)
        next_token_probabilities_node2 = self._next_token_probabilities(node_to_be_replaced)
        node_1 = ClassificationNode(leaf_1, parent=old_node, probabilities=next_token_probabilities_node1)
        node_2 = ClassificationNode(node_to_be_replaced, parent=old_node, probabilities=next_token_probabilities_node2)

        node1_cont = leaf_1 + distinguishing_string
        node1_cont_probabilities = self._next_token_probabilities(node1_cont)
        node2_cont = node_to_be_replaced + distinguishing_string
        node2_cont_probabilities = self._next_token_probabilities(node2_cont)

        old_node.childs[tuple(node1_cont_probabilities.values())] = node_1
        old_node.childs[tuple(node2_cont_probabilities.values())] = node_2
        if self._verbose:
            print("----update_node----")
            print('Old Node (new Leaf)', node_to_be_replaced)
            print('New Leaf', leaf_1)
            print(self.leaves.keys())
        self.leaves.update({
            leaf_1: node_1,
            node_to_be_replaced: node_2,
        })
        if self._verbose:
            print(self.leaves.keys())
            print("--------")       
        self._update_sift_cache(old_string)
        if self._check_max_states_in_tree:
            if len(self.leaves) > self._max_states:
                raise NumberOfStatesExceededException(
                    f"The maximum number of states ({self._max_states}) has been exceeded. "
                    f"Current number of leaves: {len(self.leaves)}."
                )

    def _update_sift_cache(self, old_string):
        keys_to_remove = []
        for seq, access_string in self._sift_cache.items():
            if access_string == old_string:
                keys_to_remove.append(seq)

        for seq in keys_to_remove:
            del self._sift_cache[seq] 

class ClassificationNode:
    def __init__(self, string: Sequence, parent: 'ClassificationNode' = None, probabilities=None):
        self.parent = parent
        self.childs = OrderedDict()
        self.string = string
        self.probabilities = probabilities
        self._depth = parent.depth + 1 if parent else 0

    @property
    def depth(self) -> int:
        return self._depth

    def is_leaf(self) -> bool:
        return len(self.childs) == 0