import random
from typing import Tuple, Sequence, Iterable

import pytest
from more_itertools import first

from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    INANIMATE_OBJECT,
    LIQUID,
    IS_BODY_PART,
    BIRD,
    TABLE,
    CHAIR,
    TRUCK,
    CAR,
    GROUND,
)
from adam.perception.perception_graph import PerceptionGraphPattern, NodePredicate
from tests.perception.perception_graph_test import (
    do_object_on_table_test,
    PatternTransform,
)

OBJECTS_TO_MATCH = GAILA_PHASE_1_ONTOLOGY.nodes_with_properties(
    INANIMATE_OBJECT, banned_properties={LIQUID, IS_BODY_PART}
)


def identity_pattern(pattern: PerceptionGraphPattern) -> PerceptionGraphPattern:
    return pattern


def randomly_ordered_pattern(pattern: PerceptionGraphPattern) -> PerceptionGraphPattern:
    def random_sorter(nodes: Iterable[NodePredicate]) -> Sequence[NodePredicate]:
        node_list = list(nodes)
        random.shuffle(node_list)
        return node_list

    return pattern.reordered_copy(random_sorter)


PATTERN_ORDERINGS_FUNCTIONS = {
    "identity order": identity_pattern,
    "random order": randomly_ordered_pattern,
}

# Trucks, cars, and chairs are known failures; and we use bird and table for testing
# Issue: https://github.com/isi-vista/adam/issues/399
def match_object(object_to_match, pattern_ordering_function: PatternTransform):
    if object_to_match not in [BIRD, TABLE, CHAIR, TRUCK, CAR, GROUND]:
        schemata = GAILA_PHASE_1_ONTOLOGY.structural_schemata(object_to_match)
        if len(schemata) == 1:
            assert do_object_on_table_test(
                object_to_match,
                first(schemata),
                BIRD,
                pattern_ordering_function=pattern_ordering_function,
            )


@pytest.mark.parametrize("object_to_match", OBJECTS_TO_MATCH)
def test_object_matching(object_to_match: OntologyNode, benchmark):
    benchmark.name = object_to_match.handle
    benchmark(match_object, object_to_match)


@pytest.mark.parametrize("object_to_match", OBJECTS_TO_MATCH)
@pytest.mark.parametrize("ordering", PATTERN_ORDERINGS_FUNCTIONS.items())
def test_node_order(
    object_to_match: OntologyNode, ordering: Tuple[str, PatternTransform], benchmark
):
    benchmark.name = object_to_match.handle
    (ordering_name, ordering_function) = ordering
    benchmark.group = ordering_name
    benchmark(match_object, object_to_match, ordering_function)
