from logging import INFO
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from attr.validators import deep_mapping, instance_of

from adam.learner.surface_templates import SurfaceTemplateVariable
from immutablecollections import immutableset, ImmutableDict
from immutablecollections.converter_utils import _to_immutabledict
from networkx import DiGraph

from adam.ontology.ontology import Ontology
from adam.perception.perception_graph import (
    MatchedObjectNode,
    MatchedObjectPerceptionPredicate,
    PerceptionGraphPattern,
    GraphLogger,
)
from attr import attrib, attrs

# Constants used to map locations in a prepositional phrase for mapping
SLOT1 = SurfaceTemplateVariable("slot1")
SLOT2 = SurfaceTemplateVariable("slot2")

_PREPOSITION_TEMPLATE_VARIABLES = {SLOT1, SLOT2}


@attrs(frozen=True, slots=True, eq=False)
class PerceptionGraphTemplate:
    graph_pattern: PerceptionGraphPattern = attrib(
        validator=instance_of(PerceptionGraphPattern), kw_only=True
    )
    template_variable_to_pattern_node: ImmutableDict[
        SurfaceTemplateVariable, MatchedObjectPerceptionPredicate
    ] = attrib(
        converter=_to_immutabledict,
        kw_only=True,
        validator=deep_mapping(
            instance_of(SurfaceTemplateVariable),
            instance_of(MatchedObjectPerceptionPredicate),
        ),
    )
    pattern_node_to_template_variable: ImmutableDict[
        MatchedObjectPerceptionPredicate, SurfaceTemplateVariable
    ] = attrib(init=False)

    @staticmethod
    def from_graph(
        perception_graph: DiGraph,
        template_variable_to_matched_object_node: Iterable[
            Tuple[SurfaceTemplateVariable, MatchedObjectNode]
        ],
    ) -> "PerceptionGraphTemplate":
        template_variable_to_matched_object_node = immutableset(
            template_variable_to_matched_object_node
        )
        if len(template_variable_to_matched_object_node) != 2:
            raise RuntimeError(
                f"Expected only two object variables in a preposition graph. Found "
                f"{len(template_variable_to_matched_object_node)} in {template_variable_to_matched_object_node}"
            )
        pattern_from_graph = PerceptionGraphPattern.from_graph(perception_graph)
        pattern_graph = pattern_from_graph.perception_graph_pattern
        matched_object_to_matched_predicate = (
            pattern_from_graph.perception_graph_node_to_pattern_node
        )

        template_variable_to_pattern_node: List[Any] = []

        for template_variable, object_node in template_variable_to_matched_object_node:
            if object_node in matched_object_to_matched_predicate:
                template_variable_to_pattern_node.append(
                    (template_variable, matched_object_to_matched_predicate[object_node])
                )

        return PerceptionGraphTemplate(
            graph_pattern=pattern_graph,
            template_variable_to_pattern_node=template_variable_to_pattern_node,
        )

    def __attrs_post_init__(self) -> None:
        object_predicate_nodes = set(self.template_variable_to_pattern_node.values())

        for object_node in object_predicate_nodes:
            if (
                object_node
                not in self.graph_pattern._graph.nodes  # pylint:disable=protected-access
            ):
                raise RuntimeError(
                    f"Expected mapping which contained graph nodes"
                    f" but got {object_node} with id {id(object_node)}"
                    f" which doesn't exist in {self.graph_pattern}"
                )

        template_variables = set(self.template_variable_to_pattern_node.keys())
        if template_variables != _PREPOSITION_TEMPLATE_VARIABLES:
            raise RuntimeError(
                f"Expected a preposition pattern to have "
                f"the object variables {_PREPOSITION_TEMPLATE_VARIABLES} "
                f"but got {template_variables}"
            )

    def intersection(
        self,
        pattern: "PerceptionGraphTemplate",
        *,
        graph_logger: Optional[GraphLogger] = None,
        ontology: Ontology,
    ) -> Optional["PerceptionGraphTemplate"]:
        intersected_pattern = self.graph_pattern.intersection(
            pattern.graph_pattern, graph_logger=graph_logger, ontology=ontology
        )
        if intersected_pattern:
            if graph_logger:
                graph_logger.log_graph(intersected_pattern, INFO, "Intersected pattern")
            mapping_builder = []
            items_to_iterate: List[
                Tuple[SurfaceTemplateVariable, MatchedObjectPerceptionPredicate]
            ] = []
            items_to_iterate.extend(self.template_variable_to_pattern_node.items())
            items_to_iterate.extend(pattern.template_variable_to_pattern_node.items())
            for template_variable, pattern_node in immutableset(items_to_iterate):
                if (
                    pattern_node
                    in intersected_pattern._graph.nodes  # pylint:disable=protected-access
                ):
                    mapping_builder.append((template_variable, pattern_node))

            return PerceptionGraphTemplate(
                graph_pattern=intersected_pattern,
                template_variable_to_pattern_node=mapping_builder,
            )
        else:
            return None

    @pattern_node_to_template_variable.default
    def _init_pattern_node_to_template_variable(
        self
    ) -> ImmutableDict[MatchedObjectPerceptionPredicate, SurfaceTemplateVariable]:
        return self.template_variable_to_pattern_node.inverse()
