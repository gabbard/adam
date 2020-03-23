import logging
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Iterable, Sequence

from adam.learner.perception_graph_template import PerceptionGraphTemplate
from adam.learner.surface_templates import SurfaceTemplate
from adam.learner.template_learner import AbstractTemplateLearner
from immutablecollections import immutabledict

from adam.language import TokenSequenceLinguisticDescription
from adam.ontology.ontology import Ontology
from adam.perception.perception_graph import DebugCallableType, LanguageAlignedPerception
from attr import Factory, attrib, attrs
from attr.validators import instance_of


@attrs
class AbstractSubsetLearner(AbstractTemplateLearner, ABC):
    _surface_template_to_hypothesis: Dict[
        SurfaceTemplate, PerceptionGraphTemplate
    ] = attrib(init=False, default=Factory(dict))
    _ontology: Ontology = attrib(validator=instance_of(Ontology), kw_only=True)
    _debug_callback: Optional[DebugCallableType] = attrib(default=None, kw_only=True)

    def _learning_step(
        self,
        preprocessed_input: LanguageAlignedPerception,
        surface_template: SurfaceTemplate,
    ) -> None:
        if surface_template in self._surface_template_to_hypothesis:
            # If already observed, get the largest matching subgraph of the pattern in the
            # current observation and
            # previous pattern hypothesis
            # TODO: We should relax this requirement for learning: issue #361
            previous_pattern_hypothesis = self._surface_template_to_hypothesis[
                surface_template
            ]

            updated_hypothesis = previous_pattern_hypothesis.intersection(
                self._hypothesis_from_perception(preprocessed_input),
                ontology=self._ontology,
            )

            if updated_hypothesis:
                # Update the leading hypothesis
                self._surface_template_to_hypothesis[
                    surface_template
                ] = updated_hypothesis
            else:
                logging.warning(
                    "Intersection of graphs had empty result; keeping original pattern"
                )

        else:
            # If it's a new description, learn a new hypothesis/pattern, generated as a pattern
            # graph frm the
            # perception graph.
            self._surface_template_to_hypothesis[
                surface_template
            ] = self._hypothesis_from_perception(preprocessed_input)

    @abstractmethod
    def _hypothesis_from_perception(
        self, preprocessed_input: LanguageAlignedPerception
    ) -> PerceptionGraphTemplate:
        pass

    def _primary_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        return (
            (surface_template, hypothesis, 1.0)
            for (
                surface_template,
                hypothesis,
            ) in self._surface_template_to_hypothesis.items()
        )

    def _fallback_templates(
        self
    ) -> Iterable[Tuple[SurfaceTemplate, PerceptionGraphTemplate, float]]:
        return tuple()

    def _post_process_descriptions(
        self,
        match_results: Sequence[
            Tuple[TokenSequenceLinguisticDescription, PerceptionGraphTemplate, float]
        ],
    ) -> Mapping[TokenSequenceLinguisticDescription, float]:
        if not match_results:
            return immutabledict()

        largest_pattern_num_nodes = max(
            len(template.graph_pattern) for (_, template, _) in match_results
        )

        return immutabledict(
            (description, len(template.graph_pattern) / largest_pattern_num_nodes)
            for (description, template, score) in match_results
        )


@attrs  # pylint:disable=abstract-method
class AbstractTemplateSubsetLearner(AbstractSubsetLearner, AbstractTemplateLearner, ABC):
    def log_hypotheses(self, log_output_path: Path) -> None:
        logging.info(
            "Logging %s hypotheses to %s",
            len(self._surface_template_to_hypothesis),
            log_output_path,
        )
        for (
            surface_template,
            hypothesis,
        ) in self._surface_template_to_hypothesis.items():
            template_string = surface_template.to_short_string()
            hypothesis.graph_pattern.render_to_file(
                template_string, log_output_path / template_string
            )
