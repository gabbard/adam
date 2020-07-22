import logging

from more_itertools import first
from typing import Tuple, Dict, Union, Optional

from attr.validators import instance_of

from attr import attrs, attrib

from adam.learner import (
    LanguageMode,
    LanguagePerceptionSemanticAlignment,
    PerceptionSemanticAlignment,
)
from adam.learner.template_learner import TemplateLearner
from adam.semantics import SyntaxSemanticsVariable, LearnerSemantics


# This class needs a better name?
@attrs(slots=True)
class StringFunctionCounter:
    _tokens_to_count: Dict[Tuple[str, ...], int] = attrib(init=False, default=dict())
    _num_instances_seen: int = attrib(init=False, default=0)

    def get_best_guess_token(self) -> Tuple[str, ...]:
        def sort_by_counts(tok_to_count: Tuple[Tuple[str, ...], int]) -> int:
            _, count = tok_to_count
            return count

        sorted_by_count = [(k, v) for k, v in self._tokens_to_count.items()]
        sorted_by_count.sort(key=sort_by_counts, reverse=True)
        toks, _ = first(sorted_by_count)
        # This should apply the tolerance principal to get the assumed token
        # If we don't know what the observation is
        # But for now we just return the highest seen argument
        return toks

    def add_example(self, tokens: Tuple[str, ...]) -> None:
        if tokens in self._tokens_to_count.keys():
            self._tokens_to_count[tokens] += 1
        else:
            self._tokens_to_count[tokens] = 1
        self._num_instances_seen += 1


@attrs
class FunctionalLearner:
    _observation_num: int = attrib(init=False, default=0)
    _language_mode: LanguageMode = attrib(validator=instance_of(LanguageMode))
    _concept_elements_to_arguments_to_function_counter: Dict[
        Tuple[Union[str, SyntaxSemanticsVariable], ...],
        Dict[SyntaxSemanticsVariable, StringFunctionCounter],
    ] = attrib(init=False)

    # HACK to deal with determiners
    def _remove_determiners(self, input: Tuple[str, ...]) -> Tuple[str, ...]:
        return tuple(tok for tok in input if tok not in ["a", "the"])

    def learn_from(
        self,
        language_perception_semantic_alignment: LanguagePerceptionSemanticAlignment,
        *,
        observation_num: int = -1,
        action_learner: TemplateLearner,
    ):
        if observation_num >= 0:
            logging.info(
                "Observation %s: %s",
                observation_num,
                language_perception_semantic_alignment.language_concept_alignment.language.as_token_string(),
            )
        else:
            logging.info(
                "Observation %s: %s",
                self._observation_num,
                language_perception_semantic_alignment.language_concept_alignment.language.as_token_string(),
            )

        self._observation_num += 1

        semantics = LearnerSemantics.from_nodes(
            language_perception_semantic_alignment.perception_semantic_alignment.semantic_nodes
        )
        language = language_perception_semantic_alignment.language_concept_alignment
        for semantic_node in semantics.actions:
            if action_learner.templates_for_concept(semantic_node.concept):
                for action_template in action_learner.templates_for_concept(
                    semantic_node.concept
                ):
                    elements = action_template.elements
                    if (
                        elements
                        not in self._concept_elements_to_arguments_to_function_counter.keys()
                    ):
                        self._concept_elements_to_arguments_to_function_counter[
                            elements
                        ] = dict(
                            (slot, StringFunctionCounter())
                            for slot in semantic_node.slot_fillings.keys()
                        )
                    for (slot, slot_filler) in semantic_node.slot_fillings.items():
                        if (
                            slot
                            not in self._concept_elements_to_arguments_to_function_counter[
                                elements
                            ].keys()
                        ):
                            raise RuntimeError(
                                f"Tried to align functional use to slot: {slot} in concept {semantic_node.concept} but {slot} didn't exist in the concept"
                            )
                        if slot_filler not in language.node_to_language_span.keys():
                            raise RuntimeError(
                                f"Tried to match slot filler: {slot_filler} to span in language but {slot_filler} wasn't in {language.node_to_language_span}"
                            )
                        span = language.node_to_language_span[slot_filler]
                        aligned_text = self._remove_determiners(
                            language.language.as_token_sequence()[span.start : span.end]
                        )
                        self._concept_elements_to_arguments_to_function_counter[elements][
                            slot
                        ].add_example(aligned_text)

    def describe(self, perception_semantic_alignment: PerceptionSemanticAlignment):
        # The functional learner doesn't quite 'describe' in the same way
        # as our other learners, rather it gets asked when needed to fill a slot
        raise NotImplementedError

    def template_for_concept(
        self,
        action_elements: Tuple[Union[str, SyntaxSemanticsVariable], ...],
        slot: SyntaxSemanticsVariable,
    ) -> Optional[Tuple[str, ...]]:
        if action_elements not in self._concept_elements_to_arguments_to_function_counter:
            return None
        if (
            slot
            not in self._concept_elements_to_arguments_to_function_counter[
                action_elements
            ]
        ):
            return None
        return self._concept_elements_to_arguments_to_function_counter[action_elements][
            slot
        ].get_best_guess_token()
