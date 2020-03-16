"""
Representations of template-with-slots-like patterns over token strings.
"""
from typing import Tuple, Union, Mapping, List

from adam.language import TokenSequenceLinguisticDescription
from adam.perception.perception_graph import LanguageAlignedPerception, MatchedObjectNode
from attr import attrs, attrib
from attr.validators import instance_of, deep_iterable

from immutablecollections.converter_utils import _to_tuple


@attrs(frozen=True, slots=True)
class SurfaceTemplateVariable:
    """
    A variable portion of a a `SurfaceTemplate`
    """

    name: str = attrib(validator=instance_of(str))


@attrs(frozen=True, slots=True)
class SurfaceTemplate:
    r"""
    A pattern over `TokenSequenceLinguisticDescription`\ s.

    Such a pattern consists of a sequence of token strings and `SurfaceTemplateVariable`\ s.
    """
    elements: Tuple[Union[str, SurfaceTemplateVariable], ...] = attrib(  # type: ignore
        converter=_to_tuple,
        validator=deep_iterable(instance_of((str, SurfaceTemplateVariable))),
    )

    @staticmethod
    def from_language_aligned_perception(
        language_aligned_perception: LanguageAlignedPerception,
        object_node_to_template_variable: Mapping[
            MatchedObjectNode, SurfaceTemplateVariable
        ],
    ) -> "SurfaceTemplate":
        if len(object_node_to_template_variable) != len(
            language_aligned_perception.node_to_language_span
        ):
            raise RuntimeError(
                "We currently only allow for the situation "
                "where every matched object corresponds to a template node."
            )

        # This will be used to build the returned SurfaceTemplate.
        # We start from the full surface string...
        template_elements: List[Union[SurfaceTemplateVariable, str]] = list(
            language_aligned_perception.language
        )

        # and the we will walk through it backwards
        # replacing any spans of text which are aligned to objects with variables.
        # We iterate backwards so we the indices don't get invalidated
        # as we replace parts of the token sequence with template variables.

        object_nodes_sorted_by_reversed_aligned_token_position = tuple(
            reversed(
                sorted(
                    language_aligned_perception.node_to_language_span.keys(),
                    key=lambda match_node: language_aligned_perception.node_to_language_span[
                        match_node
                    ],
                )
            )
        )

        for matched_object_node in object_nodes_sorted_by_reversed_aligned_token_position:
            aligned_token_span = language_aligned_perception.node_to_language_span[
                matched_object_node
            ]
            # If an object is aligned to multiple tokens,
            # we just delete all but the first.
            if len(aligned_token_span) > 1:
                for non_initial_aligned_index in range(
                    # -1 because the end index is exclusive
                    aligned_token_span.end - 1,
                    aligned_token_span.start,
                    -1,
                ):
                    del template_elements[non_initial_aligned_index]
            # Regardless, the first token is replaced by a variable.
            template_elements[
                aligned_token_span.start
            ] = object_node_to_template_variable[matched_object_node]

        return SurfaceTemplate(template_elements)

    def instantiate(
        self, template_variable_to_filler: Mapping[SurfaceTemplateVariable, Tuple[str]]
    ) -> TokenSequenceLinguisticDescription:
        """
        Turns a template into a `TokenSequenceLinguisticDescription` by filling in its variables.
        """
        output_tokens: List[str] = []
        for element in self.elements:
            if isinstance(element, SurfaceTemplateVariable):
                filler_words = template_variable_to_filler[element]
                # English-specific hack to deal with us not understanding determiners:
                # https://github.com/isi-vista/adam/issues/498
                # The "is lower" check is a hack to block adding a determiner to proper names.
                if len(filler_words) == 1 and filler_words[0][0].islower():
                    output_tokens.append("a")
                output_tokens.extend(filler_words)
            else:
                # element must be a single token str due to object validity checks.
                output_tokens.append(element)
        return TokenSequenceLinguisticDescription(output_tokens)


SLOT1 = SurfaceTemplateVariable("slot1")
SLOT2 = SurfaceTemplateVariable("slot2")
SLOT3 = SurfaceTemplateVariable("slot3")
SLOT4 = SurfaceTemplateVariable("slot4")
SLOT5 = SurfaceTemplateVariable("slot5")
SLOT6 = SurfaceTemplateVariable("slot6")

STANDARD_SLOT_VARIABLES = (SLOT1, SLOT2, SLOT3, SLOT4, SLOT5, SLOT6)