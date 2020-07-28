r"""
Code to specify what is shown to `LanguageLearner`\ s and in what order.
"""
from abc import ABC, abstractmethod
from typing import Generic, Iterable, Optional, Tuple, List
import multiprocessing as mp

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections.converter_utils import _to_tuple
from more_itertools import flatten

from adam.language import LinguisticDescriptionT
from adam.language.language_generator import LanguageGenerator
from adam.situation import SituationT
from adam.perception import (
    PerceptionT,
    PerceptualRepresentation,
    PerceptualRepresentationGenerator,
)
from adam.random_utils import SequenceChooser


class InstanceGroup(ABC, Generic[SituationT, LinguisticDescriptionT, PerceptionT]):
    r"""
    An `InstanceGroup` can provide triples of (optional)
    `Situation`\ s, `LinguisticDescription`\ s, and `PerceptualRepresentation`\ s
    for use in training or testing `LanguageLearner`\ s
    with the `Experiment` class.
    """

    @abstractmethod
    def name(self) -> str:
        """
        A human-readable name for this instance group.
        """

    @abstractmethod
    def instances(
        self
    ) -> Iterable[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ]:
        """
        The instances in the order they should be shown to the `LanguageLearner`.
        """


@attrs(frozen=True, slots=True)
class ExplicitWithoutSituationInstanceGroup(
    InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    A collection of instances where the user explicitly specifies
    the `LinguisticDescription`\ s and `PerceptualRepresentation`\ s
    but not the `Situation`\ s.
    """
    _name: str = attrib(validator=instance_of(str))
    # https://github.com/python-attrs/attrs/issues/519
    _instances: Tuple[  # type: ignore
        Tuple[LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
    ] = attrib(  # type: ignore
        converter=_to_tuple
    )

    def name(self) -> str:
        return self._name

    def instances(
        self
    ) -> Iterable[
        Tuple[SituationT, LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
    ]:
        for (linguistic_description, perception) in self._instances:
            yield (None, linguistic_description, perception)  # type: ignore


@attrs(frozen=True, slots=True)
class ExplicitWithSituationInstanceGroup(
    InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    A collection of instances where the user explicitly specifies
    the the `Situation`\ s, `LinguisticDescription`\ s, and `PerceptualRepresentation`\ s.
    """
    _name: str = attrib(validator=instance_of(str))
    # https://github.com/python-attrs/attrs/issues/519
    _instances: Tuple[  # type: ignore
        Tuple[SituationT, LinguisticDescriptionT, PerceptualRepresentation[PerceptionT]]
    ] = attrib(  # type: ignore
        converter=_to_tuple
    )

    def name(self) -> str:
        return self._name

    def instances(
        self
    ) -> Iterable[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ]:
        return self._instances


@attrs(frozen=True, slots=True)
class GeneratedFromSituationsInstanceGroup(
    InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Creates a collection of instances
    by taking an iterable of `Situation`\ s
    and deriving the `LinguisticDescription`\ s and `PerceptualRepresentation`\ s
    by applying the *language_generator* and *perception_generator*, respectively.
    """
    _name: str = attrib(validator=instance_of(str))
    """
    The name of the instance group.
    """
    _situations: Tuple[SituationT, ...] = attrib(converter=_to_tuple)
    r"""
    The sequence of `Situation`\ s to derive linguistic and perceptual representations from for
    training.

    These `Situation`\ s could themselves be produced by `SituationTemplate`\ s.
    """
    _language_generator: LanguageGenerator[SituationT, LinguisticDescriptionT] = attrib(
        validator=instance_of(LanguageGenerator)
    )
    """
    How to generate the `LanguageRepresentation` of a training `Situation`.
    """
    _perception_generator: PerceptualRepresentationGenerator[
        SituationT, PerceptionT
    ] = attrib(validator=instance_of(PerceptualRepresentationGenerator))
    """
    How to generate the `PerceptualRepresentation` of a training `Situation`
    """
    _chooser: SequenceChooser = attrib(validator=instance_of(SequenceChooser))

    def name(self) -> str:
        return self._name

    def instances(
        self
    ) -> Iterable[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ]:
        for situation in self._situations:
            # suppress PyCharm type inference bug
            # noinspection PyTypeChecker
            for linguistic_description in self._language_generator.generate_language(
                situation, self._chooser
            ):
                yield (
                    situation,
                    linguistic_description,
                    self._perception_generator.generate_perception(
                        situation, self._chooser
                    ),
                )


@attrs(frozen=True, slots=True)
class ParallelGeneratedFromSituationsInstanceGroup(
    InstanceGroup[SituationT, LinguisticDescriptionT, PerceptionT]
):
    r"""
    Creates a collection of instances
    by taking an iterable of `Situation`\ s
    and deriving the `LinguisticDescription`\ s and `PerceptualRepresentation`\ s
    by applying the *language_generator* and *perception_generator*, respectively,
    using a process pool for parallelism.
    """
    _name: str = attrib(validator=instance_of(str))
    """
    The name of the instance group.
    """
    _situations: Tuple[SituationT, ...] = attrib(converter=_to_tuple)
    r"""
    The sequence of `Situation`\ s to derive linguistic and perceptual representations from for
    training.

    These `Situation`\ s could themselves be produced by `SituationTemplate`\ s.
    """
    _language_generator: LanguageGenerator[SituationT, LinguisticDescriptionT] = attrib(
        validator=instance_of(LanguageGenerator)
    )
    """
    How to generate the `LanguageRepresentation` of a training `Situation`.
    """
    _perception_generator: PerceptualRepresentationGenerator[
        SituationT, PerceptionT
    ] = attrib(validator=instance_of(PerceptualRepresentationGenerator))
    """
    How to generate the `PerceptualRepresentation` of a training `Situation`
    """
    _chooser: SequenceChooser = attrib(validator=instance_of(SequenceChooser))

    _SUBPROCESS_POOL_SIZE: int = mp.cpu_count() - 1

    def name(self) -> str:
        return self._name

    @attrs
    class _LanguageGenerationInputs:
        situation: SituationT = attrib()
        language_generator: LanguageGenerator[SituationT, LinguisticDescriptionT] = attrib(
            validator=instance_of(LanguageGenerator)
        )
        chooser = attrib(validator=instance_of(SequenceChooser))

    @attrs
    class _PerceptionGenerationInputs:
        situation: SituationT = attrib()
        perception_generator: PerceptualRepresentationGenerator[
            SituationT, PerceptionT
        ] = attrib(validator=instance_of(PerceptualRepresentationGenerator))
        chooser = attrib(validator=instance_of(SequenceChooser))

    @staticmethod
    def all_linguistic_descriptions(input_: "ParallelGeneratedFromSituationsInstanceGroup._LanguageGenerationInputs")\
            -> Iterable[LinguisticDescriptionT]:
        return list(input_.language_generator.generate_language(input_.situation, input_.chooser))

    @staticmethod
    def all_perceptual_representations(input_: "ParallelGeneratedFromSituationsInstanceGroup._PerceptionGenerationInputs")\
            -> Iterable[PerceptualRepresentation[PerceptionT]]:
        return [input_.perception_generator.generate_perception(input_.situation, input_.chooser)]

    def instances(
            self
    ) -> Iterable[
        Tuple[
            Optional[SituationT],
            LinguisticDescriptionT,
            PerceptualRepresentation[PerceptionT],
        ]
    ]:
        pool = mp.Pool(self._SUBPROCESS_POOL_SIZE)

        all_linguistic_descriptions = pool.imap(self.all_linguistic_descriptions, [
            self._LanguageGenerationInputs(situation, self._language_generator, self._chooser) for situation in self._situations
        ])
        all_perceptual_representations = pool.imap(self.all_linguistic_descriptions, [
            self._PerceptionGenerationInputs(situation, self._perception_generator, self._chooser) for situation in self._situations
        ])

        for situation, linguistic_descriptions, perceptual_representations in zip(
            self._situations, all_linguistic_descriptions, all_perceptual_representations
        ):
            for linguistic_description in linguistic_descriptions:
                for perceptual_representation in perceptual_representations:
                    yield (
                        situation,
                        linguistic_description,
                        perceptual_representation,
                    )
