from attr import evolve
from more_itertools import one, first

from adam.curriculum.curriculum_utils import GROUND_OBJECT_TEMPLATE
from adam.language_specific.english.english_language_generator import GAILA_PHASE_1_LANGUAGE_GENERATOR, \
    SimpleRuleBasedEnglishLanguageGenerator
from adam.language_specific.english.english_phase_1_lexicon import GAILA_PHASE_1_ENGLISH_LEXICON
from adam.learner import LanguagePerceptionSemanticAlignment
from adam.learner.alignments import LanguageConceptAlignment, PerceptionSemanticAlignment
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.objects import ObjectRecognizerAsTemplateLearner
from adam.learner.surface_templates import SLOT1, SLOT2, SLOT3
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.phase1_ontology import DAD, GAILA_PHASE_1_ONTOLOGY, TABLE, AGENT, THEME, PERSON, BALL, GOAL, \
    PUT, MOM, JUMP, JUMP_INITIAL_SUPPORTER_AUX, THROW_GOAL, THROW
from adam.ontology.phase1_spatial_relations import Region, EXTERIOR_BUT_IN_CONTACT, GRAVITATIONAL_UP, PROXIMAL
from adam.perception import PerceptualRepresentation
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.random_utils import RandomChooser
from adam.semantics import SemanticNode, Concept, ObjectConcept, ObjectSemanticNode
from adam.situation import SituationObject, Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from src.vistautils.vistautils.span import Span
from tests.learner import TEST_OBJECT_RECOGNIZER
from more_itertools import one

from adam.language_specific.english.english_language_generator import GAILA_PHASE_1_LANGUAGE_GENERATOR
from adam.learner import LanguagePerceptionSemanticAlignment
from adam.learner.alignments import LanguageConceptAlignment, PerceptionSemanticAlignment
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.objects import ObjectRecognizerAsTemplateLearner
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.phase1_ontology import DAD, GAILA_PHASE_1_ONTOLOGY, TABLE, AGENT, THEME, PERSON, BALL, GOAL, \
    PUT
from adam.ontology.phase1_spatial_relations import Region, EXTERIOR_BUT_IN_CONTACT, GRAVITATIONAL_UP
from adam.perception import PerceptualRepresentation
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.random_utils import RandomChooser
from adam.situation import SituationObject, Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tests.learner import TEST_OBJECT_RECOGNIZER


def test_with_object_recognizer():
    integrated_learner = IntegratedTemplateLearner(
        object_learner=ObjectRecognizerAsTemplateLearner(
            object_recognizer=TEST_OBJECT_RECOGNIZER
        ),
        attribute_learner=None,
        relation_learner=None,
        action_learner=None,
    )

    dad_situation_object = SituationObject.instantiate_ontology_node(
        ontology_node=DAD, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dad_situation_object]
    )
    perception_generator = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
        GAILA_PHASE_1_ONTOLOGY
    )
    # We explicitly exclude ground in perception generation

    # this generates a static perception...
    perception = perception_generator.generate_perception(
        situation, chooser=RandomChooser.for_seed(0), include_ground=False
    )

    # so we need to construct a dynamic one by hand from two identical scenes
    dynamic_perception = PerceptualRepresentation(
        frames=[perception.frames[0], perception.frames[0]]
    )

    descriptions = integrated_learner.describe(dynamic_perception)

    assert len(descriptions) == 1
    assert one(descriptions.keys()).as_token_sequence() == ("Dad",)


def test_verb_learner_candidate_template_generation():
    perception_generator = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
        GAILA_PHASE_1_ONTOLOGY
    )
    language_generator = SimpleRuleBasedEnglishLanguageGenerator(
        ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON
    )
    action_learner = SubsetVerbLearnerNew(ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=2)
    mom = SituationObject.instantiate_ontology_node(ontology_node=MOM, ontology=GAILA_PHASE_1_ONTOLOGY)
    ball = SituationObject.instantiate_ontology_node(ontology_node=BALL, ontology=GAILA_PHASE_1_ONTOLOGY)
    table = SituationObject.instantiate_ontology_node(ontology_node=TABLE, ontology=GAILA_PHASE_1_ONTOLOGY)
    mom_sn = ObjectSemanticNode(concept=ObjectConcept('mom'))
    table_sn = ObjectSemanticNode(concept=ObjectConcept('table'))
    ball_sn = ObjectSemanticNode(concept=ObjectConcept('ball'))
    # Make a LanguageConceptAlignment with the recognized object spans
    jump_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom],
        actions=[
            Action(
                JUMP,
                argument_roles_to_fillers=[(AGENT, mom)],
                auxiliary_variable_bindings=[
                    (JUMP_INITIAL_SUPPORTER_AUX, GROUND_OBJECT_TEMPLATE)
                ],
            )
        ],
    )
    throw_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, ball],
        actions=[
            Action(
                THROW,
                argument_roles_to_fillers=[(AGENT, mom), (THEME, ball)],
                auxiliary_variable_bindings=[
                    (THROW_GOAL, Region(table, distance=PROXIMAL))
                ],
            )
        ],
    )
    put_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[mom, ball, table],
        actions=[
            Action(
                PUT,
                (
                    (AGENT, mom),
                    (THEME, ball),
                    (
                        GOAL,
                        Region(
                            reference_object=table,
                            distance=EXTERIOR_BUT_IN_CONTACT,
                            direction=GRAVITATIONAL_UP,
                        ),
                    ),
                ),
            )
        ],
    )

    # one test case for 1 rec object, one for 2, one for 3 rec objects
    test_list = [(jump_situation, {mom_sn: Span(0,1)}, 'jumps'), # Single object
                 (throw_situation, {mom_sn: Span(0,1), ball_sn: Span(2,4)}, 'throws'), # Two objects
                 # (put_situation, {mom_sn: Span(0,1), ball_sn: Span(2,4), table_sn: Span(5,7)}, 'puts'), # Three objects doesn't work for now
                ]

    for situation, node_to_language_span, verb_phrase in test_list:
        linguistic_description = language_generator.generate_language(
            situation=situation, chooser=RandomChooser.for_seed(0))
        language_alignment = LanguageConceptAlignment(language=first(linguistic_description),
                                                      node_to_language_span=node_to_language_span)
        language_perception_semantic_alignment = LanguagePerceptionSemanticAlignment(
            language_concept_alignment=language_alignment,
            perception_semantic_alignment=PerceptionSemanticAlignment(
                # A necessary gap filler for the test
                perception_graph=PerceptionGraph.from_frame(perception_generator.generate_perception(HighLevelSemanticsSituation(
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        salient_objects=[mom]),
                        chooser=RandomChooser.for_seed(0)).frames[0]),
                semantic_nodes=[],
            )
        )
        print(first(linguistic_description).as_token_sequence())

        candidate_templates = action_learner._candidate_templates(language_perception_semantic_alignment)
        assert len(candidate_templates) == 1
        assert verb_phrase in first(candidate_templates).surface_template.elements
        for template in action_learner._candidate_templates(language_perception_semantic_alignment):
            print(template)