from more_itertools import first

from adam.curriculum.curriculum_utils import PHASE1_CHOOSER_FACTORY, phase1_instances
from adam.language_specific.english.english_language_generator import PREFER_DITRANSITIVE
from adam.ontology.phase1_ontology import (
    DAD,
    GAILA_PHASE_1_ONTOLOGY,
    TRANSFER_OF_POSSESSION,
    PERSON,
    BABY,
    CHAIR,
    GIVE,
    AGENT,
    GOAL,
    THEME,
)
from adam.perception import PerceptualRepresentation
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.random_utils import RandomChooser
from adam.situation import SituationObject, Action
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    sampled,
)
from learner import TEST_OBJECT_RECOGNIZER


def test_trivial_dynamic_situation_with_schemaless_object():
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

    perception_graph = PerceptionGraph.from_dynamic_perceptual_representation(
        dynamic_perception
    )

    match_result = TEST_OBJECT_RECOGNIZER.match_objects(perception_graph)
    assert len(match_result.description_to_matched_object_node) == 1
    assert ("Dad",) in match_result.description_to_matched_object_node


def test_true_dynamic_situation():
    dad = object_variable("person_0", DAD)
    baby = object_variable("person_1", BABY)
    chair = object_variable("give_object_0", CHAIR)

    giving_template = Phase1SituationTemplate(
        "dad-transfer-of-possession",
        salient_object_variables=[dad, baby, chair],
        actions=[
            Action(
                GIVE,
                argument_roles_to_fillers=[(AGENT, dad), (GOAL, baby), (THEME, chair)],
            )
        ],
        syntax_hints=[PREFER_DITRANSITIVE],
    )

    (_, _, perception) = first(
        phase1_instances(
            "foo",
            sampled(
                giving_template,
                max_to_sample=1,
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
            ),
        ).instances()
    )

    perception_graph = PerceptionGraph.from_dynamic_perceptual_representation(perception)
    match_result = TEST_OBJECT_RECOGNIZER.match_objects(perception_graph)
    assert len(match_result.description_to_matched_object_node) == 1
    assert ("Dad",) in match_result.description_to_matched_object_node