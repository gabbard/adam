import logging
import random
from itertools import chain

from immutablecollections import immutableset

from adam.curriculum.phase1_curriculum import PHASE1_CHOOSER_FACTORY, phase1_instances
from adam.curriculum.pursuit_curriculum import make_simple_pursuit_curriculum
from adam.language_specific.english.english_language_generator import (
    IGNORE_COLORS,
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.learner import (
    LearningExample,
    PerceptionSemanticAlignment,
    LanguagePerceptionSemanticAlignment,
)
from adam.learner.alignments import LanguageConceptAlignment
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.objects import (
    ObjectPursuitLearner,
    SubsetObjectLearnerNew,
    PursuitObjectLearnerNew,
)
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import (
    BALL,
    BIRD,
    BOX,
    DOG,
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    HEAD,
    LEARNER,
    MOM,
    on,
    HAND,
    HOUSE,
)
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_1_PERCEPTION_GENERATOR,
)
from adam.perception.perception_graph import DumpPartialMatchCallback, PerceptionGraph
from adam.random_utils import RandomChooser
from adam.relation import flatten_relations
from adam.relation_dsl import negate
from adam.semantics import ObjectSemanticNode
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    all_possible,
    color_variable,
    object_variable,
)


def run_learner_for_object(learner: IntegratedTemplateLearner, obj: OntologyNode):
    learner_obj = object_variable("learner_0", LEARNER)
    colored_obj_object = object_variable(
        "obj-with-color", obj, added_properties=[color_variable("color")]
    )

    obj_template = Phase1SituationTemplate(
        "colored-obj-object",
        salient_object_variables=[colored_obj_object, learner_obj],
        syntax_hints=[IGNORE_COLORS],
    )

    obj_curriculum = phase1_instances(
        "all obj situations",
        situations=all_possible(
            obj_template,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
        ),
    )
    test_obj_curriculum = phase1_instances(
        "obj test",
        situations=all_possible(
            obj_template,
            chooser=PHASE1_CHOOSER_FACTORY(),
            ontology=GAILA_PHASE_1_ONTOLOGY,
        ),
    )

    for training_stage in [obj_curriculum]:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in [test_obj_curriculum]:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert gold in [
                desc.as_token_sequence() for desc in descriptions_from_learner
            ]


def test_subset_learner_ball():
    learner = IntegratedTemplateLearner(
        object_learner=SubsetObjectLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5
        )
    )
    run_learner_for_object(learner, BALL)


def test_subset_learner_dog():
    learner = IntegratedTemplateLearner(
        object_learner=SubsetObjectLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5
        )
    )
    # debug_callback = DumpPartialMatchCallback(render_path="../renders/")
    # We pass this callback into the learner; it is executed if the learning takes too long, i.e after 60 seconds.
    run_learner_for_object(learner, DOG)


def test_subset_learner_subobject():
    mom = SituationObject.instantiate_ontology_node(
        ontology_node=MOM, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    head = SituationObject.instantiate_ontology_node(
        ontology_node=HEAD, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    hand = SituationObject.instantiate_ontology_node(
        ontology_node=HAND, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    ball = SituationObject.instantiate_ontology_node(
        ontology_node=BALL, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    house = SituationObject.instantiate_ontology_node(
        ontology_node=HOUSE, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    ground = SituationObject.instantiate_ontology_node(
        ontology_node=GROUND, ontology=GAILA_PHASE_1_ONTOLOGY
    )

    mom_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=immutableset([mom])
    )

    floating_head_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([head]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(head, ground))),
    )

    # Need to include some extra situations so that the learner will prune its semantics for 'a'
    # away and not recognize it as an object.
    floating_hand_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([hand]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(hand, ground))),
    )

    floating_ball_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([ball]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(ball, ground))),
    )

    floating_house_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=immutableset([house]),
        other_objects=immutableset([ground]),
        always_relations=flatten_relations(negate(on(house, ground))),
    )

    object_learner = SubsetObjectLearnerNew(ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5)

    for situation in [
        mom_situation,
        floating_head_situation,
        floating_hand_situation,
        floating_ball_situation,
        floating_house_situation,
    ]:
        perceptual_representation = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
            situation, chooser=RandomChooser.for_seed(0)
        )
        for linguistic_description in GAILA_PHASE_1_LANGUAGE_GENERATOR.generate_language(
            situation, chooser=RandomChooser.for_seed(0)
        ):
            perception_graph = PerceptionGraph.from_frame(
                perceptual_representation.frames[0]
            )

            object_learner.learn_from(
                LanguagePerceptionSemanticAlignment(
                    language_concept_alignment=LanguageConceptAlignment.create_unaligned(
                        language=linguistic_description
                    ),
                    perception_semantic_alignment=PerceptionSemanticAlignment(
                        perception_graph=perception_graph, semantic_nodes=[]
                    ),
                )
            )

    mom_perceptual_representation = GAILA_PHASE_1_PERCEPTION_GENERATOR.generate_perception(
        mom_situation, chooser=RandomChooser.for_seed(0)
    )
    perception_graph = PerceptionGraph.from_frame(mom_perceptual_representation.frames[0])
    enriched = object_learner.enrich_during_description(
        PerceptionSemanticAlignment.create_unaligned(perception_graph)
    )

    semantic_node_types_and_debug_strings = {
        (type(semantic_node), semantic_node.concept.debug_string)
        for semantic_node in enriched.semantic_nodes
    }
    assert (ObjectSemanticNode, "Mom") in semantic_node_types_and_debug_strings
    assert (ObjectSemanticNode, "head") in semantic_node_types_and_debug_strings
    assert (ObjectSemanticNode, "hand") in semantic_node_types_and_debug_strings


def learner_test_pursuit_curriculum(learner):
    target_objects = [
        BALL,
        # PERSON,
        # CHAIR,
        # TABLE,
        # DOG,
        # BIRD,
        # BOX,
    ]
    target_train_templates = []
    target_test_templates = []
    for obj in target_objects:
        # Create train and test templates for the target objects
        train_obj_object = object_variable("obj-with-color", obj)
        obj_template = Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[train_obj_object],
            syntax_hints=[IGNORE_COLORS],
        )
        target_train_templates.extend(
            chain(
                *[
                    all_possible(
                        obj_template,
                        chooser=PHASE1_CHOOSER_FACTORY(),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                    )
                    for _ in range(50)
                ]
            )
        )

        test_obj_object = object_variable("obj-with-color", obj)
        test_template = Phase1SituationTemplate(
            "colored-obj-object",
            salient_object_variables=[test_obj_object],
            syntax_hints=[IGNORE_COLORS],
        )
        target_test_templates.extend(
            all_possible(
                test_template,
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
            )
        )
    rng = random.Random()
    rng.seed(0)
    random.shuffle(target_train_templates, random=rng.random)

    # We can use this to generate the actual pursuit curriculum
    train_curriculum = make_simple_pursuit_curriculum(
        target_objects=target_objects,
        num_instances=15,
        num_objects_in_instance=3,
        num_noise_instances=0,
    )

    test_obj_curriculum = phase1_instances("obj test", situations=target_test_templates)

    for training_stage in [train_curriculum]:
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in training_stage.instances():
            print(linguistic_description)
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    for test_instance_group in [test_obj_curriculum]:
        for (
            _,
            test_instance_language,
            test_instance_perception,
        ) in test_instance_group.instances():
            logging.info("lang: %s", test_instance_language)
            descriptions_from_learner = learner.describe(test_instance_perception)
            gold = test_instance_language.as_token_sequence()
            assert gold in [
                desc.as_token_sequence() for desc in descriptions_from_learner
            ]


# def test_get_largest_matching_pattern():
#     target_objects = [
#         # BIRD,
#         BOX,
#         # BALL
#     ]
#
#     target_train_templates = []
#     target_test_templates = []
#     for obj in target_objects:
#         # Create train and test templates for the target objects
#         train_obj_object = object_variable("obj-with-color", obj)
#         obj_template = Phase1SituationTemplate(
#             "colored-obj-object", salient_object_variables=[train_obj_object]
#         )
#         target_train_templates.extend(
#             chain(
#                 *[
#                     all_possible(
#                         obj_template,
#                         chooser=PHASE1_CHOOSER,
#                         ontology=GAILA_PHASE_1_ONTOLOGY,
#                     )
#                     for _ in range(1)
#                 ]
#             )
#         )
#         test_obj_object = object_variable("obj-with-color", obj)
#         test_template = Phase1SituationTemplate(
#             "colored-obj-object",
#             salient_object_variables=[test_obj_object],
#             syntax_hints=[IGNORE_COLORS],
#         )
#         target_test_templates.extend(
#             all_possible(
#                 test_template, chooser=PHASE1_CHOOSER, ontology=GAILA_PHASE_1_ONTOLOGY
#             )
#         )
#
#     rng = random.Random()
#     rng.seed(0)
#     random.shuffle(target_train_templates, random=rng.random)
#
#     train_curriculum = phase1_instances(
#         "all obj situations", situations=target_train_templates
#     )
#     learner = PursuitLanguageLearner(
#         learning_factor=0.5,
#         graph_match_confirmation_threshold=0.9,
#         lexicon_entry_threshold=0.7,
#     )  # type: ignore
#     for (_, _, perceptual_representation) in train_curriculum.instances():
#         perception = graph_without_learner(
#             PerceptionGraph.from_frame(
#                 perceptual_representation.frames[0]
#             ).copy_as_digraph()
#         )
#         meanings = learner.get_meanings_from_perception(
#             observed_perception_graph=perception
#         )
#         meaning = max(meanings, key=lambda x: len(x.copy_as_digraph().nodes))
#
#         whole_perception_pattern = PerceptionGraphPattern.from_graph(
#             perception.copy_as_digraph()
#         ).perception_graph_pattern
#
#         # Test complete match, where pattern is smalelr than perception
#         print("\nComplete match:")
#         for i in range(10):
#             hypothesis = PerceptionGraphPattern(meaning.copy_as_digraph())
#             common_pattern = get_largest_matching_pattern(hypothesis, perception)
#             print(
#                 i,
#                 "p:",
#                 len(perception.copy_as_digraph().nodes),
#                 "h:",
#                 len(hypothesis.copy_as_digraph().nodes),
#                 "c:",
#                 len(common_pattern.copy_as_digraph().nodes),
#             )
#
#         print("\nPartial match:")
#         # Test partial match, where perception pattern is larger than perception
#         # TODO: Partial match is not working!
#         #  it gives up as soon as it determines it is impossible to complete the match
#         #  1) different search orders = different matches
#         #  2) there’s no guarantee that a different search order won’t yield a bigger partial match
#         for i in range(10):
#             partial_perception = PerceptionGraph(
#                 subgraph(
#                     perception.copy_as_digraph(),
#                     random.sample(perception.copy_as_digraph().nodes, 5),
#                 )
#             )
#             hypothesis = whole_perception_pattern
#             common_pattern = get_largest_matching_pattern(meaning, partial_perception)
#             print(
#                 i,
#                 "p:",
#                 len(partial_perception.copy_as_digraph().nodes),
#                 "h:",
#                 len(hypothesis.copy_as_digraph().nodes),
#                 "c:",
#                 len(common_pattern.copy_as_digraph().nodes),
#             )

def test_old_pursuit_object_learner():
    # debug_callback = DumpPartialMatchCallback(render_path="../renders/")

    # All parameters should be in the range 0-1.
    # Learning factor works better when kept < 0.5
    # Graph matching threshold doesn't seem to matter that much, as often seems to be either a
    # complete or a very small match.
    # The lexicon threshold works better between 0.07-0.3, but we need to play around with it because we end up not
    # lexicalize items sufficiently because of diminishing lexicon probability through training
    rng = random.Random()
    rng.seed(0)
    learner = ObjectPursuitLearner(
        learning_factor=0.5,
        graph_match_confirmation_threshold=0.7,
        lexicon_entry_threshold=0.7,
        rng=rng,
        smoothing_parameter=0.001,
        ontology=GAILA_PHASE_1_ONTOLOGY,
        # debug_callback=debug_callback,
    )  # type: ignore
    learner_test_pursuit_curriculum(learner)

def test_new_pursuit_learner_ball():
    rng = random.Random()
    rng.seed(0)

    learner = IntegratedTemplateLearner(
        object_learner=PursuitObjectLearnerNew(
            learning_factor=0.5,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.001,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
    )
    run_learner_for_object(learner, BALL)


def test_new_pursuit_learner_dog():
    rng = random.Random()
    rng.seed(0)
    learner = IntegratedTemplateLearner(
        object_learner=PursuitObjectLearnerNew(
            learning_factor=0.5,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.001,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
    )
    # debug_callback = DumpPartialMatchCallback(render_path="../renders/")
    # We pass this callback into the learner; it is executed if the learning takes too long, i.e after 60 seconds.
    run_learner_for_object(learner, DOG)


def test_new_pursuit_object_learner():
    rng = random.Random()
    rng.seed(0)

    learner = IntegratedTemplateLearner(
        object_learner=PursuitObjectLearnerNew(
            learning_factor=0.5,
            graph_match_confirmation_threshold=0.7,
            lexicon_entry_threshold=0.7,
            rng=rng,
            smoothing_parameter=0.001,
            ontology=GAILA_PHASE_1_ONTOLOGY,
        )
    )
    learner_test_pursuit_curriculum(learner)
