import multiprocessing as mp
import queue
import logging
from itertools import chain

import pytest
from adam.curriculum.curriculum_utils import (
    PHASE1_CHOOSER_FACTORY,
    PHASE1_TEST_CHOOSER_FACTORY,
    phase1_instances,
    standard_object,
)
from adam.curriculum.phase1_curriculum import (
    _make_come_down_template,
    make_push_templates,
    make_drink_template,
    make_eat_template,
    make_fall_templates,
    make_fly_templates,
    make_give_templates,
    make_go_templates,
    make_jump_templates,
    make_move_templates,
    make_put_templates,
    make_roll_templates,
    make_sit_templates,
    make_spin_templates,
    make_take_template,
    make_throw_templates,
    make_throw_animacy_templates,
)
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.verbs import SubsetVerbLearner, SubsetVerbLearnerNew
from adam.ontology import IS_SPEAKER, THING, IS_ADDRESSEE
from adam.ontology.phase1_ontology import (
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    INANIMATE,
    AGENT,
    ANIMATE,
    GAILA_PHASE_1_ONTOLOGY,
    GOAL,
    HAS_SPACE_UNDER,
    LEARNER,
    PERSON,
    GROUND,
    COME,
    CAN_JUMP,
    EDIBLE,
    SELF_MOVING,
    HOLLOW,
    PERSON_CAN_HAVE,
    LIQUID,
)
from adam.situation import Action
from adam.situation.templates.phase1_situation_templates import (
    _go_under_template,
    _jump_over_template,
)
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    sampled,
    object_variable,
)
from immutablecollections import immutableset
from tests.learner import (
    LANGUAGE_MODE_TO_OBJECT_RECOGNIZER,
    LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER,
)


def subset_verb_language_factory(language_mode: LanguageMode) -> SubsetVerbLearner:
    return SubsetVerbLearner(
        object_recognizer=LANGUAGE_MODE_TO_OBJECT_RECOGNIZER[language_mode],
        ontology=GAILA_PHASE_1_ONTOLOGY,
        language_mode=language_mode,
    )


def integrated_learner_factory(language_mode: LanguageMode):
    return IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        action_learner=SubsetVerbLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
    )


# VerbPursuitLearner(
#         learning_factor=0.5,
#         graph_match_confirmation_threshold=0.7,
#         lexicon_entry_threshold=0.7,
#         rng=rng,
#         smoothing_parameter=0.001,
#         ontology=GAILA_PHASE_1_ONTOLOGY,
#     )  # type: ignore


def _train_curriculum(situation_template, language_generator):
    return phase1_instances(
        "train",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=10,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
            ]
        ),
        language_generator=language_generator,
    )


def _test_curriculum(situation_template, language_generator):
    return phase1_instances(
        "test",
        chain(
            *[
                sampled(
                    situation_template,
                    max_to_sample=1,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_TEST_CHOOSER_FACTORY(),
                )
            ]
        ),
        language_generator=language_generator,
    )


def run_verb_test_iterators(learner, train_iter, test_iter):
    for (_, linguistic_description, perceptual_representation) in train_iter:
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )
    for (_, test_lingustics_description, test_perceptual_representation) in test_iter:
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


def run_verb_test(learner, situation_template, language_generator):
    train_curriculum = _train_curriculum(situation_template, language_generator)
    test_curriculum = _test_curriculum(situation_template, language_generator)
    run_verb_test_iterators(
        learner, train_curriculum.instances(), test_curriculum.instances()
    )


_QUEUE_DONE = None


PARALLEL_INSTANCE_GENERATION_TIMEOUT_SECONDS = 60


def _curriculum_worker(instance_queue, situation_template, language_generator, *, train_curriculum=True):
    # NTS: If this doesn't work try instead passing language_mode to this (through calls up to
    # run_verb_test_parallel) and use the subset_verb_language_factory to create the language
    # generator
    if train_curriculum:
        curriculum = _train_curriculum(situation_template, language_generator)
    else:
        curriculum = _test_curriculum(situation_template, language_generator)

    for instance in curriculum.instances():
        instance_queue.put(instance)

    instance_queue.put(_QUEUE_DONE)


def _curriculum_generator(
        pool, manager, situation_template, language_generator, *, train_curriculum=True
):
    instance_queue = manager.Queue()

    pool.apply_async(
        _curriculum_worker,
        args=(instance_queue, situation_template, language_generator),
        kwds={'train_curriculum': train_curriculum},
        error_callback=lambda err: print(f'Worker crashed with error: {err}'),
    )

    def generator():
        while True:
            try:
                value = instance_queue.get(
                    timeout=PARALLEL_INSTANCE_GENERATION_TIMEOUT_SECONDS
                )
                if value != _QUEUE_DONE:
                    yield value
                else:
                    break
            except queue.Empty:
                logging.warning("Timed out while waiting for next instance.")
                break
    return generator()


def _make_parallel_train_test_iterators(situation_templates, language_generator):
    train_test_pairs = []
    with mp.Pool() as pool:
        manager = mp.Manager()
        for situation_template in situation_templates:
            train = _curriculum_generator(pool, manager, situation_template, language_generator)
            test = _curriculum_generator(
                pool, manager, situation_template, language_generator, train_curriculum=False
            )
            train_test_pairs.append((train, test))
    return train_test_pairs


def run_verb_test_parallel(learner, situation_templates, language_generator):
    train_test_pairs = _make_parallel_train_test_iterators(
        situation_templates, language_generator
    )
    for train, test in train_test_pairs:
        run_verb_test_iterators(learner, train, test)


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_eat_simple(language_mode, learner):
    object_to_eat = standard_object("object_0", required_properties=[EDIBLE])
    eater = standard_object(
        "eater_0",
        THING,
        required_properties=[ANIMATE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    run_verb_test_parallel(
        learner(language_mode),
        [make_eat_template(eater, object_to_eat)],
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_drink(language_mode, learner):
    object_0 = standard_object(
        "object_0",
        required_properties=[HOLLOW, PERSON_CAN_HAVE],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    liquid_0 = object_variable("liquid_0", required_properties=[LIQUID])
    person_0 = standard_object(
        "person_0", PERSON, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    run_verb_test_parallel(
        learner(language_mode),
        [make_drink_template(person_0, liquid_0, object_0, None)],
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_sit(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_sit_templates(None),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_put(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_put_templates(None),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_push(language_mode, learner):
    push_templates = make_push_templates(
        agent=standard_object(
            "pusher",
            THING,
            required_properties=[ANIMATE],
            banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        ),
        theme=standard_object("pushee", INANIMATE_OBJECT),
        push_surface=standard_object(
            "push_surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
        ),
        push_goal=standard_object("push_goal", INANIMATE_OBJECT),
        use_adverbial_path_modifier=False,
    )
    run_verb_test_parallel(
        learner(language_mode),
        push_templates,
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_go(language_mode, learner):
    goer = standard_object("goer", THING, required_properties=[ANIMATE])
    under_goal_reference = standard_object(
        "go-under-goal", THING, required_properties=[HAS_SPACE_UNDER]
    )

    under_templates = [
        _go_under_template(goer, under_goal_reference, [], is_distal=is_distal)
        for is_distal in (True, False)
    ]

    run_verb_test_parallel(
        learner(language_mode),
        make_go_templates(None),
        language_generator=phase1_language_generator(language_mode)
    )

    run_verb_test_parallel(
        learner(language_mode),
        under_templates,
        language_generator=phase1_language_generator(language_mode)
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_come(language_mode, learner):
    movee = standard_object(
        "movee",
        required_properties=[SELF_MOVING],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    learner_obj = standard_object("leaner_0", LEARNER)
    speaker = standard_object(
        "speaker",
        PERSON,
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
        added_properties=[IS_SPEAKER],
    )
    object_ = standard_object(
        "object_0", THING, banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )
    ground = standard_object("ground", root_node=GROUND)

    come_to_speaker = Phase1SituationTemplate(
        "come-to-speaker",
        salient_object_variables=[movee, speaker],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, speaker)])
        ],
    )
    come_to_learner = Phase1SituationTemplate(
        "come-to-leaner",
        salient_object_variables=[movee],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, learner_obj)])
        ],
    )
    come_to_object = Phase1SituationTemplate(
        "come-to-object",
        salient_object_variables=[movee, object_],
        actions=[
            Action(COME, argument_roles_to_fillers=[(AGENT, movee), (GOAL, object_)])
        ],
    )
    come_templates = [
        _make_come_down_template(movee, object_, speaker, ground, immutableset()),
        come_to_speaker,
        come_to_learner,
        come_to_object,
    ]
    run_verb_test_parallel(
        learner(language_mode),
        come_templates,
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_take(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        [make_take_template(
            agent=standard_object("taker_0", THING, required_properties=[ANIMATE]),
            theme=standard_object("object_taken_0", required_properties=[INANIMATE]),
            use_adverbial_path_modifier=False,
        )],
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_give(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_give_templates(immutableset()),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_spin(language_mode, learner):
    run_verb_test(
        learner(language_mode),
        make_spin_templates(None),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_fall(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_fall_templates(immutableset()),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_throw(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_throw_templates(None),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize(
    "language_mode",
    [LanguageMode.CHINESE, pytest.param(LanguageMode.ENGLISH, marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
# this tests gei vs. dau X shang for Chinese throw to
# TODO: fix English implementation https://github.com/isi-vista/adam/issues/870
# Not yet parallelized.
def test_throw_animacy(language_mode, learner):
    # shuffle both together for the train curriculum
    train_curriculum = phase1_instances(
        "train",
        chain(
            *[
                sampled(
                    situation_template=situation_template,
                    max_to_sample=10,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
                for situation_template in make_throw_animacy_templates(None)
            ]
        ),
        language_generator=phase1_language_generator(language_mode),
    )
    # shuffle both together for test curriculum
    test_curriculum = phase1_instances(
        "test",
        chain(
            *[
                sampled(
                    situation_template=situation_template,
                    max_to_sample=1,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                    chooser=PHASE1_CHOOSER_FACTORY(),
                )
                for situation_template in make_throw_animacy_templates(None)
            ]
        ),
        language_generator=phase1_language_generator(language_mode),
    )
    # instantiate and test the learner
    learner = learner(language_mode)
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in train_curriculum.instances():
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )

    for (
        _,
        test_lingustics_description,
        test_perceptual_representation,
    ) in test_curriculum.instances():
        descriptions_from_learner = learner.describe(test_perceptual_representation)
        gold = test_lingustics_description.as_token_sequence()
        assert descriptions_from_learner
        assert gold in [desc.as_token_sequence() for desc in descriptions_from_learner]


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_move(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_move_templates(None),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_jump(language_mode, learner):

    jumper = standard_object(
        "jumper_0",
        THING,
        required_properties=[CAN_JUMP],
        banned_properties=[IS_SPEAKER, IS_ADDRESSEE],
    )
    jumped_over = standard_object(
        "jumped_over", banned_properties=[IS_SPEAKER, IS_ADDRESSEE]
    )

    run_verb_test_parallel(
        learner(language_mode),
        make_jump_templates(None),
        language_generator=phase1_language_generator(language_mode),
    )

    run_verb_test_parallel(
        learner(language_mode),
        [_jump_over_template(jumper, jumped_over, immutableset())],
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_roll(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_roll_templates(None),
        language_generator=phase1_language_generator(language_mode),
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
@pytest.mark.parametrize(
    "learner",
    [
        pytest.param(
            subset_verb_language_factory,
            marks=pytest.mark.skip("No Longer Need to Test Old Learners"),
        ),
        integrated_learner_factory,
    ],
)
def test_fly(language_mode, learner):
    run_verb_test_parallel(
        learner(language_mode),
        make_fly_templates(immutableset()),
        phase1_language_generator(language_mode),
    )
