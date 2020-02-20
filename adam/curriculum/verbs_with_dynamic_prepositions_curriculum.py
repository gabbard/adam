from itertools import chain
from typing import Iterable

from immutablecollections import immutableset
from more_itertools import flatten

from adam.axes import (
    HorizontalAxisOfObject,
    FacingAddresseeAxis,
    GRAVITATIONAL_AXIS_FUNCTION,
)
from adam.curriculum.curriculum_utils import (
    standard_object,
    phase1_instances,
    PHASE1_CHOOSER,
    Phase1InstanceGroup,
)
from adam.ontology import THING
from adam.ontology.during import DuringAction
from adam.ontology.phase1_ontology import (
    AGENT,
    bigger_than,
    GOAL,
    GAILA_PHASE_1_ONTOLOGY,
    HOLLOW,
    HAS_SPACE_UNDER,
    PUSH,
    THEME,
    PUSH_SURFACE_AUX,
    on,
    ANIMATE,
    INANIMATE_OBJECT,
    CAN_HAVE_THINGS_RESTING_ON_THEM,
    GO,
    _GO_GOAL,
    above,
)
from adam.ontology.phase1_spatial_relations import (
    Region,
    PROXIMAL,
    INTERIOR,
    Direction,
    GRAVITATIONAL_DOWN,
    DISTAL,
    GRAVITATIONAL_UP,
    SpatialPath,
    VIA,
)
from adam.situation import Action
from adam.situation.templates.phase1_templates import (
    TemplateObjectVariable,
    Phase1SituationTemplate,
    sampled,
)

BOOL_SET = immutableset([True, False])

# PUSH templates


def _push_to_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-to-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=PROXIMAL)),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
        ],
    )


def _push_in_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-in-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (GOAL, Region(goal_reference, distance=INTERIOR)),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
            bigger_than(goal_reference, theme),
        ],
    )


def _push_under_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-under-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=GRAVITATIONAL_DOWN,
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
            bigger_than(goal_reference, theme),
        ],
    )


def _push_beside_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-beside-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=PROXIMAL,
                            direction=Direction(
                                positive=is_right,
                                relative_to_axis=HorizontalAxisOfObject(
                                    goal_reference, index=0
                                ),
                            ),
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
        ],
    )


def _push_in_front_of_behind_template(
    agent: TemplateObjectVariable,
    theme: TemplateObjectVariable,
    goal_reference: TemplateObjectVariable,
    surface: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    is_distal: bool,
    is_in_front: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"{agent.handle}-pushes-{theme.handle}-in-front-of-behind-{goal_reference.handle}",
        salient_object_variables=[agent, theme, goal_reference],
        background_object_variables=background,
        actions=[
            Action(
                PUSH,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (THEME, theme),
                    (
                        GOAL,
                        Region(
                            goal_reference,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=Direction(
                                positive=is_in_front,
                                relative_to_axis=FacingAddresseeAxis(goal_reference),
                            ),
                        ),
                    ),
                ],
                auxiliary_variable_bindings=[(PUSH_SURFACE_AUX, surface)],
                during=DuringAction(continuously=[on(theme, surface)]),
            )
        ],
        constraining_relations=[
            bigger_than(surface, agent),
            bigger_than(surface, goal_reference),
        ],
    )


def _go_to_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_to-{agent.handle}-to-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (GOAL, Region(goal_object, distance=PROXIMAL)),
                ],
            )
        ],
        gazed_objects=[agent],
    )


def _go_in_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_in-{agent.handle}-in-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (GOAL, Region(goal_object, distance=INTERIOR)),
                ],
            )
        ],
        constraining_relations=[bigger_than(goal_object, agent)],
    )


def _go_beside_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_right: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_beside-{agent.handle}-beside-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object,
                            distance=PROXIMAL,
                            direction=Direction(
                                positive=is_right,
                                relative_to_axis=HorizontalAxisOfObject(
                                    goal_object, index=0
                                ),
                            ),
                        ),
                    ),
                ],
            )
        ],
    )


def _go_behind_in_front_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_behind: bool,
) -> Phase1SituationTemplate:
    return Phase1SituationTemplate(
        f"go_behind-{agent.handle}-behind-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=Direction(
                                positive=False if is_behind else True,
                                relative_to_axis=FacingAddresseeAxis(goal_object),
                            ),
                        ),
                    ),
                ],
            )
        ],
    )


def _go_over_under_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_distal: bool,
    is_over: bool,
) -> Phase1SituationTemplate:
    handle = "over" if is_over else "under"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}",
        salient_object_variables=[agent, goal_object],
        background_object_variables=background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[
                    (AGENT, agent),
                    (
                        GOAL,
                        Region(
                            goal_object,
                            distance=DISTAL if is_distal else PROXIMAL,
                            direction=GRAVITATIONAL_UP if is_over else GRAVITATIONAL_DOWN,
                        ),
                    ),
                ],
            )
        ],
    )


def _go_behind_in_front_path_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    path_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_behind: bool,
) -> Phase1SituationTemplate:
    additional_background = [goal_object, path_object]
    additional_background.extend(background)
    total_background = immutableset(additional_background)
    handle = "behind" if is_behind else "in-front-of"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}-via-{path_object.handle}",
        salient_object_variables=[agent],
        background_object_variables=total_background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(_GO_GOAL, goal_object)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            path_object,
                            SpatialPath(
                                operator=VIA,
                                reference_object=path_object,
                                reference_axis=FacingAddresseeAxis(path_object),
                                orientation_changed=True,
                            ),
                        )
                    ],
                    # TODO: ADD 'at_some_point' condition for in_front or behind regional conditions
                    # See: https://github.com/isi-vista/adam/issues/583
                ),
            )
        ],
        gazed_objects=[agent],
    )


def _go_over_under_path_template(
    agent: TemplateObjectVariable,
    goal_object: TemplateObjectVariable,
    path_object: TemplateObjectVariable,
    background: Iterable[TemplateObjectVariable],
    *,
    is_over: bool,
) -> Phase1SituationTemplate:
    additional_background = [goal_object, path_object]
    additional_background.extend(background)
    total_background = immutableset(additional_background)
    handle = "over" if is_over else "under"
    return Phase1SituationTemplate(
        f"go_{handle}-{agent.handle}-{handle}-{goal_object.handle}-via-{path_object.handle}",
        salient_object_variables=[agent],
        background_object_variables=total_background,
        actions=[
            Action(
                GO,
                argument_roles_to_fillers=[(AGENT, agent)],
                auxiliary_variable_bindings=[(_GO_GOAL, goal_object)],
                during=DuringAction(
                    objects_to_paths=[
                        (
                            path_object,
                            SpatialPath(
                                operator=VIA,
                                reference_object=path_object,
                                reference_axis=GRAVITATIONAL_AXIS_FUNCTION,
                            ),
                        )
                    ],
                    at_some_point=[
                        above(agent, path_object)
                        if is_over
                        else above(path_object, agent)
                    ],
                ),
            )
        ],
        gazed_objects=[agent],
    )


def _make_push_with_prepositions(
    num_samples: int = 5, *, noise_objects: int = 0
) -> Phase1InstanceGroup:
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    theme = standard_object("theme", INANIMATE_OBJECT)
    goal_reference = standard_object("goal_reference", INANIMATE_OBJECT)
    goal_in = standard_object("goal_in", INANIMATE_OBJECT, required_properties=[HOLLOW])
    goal_under = standard_object(
        "goal_under", INANIMATE_OBJECT, required_properties=[HAS_SPACE_UNDER]
    )
    surface = standard_object(
        "surface", THING, required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM]
    )
    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )
    to_in_templates = [
        _push_to_template(agent, theme, goal_reference, surface, background),
        _push_in_template(agent, theme, goal_in, surface, background),
    ]

    return phase1_instances(
        "Push + PP",
        chain(
            # to, in
            flatten(
                [
                    sampled(
                        template,
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for template in to_in_templates
                ]
            ),
            # beside
            flatten(
                [
                    sampled(
                        _push_beside_template(
                            agent,
                            theme,
                            goal_reference,
                            surface,
                            background,
                            is_right=is_right,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # under
            flatten(
                [
                    sampled(
                        _push_under_template(
                            agent,
                            theme,
                            goal_under,
                            surface,
                            background,
                            is_distal=is_distal,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                ]
            ),
            # in front of, behind
            flatten(
                [
                    sampled(
                        _push_in_front_of_behind_template(
                            agent,
                            theme,
                            goal_reference,
                            surface,
                            background,
                            is_distal=is_distal,
                            is_in_front=is_in_front,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_in_front in BOOL_SET
                ]
            ),
        ),
    )


def _make_go_with_prepositions(num_samples: int = 5, *, noise_objects: int = 0):
    agent = standard_object("agent", THING, required_properties=[ANIMATE])
    goal_object = standard_object("goal_object")
    goal_object_hollow = standard_object(
        "goal_object_hollow", required_properties=[HOLLOW]
    )
    path_object = standard_object(
        "path_object",
        required_properties=[CAN_HAVE_THINGS_RESTING_ON_THEM, HAS_SPACE_UNDER],
    )

    background = immutableset(
        standard_object(f"noise_object_{x}") for x in range(noise_objects)
    )

    return phase1_instances(
        "Go + PP",
        chain(
            # To
            flatten(
                [
                    sampled(
                        _go_to_template(agent, goal_object, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                ]
            ),
            # In
            flatten(
                [
                    sampled(
                        _go_in_template(agent, goal_object_hollow, background),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                ]
            ),
            # Beside
            flatten(
                [
                    sampled(
                        _go_beside_template(
                            agent, goal_object, background, is_right=is_right
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_right in BOOL_SET
                ]
            ),
            # Behind & In Front Of
            flatten(
                [
                    sampled(
                        _go_behind_in_front_template(
                            agent,
                            goal_object,
                            background,
                            is_distal=is_distal,
                            is_behind=is_behind,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_behind in BOOL_SET
                ]
            ),
            # Over & Under
            flatten(
                [
                    sampled(
                        _go_over_under_template(
                            agent,
                            goal_object,
                            background,
                            is_distal=is_distal,
                            is_over=is_over,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_distal in BOOL_SET
                    for is_over in BOOL_SET
                ]
            ),
            # Behind & In Front Of Paths
            flatten(
                [
                    sampled(
                        _go_behind_in_front_path_template(
                            agent,
                            goal_object,
                            path_object,
                            background,
                            is_behind=is_behind,
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_behind in BOOL_SET
                ]
            ),
            # Over & Under Paths
            flatten(
                [
                    sampled(
                        _go_over_under_path_template(
                            agent, goal_object, path_object, background, is_over=is_over
                        ),
                        ontology=GAILA_PHASE_1_ONTOLOGY,
                        chooser=PHASE1_CHOOSER,
                        max_to_sample=num_samples,
                    )
                    for is_over in BOOL_SET
                ]
            ),
        ),
    )


def make_verb_with_dynamic_prepositions_curriculum(
    num_samples: int = 5, *, num_noise_objects: int = 0
):
    return [
        _make_push_with_prepositions(num_samples, noise_objects=num_noise_objects),
        _make_go_with_prepositions(num_samples, noise_objects=num_noise_objects),
    ]
