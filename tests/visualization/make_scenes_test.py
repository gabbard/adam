from adam.visualization.make_scenes import SceneCreator
from adam.curriculum.phase1_curriculum import GAILA_PHASE_1_CURRICULUM
from adam.visualization.utils import Shape
from typing import Tuple


def test_scenes_creation() -> Tuple[float, float, float]:
    for i, scene in enumerate(SceneCreator.create_scenes(GAILA_PHASE_1_CURRICULUM)):
        for obj, _ in scene.items():
            # only interested in rendering geons
            if (
                obj.geon is None
                or SceneCreator.cross_section_to_geo(obj.geon.cross_section)
                == Shape.IRREGULAR
            ):
                continue
            point = SceneCreator.random_position()
        # automated test shouldn't go through every single scene
        if i > 5:
            break
    return point
