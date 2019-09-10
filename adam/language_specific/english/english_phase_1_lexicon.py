from adam.language.dependency.universal_dependencies import PROPER_NOUN, NOUN, VERB
from adam.language.lexicon import LexiconEntry, LexiconProperty
from adam.language.ontology_dictionary import OntologyLexicon
from adam.ontology.phase1_ontology import (
    MOM,
    BALL,
    TABLE,
    PUT,
    PUSH,
    BOOK,
    HOUSE,
    CAR,
    WATER,
    JUICE,
    CUP,
    BOX,
    CHAIR,
    HEAD,
    MILK,
    HAND,
    TRUCK,
    DOOR,
    HAT,
    COOKIE,
    DAD,
    BABY,
    DOG,
    BIRD,
    GO,
    COME,
    TAKE,
    EAT,
    GIVE,
    TURN,
    SIT,
    DRINK,
    FALL,
    THROW,
    MOVE,
    JUMP,
    HAVE,
    ROLL,
    FLY,
)

MASS_NOUN = LexiconProperty("mass-noun")

GAILA_PHASE_1_ENGLISH_LEXICON = OntologyLexicon(
    (
        (MOM, LexiconEntry("Mom", PROPER_NOUN)),
        (BALL, LexiconEntry("ball", NOUN)),
        (TABLE, LexiconEntry("table", NOUN)),
        (PUT, LexiconEntry("put", VERB)),
        (PUSH, LexiconEntry("push", VERB)),
        (BOOK, LexiconEntry("book", NOUN)),
        (HOUSE, LexiconEntry("house", NOUN)),
        (CAR, LexiconEntry("car", NOUN)),
        (WATER, LexiconEntry("water", NOUN, [MASS_NOUN])),
        (JUICE, LexiconEntry("juice", NOUN, [MASS_NOUN])),
        (CUP, LexiconEntry("cup", NOUN)),
        (BOX, LexiconEntry("box", NOUN)),
        (CHAIR, LexiconEntry("chair", NOUN)),
        (HEAD, LexiconEntry("head", NOUN)),
        (MILK, LexiconEntry("milk", NOUN, [MASS_NOUN])),
        (HAND, LexiconEntry("hand", NOUN)),
        (TRUCK, LexiconEntry("truck", NOUN)),
        (DOOR, LexiconEntry("door", NOUN)),
        (HAT, LexiconEntry("hat", NOUN)),
        (COOKIE, LexiconEntry("cookie", NOUN)),
        (DAD, LexiconEntry("Dad", PROPER_NOUN)),
        (BABY, LexiconEntry("bird", NOUN)),
        (DOG, LexiconEntry("dog", NOUN)),
        (BIRD, LexiconEntry("bird", NOUN)),
        (GO, LexiconEntry("go", VERB)),
        (COME, LexiconEntry("come", VERB)),
        (TAKE, LexiconEntry("take", VERB)),
        (EAT, LexiconEntry("eat", VERB)),
        (GIVE, LexiconEntry("give", VERB)),
        (TURN, LexiconEntry("turn", VERB)),
        (SIT, LexiconEntry("sit", VERB)),
        (DRINK, LexiconEntry("drink", VERB)),
        (FALL, LexiconEntry("fall", VERB)),
        (THROW, LexiconEntry("throw", VERB)),
        (MOVE, LexiconEntry("move", VERB)),
        (JUMP, LexiconEntry("jump", VERB)),
        (HAVE, LexiconEntry("have", VERB)),
        (ROLL, LexiconEntry("roll", VERB)),
        (FLY, LexiconEntry("fly", VERB)),
    )
)
