"""IOI source prompts, paraphrases, and greater-than control prompts."""

from dataclasses import dataclass
from typing import List


@dataclass
class IOIPrompt:
    text: str
    io_name: str   # Indirect Object (model should predict this)
    s_name: str    # Subject (repeated name, distractor)


# ---------------------------------------------------------------------------
# IOI source set – 10 prompts, varied name pairs, same syntactic template
# ---------------------------------------------------------------------------

IOI_SOURCE: List[IOIPrompt] = [
    IOIPrompt("When Mary and John went to the store, John gave the bag to", "Mary", "John"),
    IOIPrompt("When Tom and Alice went to the park, Alice gave the ball to", "Tom", "Alice"),
    IOIPrompt("When Bob and Carol went to the office, Carol gave the file to", "Bob", "Carol"),
    IOIPrompt("When Sam and Peter went to the school, Peter gave the book to", "Sam", "Peter"),
    IOIPrompt("When Emma and David went to the market, David gave the gift to", "Emma", "David"),
    IOIPrompt("When Lucy and James went to the gym, James gave the key to", "Lucy", "James"),
    IOIPrompt("When Anna and Mark went to the bank, Mark gave the card to", "Anna", "Mark"),
    IOIPrompt("When Kate and Paul went to the mall, Paul gave the bag to", "Kate", "Paul"),
    IOIPrompt("When Jane and Chris went to the cafe, Chris gave the cup to", "Jane", "Chris"),
    IOIPrompt("When Rose and Steve went to the lab, Steve gave the pen to", "Rose", "Steve"),
]

# ---------------------------------------------------------------------------
# Easy paraphrases – same structure, fresh name pairs (check 02)
# ---------------------------------------------------------------------------

IOI_PARAPHRASE: List[IOIPrompt] = [
    IOIPrompt("When Grace and Mike went to the garden, Mike gave the flower to", "Grace", "Mike"),
    IOIPrompt("When Lily and Dan went to the library, Dan gave the note to", "Lily", "Dan"),
    IOIPrompt("When Eva and Leo went to the hall, Leo gave the ticket to", "Eva", "Leo"),
    IOIPrompt("When Nina and Carl went to the yard, Carl gave the tool to", "Nina", "Carl"),
    IOIPrompt("When Vera and Jack went to the field, Jack gave the ring to", "Vera", "Jack"),
]

# ---------------------------------------------------------------------------
# Greater-than control prompts – year completion, unrelated to IOI (check 03)
# Template from Hanna et al. 2023; D = logit(later decade) - logit(earlier decade)
# ---------------------------------------------------------------------------

@dataclass
class GTPrompt:
    text: str
    high_tok: str   # two-digit string for a year AFTER the start year
    low_tok: str    # two-digit string for a year BEFORE the start year


GT_CONTROL: List[GTPrompt] = [
    GTPrompt("The war lasted from the year 1815 to the year 18", "20", "10"),
    GTPrompt("The treaty was signed between the year 1823 to the year 18", "30", "20"),
    GTPrompt("The expedition ran from the year 1847 to the year 18", "50", "40"),
    GTPrompt("The building was constructed from the year 1862 to the year 18", "70", "60"),
    GTPrompt("The reign lasted from the year 1878 to the year 18", "80", "70"),
]
