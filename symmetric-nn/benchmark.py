"""Labelled term-pair benchmark for the symmetric-NN grouping step.

The question the grouping step answers is narrow: given two surface forms
extracted from the same patent corpus, do they name ONE technology that should
occupy one row of the landscape, or two that should stay apart?

Labels are mine and the criterion is the product decision, not linguistic
synonymy. "alkaline electrolyser" and "PEM electrolyser" are close in meaning
and must NOT merge, because an analyst comparing them is the entire point of the
tool. That asymmetry is why this cannot be scored on a general STS benchmark.

Pairs are grouped by the relation that generates them so that per-category
accuracy is visible: a method can be excellent overall and still fail every
abbreviation, and an average hides that.
"""

# (a, b, same_technology)
POSITIVE = {
    "head-truncation": [
        ("leak detection sensor", "hydrogen leak detection sensor"),
        ("optic sensor array", "fibre optic sensor array"),
        ("storage tank", "cryogenic storage tank"),
        ("exchange membrane", "proton exchange membrane"),
        ("pressure vessel", "composite pressure vessel"),
        ("emission sensor", "acoustic emission sensor"),
        ("girth weld", "pipeline girth weld"),
        ("bipolar plate", "electrolyser bipolar plate"),
        ("sensing element", "catalytic sensing element"),
        ("overwrap", "carbon fibre overwrap"),
    ],
    "category-suffix": [
        ("cryogenic storage tank", "cryogenic storage tank assembly"),
        ("electrolyser stack", "electrolyser stack apparatus"),
        ("fuel cell membrane", "fuel cell membrane device"),
        ("hydrogen compressor", "hydrogen compressor unit"),
        ("polymer liner", "polymer liner structure"),
        ("gas diffusion layer", "gas diffusion layer arrangement"),
    ],
    "modifier-drop": [
        ("cryogenic tank", "cryogenic storage tank"),
        ("composite vessel", "composite pressure vessel"),
        ("optical sensor", "optical fibre sensor"),
        ("metal hydride", "metal hydride bed"),
        ("liquefaction plant", "hydrogen liquefaction plant"),
    ],
    "abbreviation": [
        ("PEM", "proton exchange membrane"),
        ("PEM electrolyser", "proton exchange membrane electrolyser"),
        ("SOEC", "solid oxide electrolysis cell"),
        ("MEA", "membrane electrode assembly"),
        ("GDL", "gas diffusion layer"),
        ("BOP", "balance of plant"),
        ("LOHC", "liquid organic hydrogen carrier"),
        ("CCS", "carbon capture and storage"),
    ],
    "spelling-register": [
        ("fibre optic sensor", "fiber optic sensor"),
        ("carbon fibre overwrap", "carbon fiber overwrap"),
        ("sulphur poisoning", "sulfur poisoning"),
        ("catalyser", "catalyzer"),
        ("vapour barrier", "vapor barrier"),
        ("aluminium liner", "aluminum liner"),
    ],
    "true-synonym": [
        ("hydrogen embrittlement", "hydrogen induced cracking"),
        ("boil-off gas", "vaporised gas"),
        ("electrolyser", "electrolyzer"),
        ("water electrolysis", "water splitting"),
        ("vacuum insulated tank", "vacuum jacketed tank"),
        ("refuelling nozzle", "dispensing nozzle"),
    ],
}

NEGATIVE = {
    "shared-modifier": [
        ("hydrogen sensor", "hydrogen tank"),
        ("hydrogen compressor", "hydrogen sensor"),
        ("carbon fibre overwrap", "carbon capture unit"),
        ("cryogenic tank", "cryogenic valve"),
        ("pipeline coating", "pipeline sensor"),
        ("membrane electrode assembly", "membrane filtration unit"),
        ("thermal insulation", "thermal camera"),
        ("gas diffusion layer", "gas detection alarm"),
    ],
    "thing-vs-system": [
        ("pipeline segment", "pipeline monitoring system"),
        ("storage tank", "tank inspection robot"),
        ("fuel cell", "fuel cell test bench"),
        ("weld seam", "weld inspection method"),
        ("catalyst layer", "catalyst regeneration process"),
    ],
    "sibling-technology": [
        ("alkaline electrolyser", "PEM electrolyser"),
        ("solid oxide electrolysis cell", "proton exchange membrane electrolyser"),
        ("metal hydride storage", "compressed gas storage"),
        ("liquid hydrogen storage", "compressed hydrogen storage"),
        ("steam methane reforming", "water electrolysis"),
        ("type III pressure vessel", "type IV pressure vessel"),
        ("acoustic emission sensor", "fibre optic sensor"),
        ("catalytic sensor", "electrochemical sensor"),
    ],
    "component-vs-component": [
        ("catalyst layer", "catalytic converter"),
        ("bipolar plate", "end plate"),
        ("polymer liner", "metal liner"),
        ("anode catalyst", "cathode catalyst"),
        ("inlet valve", "relief valve"),
        ("vacuum jacket", "vacuum pump"),
    ],
    "material-vs-application": [
        ("carbon fibre", "carbon footprint"),
        ("stainless steel", "steel mill"),
        ("nickel catalyst", "nickel mining"),
    ],
    "unrelated": [
        ("cryogenic storage tank", "catalyst layer"),
        ("girth weld", "proton exchange membrane"),
        ("acoustic emission sensor", "polymer liner"),
        ("hydrogen embrittlement", "gas diffusion layer"),
        ("refuelling nozzle", "bipolar plate"),
        ("vacuum jacket", "steam methane reforming"),
    ],
}


def pairs():
    """[(a, b, label, category)] with label True when they are one technology."""
    out = []
    for cat, ps in POSITIVE.items():
        out += [(a, b, True, cat) for a, b in ps]
    for cat, ps in NEGATIVE.items():
        out += [(a, b, False, cat) for a, b in ps]
    return out


def vocabulary():
    """Every distinct surface form, as the grouping step would see them.

    Methods that need a corpus - CSLS needs neighbourhoods, an adaptive
    threshold needs a distribution - get this rather than the pair list, so they
    cannot peek at which pairs are being scored.
    """
    seen = {}
    for a, b, _label, _cat in pairs():
        seen.setdefault(a, None)
        seen.setdefault(b, None)
    return list(seen)


if __name__ == "__main__":
    ps = pairs()
    pos = sum(1 for p in ps if p[2])
    print("pairs: %d  (%d positive, %d negative)" % (len(ps), pos, len(ps) - pos))
    print("vocabulary: %d distinct terms" % len(vocabulary()))
    for name, group in (("POSITIVE", POSITIVE), ("NEGATIVE", NEGATIVE)):
        print("\n%s" % name)
        for cat, items in group.items():
            print("  %-24s %d" % (cat, len(items)))
