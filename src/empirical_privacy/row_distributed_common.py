import numpy as np


def gen_random_subset(n, n_part, seed_str, i_exclude=[], i_include=[]):
    """
    Generate a random subset of indices.
    Parameters
    ----------
    n: int
        largest possible index
    n_part: int
        how many indices to select
    seed_str: str
        seed string from for PRNG
    i_exclude: int or list of ints
        which indices must be excluded
    i_include: int or list of ints
        which indices must be included

    Returns
    -------
    rtv: list
        list of indexes
    """
    S = set(range(n))
    if not hasattr(i_exclude,'__iter__'):
        i_exclude=[i_exclude]
    if not hasattr(i_include,'__iter__'):
        i_include=[i_include]

    if i_exclude:
        S -= set(i_exclude)
    S = list(S)
    seed_val = hash(seed_str) % 4294967296  # max seed +1
    np.random.seed(hash(seed_val))
    np.random.shuffle(S)  # this modifies S
    rtv = []
    if i_include:
        n_part -= len(i_include)
        rtv = i_include
    rtv = rtv+S[0:(n_part)]
    np.random.shuffle(rtv)
    return rtv


def gen_attacker_and_defender_indices(n, part_fraction, doc_ind, seed):
    n_part = int(round(n * part_fraction))

    I_defender_without_doc = gen_random_subset(n, n_part, seed +  'without',
                                             i_exclude=doc_ind)
    I_defender_with_doc = gen_random_subset(n, n_part, seed + 'with',
                                          i_exclude=doc_ind)
    I_defender_with_doc[0] = doc_ind
    I_attacker = gen_random_subset(n, n_part, seed+'adversary')

    I_defender_with_doc = sorted(I_defender_with_doc)
    I_defender_without_doc = sorted(I_defender_without_doc)
    I_attacker = sorted(I_attacker)

    return {'I_defender_with':I_defender_with_doc,
            'I_defender_without':I_defender_without_doc,
            'I_attacker':I_attacker
    }