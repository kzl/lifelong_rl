import copy
import itertools
import random


def set_dict_key(dict, path, value):
    if len(path) == 1:
        dict[path[0]] = value
    else:
        set_dict_key(dict[path[0]], path[1:], value)


def generate_variants(base_variant, sweep_values, num_seeds=1):
    variants = []
    for _ in range(num_seeds):
        for params in itertools.product(*[s for s in sweep_values.values()]):
            variant = copy.deepcopy(base_variant)
            for i, loc in enumerate(sweep_values.keys()):
                path = loc.split('/')
                set_dict_key(variant, path, params[i])
            variant['seed'] = random.randint(0, int(1e8))
            variants.append(variant)
    return variants
