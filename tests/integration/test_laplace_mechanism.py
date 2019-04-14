
from empirical_privacy.laplace_mechanism import GenSampleLaplaceMechanism


def test_gen_sample_laplace_constructor():
    gs = GenSampleLaplaceMechanism(
        dataset_settings = {
            'dimension': 3,
            'epsilon': 0.1,
            'delta': 0
            },
        generate_positive_sample=True,
        random_seed=0
        )


    X, y = gs.gen_sample(0)
    assert X.shape == (1,3)
    assert y ==1