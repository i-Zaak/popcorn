import os
import pooch

data_root = os.path.abspath(
        os.path.join(
            os.path.dirname(
                os.path.dirname(__file__)
            ),
            'data'
        )
)

def path(relp):
    return os.path.join(data_root, os.path.normpath(relp))

def fetch_precomputed(url='https://data-proxy.ebrains.eu/api/permalinks/c022e2b1-003b-4647-8245-dd806d14ca23'):
    pooch.retrieve(
        # URL to one of Pooch's test files
        url=url,
        known_hash='8ddb9d01196484462a99a9c4a8146c9a4ca849797b6557eb2bcd487c6d699723',
        fname='Es_stim.npy',
        path=path('interim/precomputed')
    )

def fetch_stim(url='https://data-proxy.ebrains.eu/api/permalinks/161e4828-8860-459e-aa5a-025b1e99aaa0'):
    pooch.retrieve(
        # URL to one of Pooch's test files
        url=url,
        known_hash='fa76cbd206284e08f48504803f09e44f43ccd288764eb5b7bfcec48cead2b3c6',
        fname='stim.npz',
        path=path('interim')
    )
