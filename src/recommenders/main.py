# %%
from simple import *
from contentBased import *
from collaborativeBased import *
from hybrid import *

# %% [markdown]
# ## Simple Recommender
simpleRec = SimpleRecommender()
simpleRec.build_chart('Romance').head(10)
simpleRec.build_chart('Mystery').head(20)

# %% [markdown]
# ## Content-based Recommender
contentRec = ContentBasedRecommender()
contentRec.prepareMetadataBased()
contentRec.improved_recommendations('The Dark Knight').head(10)


# %% [markdown]
# ## Collaborative-based Recommender
collabRec = CollaborativeBasedRecommender()

# %% [markdown]
# ## Hybrid Recommender
hybridRec = HybridRecommender()
hybridRec.hybrid(1, 'Mean Girls')

