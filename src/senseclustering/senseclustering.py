import numpy as np
import wikipedia
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cosine, pdist
from scipy.stats import entropy
from utils import load_vecs
from nltk.corpus import stopwords
from collections import defaultdict

def __clean_summary(summary, tag):
    return set([word.lower() for word in summary.split(" ") if word not in stopwords.words("english") and word != tag])


def get_wikipedia_senses(tag):
    try:
        page = wikipedia.page(tag)
        senses = {tag: __clean_summary(page.summary, tag)}
    except wikipedia.DisambiguationError as e:
        senses = {}
        for pagetitle in str(e).split("\n")[1:]:
            if pagetitle.lower() != tag:
                try:
                    senses[pagetitle] = __clean_summary(wikipedia.page(pagetitle, auto_suggest=False).summary, tag)
                except Exception:
                    pass
    except Exception as e:
        senses = {tag: "EXCEPTION: " + str(e)}
    return (tag, senses)


def get_sense_context(tag, cooccs, vectors, context_size=20):
    context = np.array([context_tag for (context_tag, occ) in cooccs[tag][:context_size]])
    X = np.array([vectors[context_tag] for context_tag in context])
    return context, linkage(pdist(X, metric=lambda x,y: cosine(x,y) / 2.0), method='ward')


def get_sense_counts(dist):
    sense_counts = defaultdict(int)
    for tag in cooccs_dict_bc.value.keys():
        try:
            context, clustering = get_sense_context(tag, cooccs_dict_bc.value, vectors_bc.value)
            clusterids = fcluster(clustering, dist, criterion='distance')
            sense_counts[len(set(clusterids))] += 1
        except ValueError as e:
            print(e)
            print(tag, len(cooccs_dict[tag]), cooccs_dict[tag])
            pass
            # this only happens when the context of a tag contains only one other tag. sad little tag.
    return (dist, sense_counts)


def get_cluster_representative(tag, cluster, vectors):
    tag_vec = vectors[tag]
    representative = None
    max_relatedness = 0.0
    for context_tag in cluster:
        if 1 - cosine(tag_vec, vectors[context_tag]) > max_relatedness:
            max_relatedness = 1 - cosine(tag_vec, vectors[context_tag])
            representative = context_tag
    return representative


def get_senses(dist):
    senses = {}
    for tag in cooccs_dict_bc.value.keys():
        try:
            context, clustering = get_sense_context(tag, cooccs_dict_bc.value, vectors_bc.value)
            clusterids = fcluster(clustering, dist, criterion='distance')
            for i in set(clusterids):
                cluster = context[clusterids == i]
                senses[get_cluster_representative(tag, cluster, vectors_bc.value)] = set(cluster)
        except ValueError:
            pass
    return senses


def evaluate_clustering(clustersenses, wikisenses):
    precisions = []
    recalls = []
    for tag in clustersenses:
        precision = 0.0
        recall = 0.0
        if tag in wikisenses:
            cluster_meanings = clustersenses[tag]
            wiki_meanings = wikisenses[tag]
            matched_wiki_meanings = []
            for cmean in cluster_meanings:
                for wmean in wiki_meanings:
                    if len(cluster_meanings[cmean].intersection(wiki_meanings[wmean])) > 0:
                        precision += 1.0
                        matched_wiki_meanings.append(wmean)
            recall += len(matched_wiki_meanings)
        precision /= len(clustersenses[tag])
        recall /= len(wikisenses[tag])
        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls)



# this adds some variety to our game <3
vectors = load_vecs("scratch/luzian/resultVectors/glove/tagging/bibsonomy/vec_complete_file_glove_dim100")
vectors_bc = sc.broadcast(vectors)


cooccs = sc.textFile("data/bibsonomy/tas/nospam2015_tag_top10k_min5user_min10docs_edgelist", 100)\
        .map(lambda line: line.strip().split("\t"))\
        .map(lambda (tag1, tag2, coocc): (tag1, (tag2, int(coocc))))\
        .groupByKey()\
        .map(lambda (tag, coocclist): (tag, sorted(coocclist, key=lambda x: x[1], reverse=False)))\
        .cache()


# take care: This takes a long time!
wikisenses = cooccs.map(lambda x: x[0]).map(get_wikipedia_senses).cache()
wikisenses_dict = defaultdict(int, wikisenses.collect())
wikisenses_counts = defaultdict(int)
for tag in wikisenses_dict:
    wikisenses_counts[len(wikisenses_dict[tag])] += 1


cooccs_dict = dict(cooccs.filter(lambda x: len(x[1]) > 1).collect())
cooccs_dict_bc = sc.broadcast(cooccs_dict)


dist = sc.parallelize(list(np.arange(0, 2.05, 0.05)))


dist_sense_counts = dict(dist.map(get_sense_counts).collect())
dist_sense_counts = {round(key, 2): value for key, value in dist_sense_counts.items()}
kldivs = defaultdict(float)
max_i = 30
for dist_value in dist_sense_counts:
    pk = [dist_sense_counts[dist_value][i] for i in range(max_i)]
    qk = [wikisenses_counts[i] for i in range(max_i)]
    kldivs[dist_value] = entropy(pk, qk)

for key in sorted(kldivs):
    print(key, kldivs[key])

print("Minimum IG: " + str(key))
