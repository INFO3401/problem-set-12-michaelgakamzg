import pandas as pandas
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.neighbors import kNewighborsClassifier
from skelarn.cluster imprt kMeans

def loadData (datafile):
    with open(datafile, 'r', encodings = 'latin1') as csvfile:
        data = pd.read_csv(csvfile)

    print (data.columns.values)

    return data

def runKNN(dataset, prediction, ignore):
    X = dataset.drop(columns=[prediction, ignore])
    Y = dataset[prediction].values

    x_train, x_test, y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 1, stratify = Y)

    knn = kNeighborsClassifier(n_neighbors=5)

    knn.fit(x-train, y_train)

    score = knn.score(x_test, y_test)
    print("Predicts" + prediction + "with" + str(score) + "accuracy")
    print("Chance is: " + str(1.0/len(dataset.groupby(prediction))))


    return knn

def classifyPlayer(targetRow, data, model, prediction, ignore):
    x = targetRow.drop(columns=[prediction,ignore])

    neighbors = model.kneighbors(X, n_neighbors = 5, return_distance=False)

    for neigbor in neighbors[0]:
        print(data.iloc[neighbor])

def runKMeans(dataset, ignore):
    X = dataset.drop(columns=ignore)

    kmeans = KMeans(n_clusters=5)

    kmeans.fit(X)

    dataset['cluster'] = pd.Series(kmeans.predict(X), index=dataset.index)

    scatterMatrix = sns.pairplot(dataset.drop(columns=ignore), hue='cluster', palette='Set2')

    scatterMatrix.savefig("kmeanClusters.png")

    return kmeans

nbaData = loadData("nba_2013_clean.csv")
knnModel = runKNN(nbaData, "pos", "player")
classifyPlayer(nbaData.loc[nbaData['player'] == 'LeBron James'], nbaData, knnModel, 'pos','player')

kmeansmodel = runMeans(nbaData, ['pos','player'])

