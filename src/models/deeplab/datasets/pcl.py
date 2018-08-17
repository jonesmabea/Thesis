import argapse 
import sklearn
from collections import Counter
import pydriver

# function the vocabularies will use to create category storages
def storageGenerator(dims, category):
    sto = pydriver.detectors.vocabularies.Storage(dims, category,
        preprocessors=[],
        regressor=sklearn.neighbors.KNeighborsRegressor(n_neighbors=1),
        )
    return sto
# function the detector will use to create vocabularies
def vocabularyGenerator(dimensions, featureName):
#     ada_clf=sklearn.ensemble.AdaBoostClassifier(n_estimators=75)
#     svm_clf = sklearn.svm.SVC(probability=True)
#     rand_clf =sklearn.ensemble.RandomForestClassifier()
#     voting_clf = sklearn.ensemble.VotingClassifier(estimators=[('ada', ada_clf),('svc',svm_clf),('rf',rand_clf)],voting='soft')
#     bag_clf = sklearn.ensemble.BaggingClassifier(n_estimators=65,n_jobs=-1,max_samples=10)

    voc = pydriver.detectors.vocabularies.Vocabulary(
        dimensions,
        preprocessors=[
            sklearn.cluster.MiniBatchKMeans(n_clusters=50, batch_size=10),#, max_iter=100),
            ],
        classifier=sklearn.ensemble.AdaBoostClassifier(n_estimators=120),
        storageGenerator=storageGenerator,
        balanceNegatives=True,
        )
    return voc





def predict_context(detection,thresh=False):
    contexts = [det['category'].decode('UTF-8') for det in detection]
    c = Counter(contexts)
    cls,num = c.most_common(1)[0]
    tot = sum(c.values())
    prob = num/tot
    if cls=='negative':
        cls='non-urban'
    if thresh:
         if prob<0.6:
            cls='urban'
            prob=1-prob
    return cls,prob

USE_IMAGE_COLOR = False
reader = pydriver.datasets.kitti.KITTIObjectsReader(path)
reconstructor = pydriver.preprocessing.LidarReconstructor(
    useImageColor=USE_IMAGE_COLOR,
    removeInvisible=True,
    )

SHOT_RADIUS = 2.0
preprocessor = pydriver.preprocessing.Preprocessor(reconstructor)
keypointExtractor = pydriver.keypoints.ISSExtractor(salientRadius=0.25, nonMaxRadius=0.25)
featureExtractor = pydriver.features.SHOTColorExtractor(shotRadius=SHOT_RADIUS, fixedY=-1.0)
featureTypes = [('myfeature', featureExtractor.dims),]

# initialize detector that will perform learning and recognition
detector = pydriver.detectors.Detector(featureTypes, vocabularyGenerator=vocabularyGenerator)
detector=detector.load('/mnt/storage/home/ja17618/scratch/models/research/deeplab/datasetsworking_model')

path = "/mnt/storage/home/ja17618/scratch/DATA_DIR/training/"
f_lidar = glob.glob(os.path.join(path, 'velodyne', '*.bin'))
pcl =f_lidar[0]
raw_lidar = np.fromfile(pcl,dtype=np.float32).reshape((-1,4))

#scene = preprocessor.process(frame)
keypointCloud = keypointExtractor.getKeypointCloud(raw_lidar)


# extract keypoints and features for the whole scene
fkeypoints, features = featureExtractor.getFeatures(scene, keypointCloud)
    # perform recognition on extracted features

detections = detector.recognize({'myfeature': (fkeypoints,features)})