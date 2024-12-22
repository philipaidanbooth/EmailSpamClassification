'''knn.py
K-Nearest Neighbors algorithm for classification
Philip Booth
CS 251: Data Analysis and Visualization
Fall 2024
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classifier import Classifier

class KNN(Classifier):
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Call superclass constructor
        '''

        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None
        self.num_classes = num_classes

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''
        self.exemplars = data
        self.classes = y
        



    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.
        '''
        num_test_samps = data.shape[0]
        y_pred = np.zeros(num_test_samps)
        unique_labels = np.unique(self.classes)
        
        for i in range(num_test_samps):
            sample = data[i]
            distances = np.linalg.norm(self.exemplars - sample, axis=1)
            closest_indices = np.argsort(distances)[:k]

            class_votes = np.zeros(len(unique_labels))

            for j in range(k):
                label = self.classes[closest_indices[j]]
                index = np.where(unique_labels == label)[0][0]
                class_votes[index] += 1
            
            max_vote = np.argmax(class_votes)
            y_pred[i] = unique_labels[max_vote]

        return y_pred




    def predict_mahnattan(self, data, k):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.
        '''
        num_test_samps = data.shape[0]
        y_pred = np.zeros(num_test_samps)
        unique_labels = np.unique(self.classes)
        
        for i in range(num_test_samps):
            sample = data[i]
            distances = np.sum(np.abs(self.exemplars - sample), axis=1)
            closest_indices = np.argsort(distances)[:k]

            class_votes = np.zeros(len(unique_labels))

            for j in range(k):
                label = self.classes[closest_indices[j]]
                index = np.where(unique_labels == label)[0][0]
                class_votes[index] += 1
            
            max_vote = np.argmax(class_votes)
            y_pred[i] = unique_labels[max_vote]

        return y_pred
    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use either the Okabe & Ito or one of the Petroff color palettes: https://github.com/proplot-dev/proplot/issues/424
        - Wrap your colors list as a `ListedColormap` object (already imported above) so that matplotlib can parse it.
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        cmaplight = ListedColormap(['#333333','#BBBBBB'])
        cmapdark = ListedColormap(['#eeeedd','#443355' ])
        

        sampling_vector = np.linspace(-40, 40, n_sample_pts)
        x, y = np.meshgrid(sampling_vector, sampling_vector)
        
        x_flattened = x.flatten()
        y_flattened = y.flatten()   
        x_reshaped = x_flattened.reshape(-1, 1)
        y_reshaped = y_flattened.reshape(-1, 1)

        # print(x.shape)
        xy_stacked = np.hstack((x_reshaped, y_reshaped))
        # print(xy_stacked.shape)

        y_predicted = self.predict(xy_stacked, k)
        # print(y_predicted.shape)
        y_predicted_reshaped = y_predicted.reshape(n_sample_pts, n_sample_pts)

        plt.pcolormesh(x, y, y_predicted_reshaped, cmap=cmaplight)
        plt.scatter(self.exemplars[:, 0], self.exemplars[:, 1], c=self.classes, cmap=cmapdark)
        plt.title('KNN Classification with Training Data')
        plt.xlabel('X ')
        plt.ylabel('Y ')
        plt.show()

