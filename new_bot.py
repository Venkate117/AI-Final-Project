# required libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import export_graphviz
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    '''
    This is the main function, that will be called at the time of execution.
    '''
    # 1. Data Loading

    # loading the data (both train and test dataset)
    training = pd.read_csv('Training.csv')
    testing  = pd.read_csv('Testing.csv')

    # cols refer to the columns in the training dataset, these columns are the features.
    cols     = training.columns
    # there are 133 columns, of which 132 are taken as input features, and 1 col is taken as output.
    # all the columns except the last one column('prognosis') will be used as inputs features.
    cols     = cols[:-1]
    # training inputs
    x        = training[cols]
    # training outputs.
    y        = training['prognosis']
    y1       = y

    # this is to group by the names in the column 'prognosis' and the max will be used.
    max_data_in_dataset = training.groupby(training['prognosis']).max()

    # 2. Data Preprocessing

    #mapping strings to numbers
    # As we are using decision trees, the decision condition can be one of the features crossing a threshold 
    # value so it is important to have all the features as integers.
    le = preprocessing.LabelEncoder()
    # object of label encoder.
    le.fit(y)
    y = le.transform(y)


    # 3. Data splitting

    # splitting of x,y data into train and test.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=26)
    testx    = testing[cols]
    testy    = testing['prognosis']  
    testy    = le.transform(testy)

    # 4. Decision Tree Classifier Model building

    clf1  = DecisionTreeClassifier()
    # fitting the model with training features and labels.
    clf = clf1.fit(x_train,y_train)

    # Based on the Gini Impurity and other factors, Decision Tree Classifier
    # might consider only a few features dominating.
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = cols

    # to print the features that are considered as important by decision_tree_classifier.
    #for f in range(10):
    #    print("%d. feature %d - %s (%f)" % (f + 1, indices[f], features[indices[f]] ,importances[indices[f]]))

    print("=================================================")
    print("Hi, This is an interactive chatbot to help you with medical diagnosis")
    print("Please reply Yes or No for the following symptoms") 
    print(" ")

    #  helper function to map the output of the model to the name of the disease.
    def print_disease_to_user(node):
        '''
        This methods takes node as input and gives the corresponding disease
        by performing inverse_transform of Label encoder.
        '''
        #print(node)
        node = node[0]
        #print(len(node))
        val  = node.nonzero() 
        #print(val)
        disease = le.inverse_transform(val[0])
        return disease

    # helper function to display the symptoms present.
    def decision_tree_bot(tree, feature_names):
        '''
        This method takes the tree object and the feature_names, this is will be used to collect
        the symptoms of the user and predict the possible disease the user is suffering from.
        '''
        tree_ = tree.tree_
        #print(tree_)
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        #print("def tree({}):".format(", ".join(feature_names)))
        symptoms_present = []
        # helper function for searching the node of the decision tree classifier.
        def binary_search_in_tree(node, depth):
            '''
            This method does the binary search to reach the node of the decision tree classifer,
            based on the depth of the tree.
            '''
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                print(name + " ?")
                ans = input()
                ans = ans.lower()
                if ans == 'yes':
                    val = 1
                else:
                    val = 0
                if  val <= threshold:
                    binary_search_in_tree(tree_.children_left[node], depth + 1)
                else:
                    symptoms_present.append(name)
                    binary_search_in_tree(tree_.children_right[node], depth + 1)
            else:
                present_disease = print_disease_to_user(tree_.value[node])
                print( "You may have " +  present_disease )
                red_cols = max_data_in_dataset.columns 
                symptoms_given = red_cols[max_data_in_dataset.loc[present_disease].values[0].nonzero()]
                print("symptoms present  " + str(list(symptoms_present)))
                print("symptoms given "  +  str(list(symptoms_given)) )  
                confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
                print("confidence level is " + str(confidence_level))

        binary_search_in_tree(0, 1)

    decision_tree_bot(clf,cols)

# execution starts here.
if __name__ == "__main__":
    main()