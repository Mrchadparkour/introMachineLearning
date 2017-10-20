from sklearn import tree

#define classifiers
features = [[140, 1], [130, 1],[150, 0], [170, 0]]  #training data, 1 = smooth
                                                    #0 = bumpy skin

labels = ["apple", "apple", "orange", "orange"]     #outputs for features

clf = tree.DecisionTreeClassifier()    #the learning algorithm
clf = clf.fit(features, labels)        #inputs training data

print (clf.predict([[150, 0]]))          #prints its prediction
