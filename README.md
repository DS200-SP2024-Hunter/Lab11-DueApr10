# Lab Assignment 11, Due on [Canvas](https://psu.instructure.com/courses/2306358/assignments/16035342), Apr. 10 at 11:59pm
## Implement a Naive Bayes Classifier

**On this and all labs for the rest of the semester: When you have a choice between the [datascience package](https://www.data8.org/datascience/) and the [pandas library](https://pandas.pydata.org/docs/), you are free to use either method to complete your work.**

The main objective of today's lab is to use Bayes Theorem and some simplifying assumptions to create a classfier, known as a naive Bayes classifier, for predicting the category of wines based on several quantitative measurements.  


**Objective**:  Use the `wine` dataset from [Section 17.4](https://inferentialthinking.com/chapters/17/4/Implementing_the_Classifier.html) to train a naive Bayes classifier to predict the type of wine from quantitative measurements.  We'll make two simplifying assumptions:  conditional independence of features and the quantitative variables are all normally distributed.  

The basic idea of a naive Bayes classifier is described in [this document](https://github.com/DS200-SP2022-Hunter/Week12-Apr05/blob/main/NaiveBayes.pdf).

**Your assignment** is as follows:

1. Load the Jupyter notebook for [Section 17.4](https://inferentialthinking.com/chapters/17/4/Implementing_the_Classifier.html) of the textbook from GitHub as you've done in the past. You should change the `path_data` object so that it points to the URL where the dataset can be found: `https://raw.githubusercontent.com/data-8/textbook/main/assets/data/`

2. As in Section 17.4.3, read the dataset from the `wine.csv` file as a `Table` object and give it the name `wine`.  Do NOT convert this dataset to a new one with only two classes of wine.  We will keep all three classes for this assignment.

3. As you will see from the [naive Bayes classifier document](https://github.com/DS200-SP2022-Hunter/Week12-Apr05/blob/main/NaiveBayes.pdf), you'll need the means and standard deviations of the quantitative variables for each wine class.  You can get them using the `group` method, which allows for an optional function to be used on the values in each group:
```
wineMeans = wine.group("Class", np.mean)
wineSDs = wine.group("Class", np.std)
```

4. You'll also notice from the [naive Bayes classifier document](https://github.com/DS200-SP2022-Hunter/Week12-Apr05/blob/main/NaiveBayes.pdf) that you'll need prior probabilities of the three wine categories.  For this purpose, instead of assuming that each class is equally probable a priori, let us suppose that our sample of wines is representative of the population we care about, so that the prior probabilities may be taken to be the overall proportions of the three classes.  We can calculate these proportions by using the `group` method to count how many are in each class:
```
prior = wine.group("Class").column('count') / wine.num_rows
prior
```

5. To calculate conditional probabilities of the form P(X | C=1) will require the formula for the normal curve with mean and SD given by corresponding values from the `wineMeans` and `wineSDs` objects.  The function that evaluates the normal curve is `norm.pdf` from the `scipy.stats` library.  Verify that the following code returns the value of the normal curve evaluated at 2 when the mean is 2 and the SD is 1, which should equal the square root of (1/2&pi;):
```
from scipy import stats
stats.norm.pdf(2, 2, 1)
```

6.  We can now begin to collect the pieces.  Suppose we'd like to focus on the wine in row r.  Here is a function that takes r and the name of a variable, then returns the three probabilities P(X | C=1), P(X | C=2), and P(X | C=3), where X is the value of the selected variable in the selected row:
```
def prob(r, variable):
  means = wineMeans.column(variable + ' mean') # This is an array with 3 elements
  SDs = wineSDs.column(variable + ' std') # This is an array with 3 elements
  X = wine.column(variable).item(r) # This is a single value
  return stats.norm.pdf(X, means, SDs) # This is an array with 3 elements
```
 
7.  Let us now focus on the variables `Ash` and `Alcohol`.  Suppose we want to predict the class of the wine in row 0.  We need to combine the three probabilities found by the `prob` function we just defined (three for each of `Ash` and `Alcohol`) together with the prior probabilities to obtain the three terms in the denominator of Bayes' Theorem in the [naive Bayes classifier document](https://github.com/DS200-SP2022-Hunter/Week12-Apr05/blob/main/NaiveBayes.pdf):
```
prob(0, 'Ash') * prob(0, 'Alcohol') * prior
```
According to Bayes' theorem, the probabilities of the three classes are determined by our naive Bayes classifier by dividing each of the three values above by their sum:
```
bayes = prob(0, 'Ash') * prob(0, 'Alcohol') * prior
bayes / np.sum(bayes)
```
Which class has the greatest probability according to our classifier?  Does this agree with the true class according to the 0th row of the `wine` table?

8. The `numpy` function called `argmax` returns the index (0, 1, or 2) of the maximum value of a 3-item array.  If we add 1 to this index, we now have our classifier.  For instance, let's classify the wine in row 10:
```
bayes = prob(10, 'Ash') * prob(10, 'Alcohol') * prior
1 + np.argmax(bayes / np.sum(bayes))

# Strictly speaking, we could have just used 1 + np.argmax(bayes).  But dividing by the sum makes the connection with Bayes' Theorem more obvious.
```
9.  We can now apply this naive Bayes classifier to each row in the `wine` table using a `for` loop.  **Caution!** This is dangerous because we are using the same dataset (the `wine` table) to train the classifier as we are now using to test it.  Ordinarily we should, for example, split the dataset into multiple parts, and use only one part for testing at a time while using the others for training; this procedure is called cross-validation.  In the extreme, we can test each row after training the classifier on all the other rows; this is called leave-one-out cross-validation.  However, for the sake of simplicity on this assignment we will use the whole dataset for training AND for testing.  Here is a for loop that puts all of the previous steps together:
```
predictions = make_array() # Start with an empty array

# Now repeat our earlier procedure on each row
for r in np.arange(wine.num_rows): 
  bayes = prob(r, 'Ash') * prob(r, 'Alcohol') * prior
  predictions = np.append(predictions, 1 + np.argmax(bayes / np.sum(bayes)))

# Finally, create a new table that adds a column for predicted class values
results = wine.with_columns('Predicted Class', predictions)
```

10. We can determine how many times each of the possible actual class / predicted class pairs occurred using the `pivot` method.  Using this information, calculate what percent of the predictions our naive Bayes classifier, based on `Ash` and `Alcohol`, got correct:
```
results.pivot('Class', 'Predicted Class')
```

11. You can create a scatterplot of `Ash` vs. `Alcohol` with points colored according to `Class` using `wine.scatter('Ash', 'Alcohol', group='Class')`.  Search for a different pair of variables that seems to separate the three groups better than `Ash` and `Alcohol`.  Include a scatterplot of these two variables with points colored according to `Class`, then implement the naive Bayes classifier using those two variables.  Check that they give a better percentage of correct predictions than in Step 10.

12.  Modify your naive Bayes classifier so that it uses more than two variables.  Search for combinations of three of more variables that achieve a correct prediction rate of higher than 90%.  How high can you achieve?  Include your code that shows which variables you use as well as your pivot table demonstrating your classifier's prediction performance.  (Something to consider:  Is it always the case that more variables lead to better predictions?)

12.  Finally, make sure that your Jupyter notebook only includes code and text that is relevant to this assignment.  For instance, if you have been completing this assignment by editing the original code from Section 13.2, make sure to delete the material that isn't relevant before turning in your work.

When you've completed this, you should select "Print" from the File menu, then save to pdf using this option.  The pdf file that you create in this way is the file that you should upload to Canvas for grading.  If you have trouble with this step, try selecting the "A3" paper size from the advanced options and making sure that your colab is zoomed out all the way (using ctrl-minus or command-minus).  As an alternative, you can create the pdf within your google drive space and then download it from there.  Here's a [Jupyter noteboook](https://github.com/DS200-SP2022-Hunter/Week11-Mar29/blob/main/convert_pdf.ipynb) shared by Xinyu Dou that creates the pdf within the google drive space (you may need to modify it depending on your directory names and the name of your lab file).

**Here is a [short jupyter notebook](https://github.com/DS200-SP2024-Hunter/Lab09-DueMar27/blob/main/convert_pdf.ipynb) that shows how you can create a pdf document from a .ipynb file directly in your google drive space. Then you can download it to turn in.**

