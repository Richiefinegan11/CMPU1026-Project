import pandas as pd
import matplotlib.pyplot as plt


DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

PERCENT = 75


def get_data(url):
    """
    Here we read in the data from the web. We then create a dataframe, where we add
    headers to columns in the dataframe, so we can easily target the columns and
    and manipulate the data. We give any discrete attributes a numeric weight by
    using built in functions to count home many time it appears in the list.
    """

    # read in csv file with pandas module
    data_frame = pd.read_csv(url)

    # add column names to the data
    data_frame.columns = ['age', 'workclass', 'fnglwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                          'capital-loss', 'hours-per-week', 'native-country', 'salary']
    # print(data_frame.head())

    # remove unnecessary columns from dataframe
    data_frame.drop(labels=['fnglwgt', 'education', 'native-country'], axis=1, inplace=True)
    data_frame.dropna(how='any', inplace=True)

    # convert salary to an integer
    salary = {' <=50K': 1, ' >50K': 0}
    data_frame['salary'] = data_frame['salary'].map(salary).astype(int)

    # convert sex numeric weight
    sex_count = data_frame['sex'].value_counts(normalize=True)
    sex = {' Male': sex_count[0], ' Female': sex_count[1]}
    data_frame['sex'] = data_frame['sex'].map(sex).astype(float)

    # convert workclass numeric weight
    work_count = data_frame['workclass'].value_counts(normalize=True)
    workclass = {' Private': work_count[0], ' Self-emp-not-inc': work_count[1], ' Local-gov': work_count[2], ' ?': work_count[3], ' State-gov': work_count[4], ' Self-emp-inc': work_count[5],
                 ' Federal-gov': work_count[6], ' Without-pay': work_count[7], 'Never-worked': work_count[8]}
    data_frame['workclass'] = data_frame['workclass'].map(workclass).astype(float)

    # convert marital-status numeric weight
    maritial_count = data_frame['marital-status'].value_counts(normalize=True)
    marital_status = {' Married-civ-spouse': maritial_count[0], ' Never-married': maritial_count[1],
                      ' Divorced': maritial_count[2], ' Separated': maritial_count[3], ' Widowed': maritial_count[4],
                      ' Married-spouse-absent': maritial_count[5], ' Married-AF-spouse': maritial_count[6]}
    data_frame['marital-status'] = data_frame['marital-status'].map(marital_status).astype(float)

    # convert occupation numeric weight
    occupation_count = data_frame['occupation'].value_counts(normalize=True)
    # print(occupation_count)
    occupation = {' Prof-specialty': occupation_count[0], ' Craft-repair': occupation_count[1], ' Exec-managerial': occupation_count[2], ' Adm-clerical': occupation_count[3], ' Sales': occupation_count[4],
                  ' Other-service': occupation_count[5], ' Machine-op-inspct': occupation_count[6], ' ?': occupation_count[7], ' Transport-moving': occupation_count[8],
                  ' Handlers-cleaners': occupation_count[9], ' Farming-fishing': occupation_count[10], ' Tech-support': occupation_count[11],
                  ' Protective-serv': occupation_count[12], ' Priv-house-serv': occupation_count[13], ' Armed-Forces': occupation_count[14]}
    data_frame['occupation'] = data_frame['occupation'].map(occupation).astype(float)

    # convert relationship numeric weight
    relationship_count = data_frame['relationship'].value_counts(normalize=True)
    relationship = {' Husband': relationship_count[0], ' Not-in-family': relationship_count[1], ' Own-child': relationship_count[2], ' Unmarried': relationship_count[3], ' Wife': relationship_count[4],
                    ' Other-relative': relationship_count[5]}
    data_frame['relationship'] = data_frame['relationship'].map(relationship).astype(float)

    # convert race numeric weight
    race_count = data_frame['race'].value_counts(normalize=True)
    race = {' White': race_count[0], ' Black': race_count[1], ' Asian-Pac-Islander': race_count[2], ' Amer-Indian-Eskimo': race_count[3], ' Other': race_count[4]}
    data_frame['race'] = data_frame['race'].map(race).astype(float)
    # print(data_frame['race'])

    cleaned_dataset = data_frame.values.tolist()

    return tuple(cleaned_dataset)

# print(get_data(DATA_URL))


def create_classifier(training_dataset):

    '''For each record we average the values for each attribute in a list of positive results and, separately, a
    list of negative results. The positive and negative averages are then averaged against each other to
    compute midpoint values. These will be used to compare each attribute in a record and assign it a status -
    negative or positive. The overall result is the greater of the number of the positive / negative status values.
    '''

    positive_attributes = [0] * 12
    negative_attributes = [0] * 12
    positive_count = 0
    negative_count = 0
    classifier_mid_points = [0] * 12

    # Compute the totals for each factor
    for record in training_dataset:
        print(record)
        if record[-1] == float(1):
            positive_count += 1
            for attribute in range(len(record[1:-1])):
                positive_attributes[attribute] += record[attribute + 1]
        elif record[-1] == float(0):
            negative_count += 1
            for attribute in range(len(record[1:-1])):
                negative_attributes[attribute] += record[attribute + 1]

    # Compute the average values for each factor
    for attribute in range(len(positive_attributes)):
        positive_attributes[attribute] = positive_attributes[attribute] / positive_count
    for attribute in range(len(negative_attributes)):
        negative_attributes[attribute] = negative_attributes[attribute] / negative_count

    # Compute the midpoints - the average of the benign & malignant factors in each case
    for attribute in range(len(classifier_mid_points)):
        classifier_mid_points[attribute] = (positive_attributes[attribute] + negative_attributes[attribute]) / 2

    print(f"Classifier values\n{'-' * 50}")
    for item in classifier_mid_points:
        print(f"{item:.4f}", end=", ")
    print("\n")

    return tuple(classifier_mid_points)


def test_classifier(testing_dataset, classifier_mid_points):
    """
    We apply the classifier list against each record in the test set. We compare each attribute against its
    equivalent value in the classifier list. Based on this, the attribute gets a status - 0 or 1. The count of
    the status values for a record determines the result.
    We also plot a bar chart to show a visual of the results.
    """

    false_count = 0
    true_count = 0
    total_count = 0

    temp_result_list = [''] * 12
    for record in testing_dataset:
        temp_result_list[0] = record[0]
        for attribute in range(len(record[1:-1])):
            if record[attribute + 1] < classifier_mid_points[attribute]:
                temp_result_list[attribute + 1] = float(0)
            else:
                temp_result_list[attribute + 1] = float(1)

        if temp_result_list.count(float(0)) >= 10:
            temp_result_list[-1] = float(0)
        else:
            temp_result_list[-1] = float(1)

        print(temp_result_list, end=' ')
        total_count += 1
        if record[-1] == temp_result_list[-1]:
            true_count += 1
            print("CORRECT")
        else:
            false_count += 1
            print("FALSE")

    print(f"\nCORRECT: {true_count}, {true_count / total_count:.2%},  INCORRECT: {false_count}, "
          f"{false_count / total_count:.2%},  TOTAL COUNT: {total_count}")

    # plot the test results in a bar chart
    left = [1, 2, 3]
    height = [true_count, false_count, total_count]
    tick_label = ['Correct', 'Incorrect', 'Total Count']
    plt.bar(left, height, tick_label=tick_label, width=0.8, color=['green', 'red', 'blue'])
    plt.xlabel('Tests')
    plt.ylabel('Count')
    plt.title('Classifier test results')
    plt.show()


def main():

    # Make a tuple of tuples from the raw data (a spreadsheet-like 2D array)
    cleaned_dataset = get_data(DATA_URL)

    # Break out our dataset into a training and test sets where the training set has a number of records determined
    # by the PERCENT value. The test set has the remaining records.
    training_dataset = cleaned_dataset[:int(len(cleaned_dataset) * PERCENT / 100)]
    test_dataset = cleaned_dataset[int(len(cleaned_dataset) * PERCENT / 100):]
    # print(training_dataset)
    # Create the classifier values
    classifier_mid_points = create_classifier(training_dataset)

    # Apply classifier against test file.
    # Given that we know the outcome for each test record we can verify the classifier
    test_classifier(test_dataset, classifier_mid_points)


if __name__ == "__main__":
    main()
