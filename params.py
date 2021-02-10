
class Params:
    def __init__(self, number_we_are_interested_in, x_train, y_train, y_val, x_val, train_text, test_text, target, target_test, cv):
        self.Number_we_are_interested_in = number_we_are_interested_in
        self.X_train = x_train
        self.Y_train = y_train
        self.Y_val = y_val
        self.X_val = x_val
        self.Train_text = train_text
        self.Test_text = test_text
        self.Target = target
        self.Target_test = target_test
        self.Cv = cv