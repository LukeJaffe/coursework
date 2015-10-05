import numpy as np
np.set_printoptions(precision=2)

X = np.eye(8)

hidden = []
for i in range(3):
    hidden.append([np.random.random(8), np.random.random()])

final = []
for i in range(8):
    final.append([np.random.random(3), np.random.random()])

l = 0.00001
it = 100000
for i in range(it):
    for x in X:
        # Propagate the inputs to hidden layer
        hidden_outputs = []
        for w,b in hidden:
            I = np.dot(w, x) + b
            O = 1.0 / (1.0 + np.exp(-I))
            hidden_outputs.append(O)
        hidden_outputs = np.array(hidden_outputs)
        # Propagate hidden outputs to final layer
        final_outputs = []
        for w,b in final:
            I = np.dot(w, hidden_outputs) + b
            O = 1.0 / (1.0 + np.exp(-I))
            final_outputs.append(O)
        final_outputs = np.array(final_outputs)
        # Backpropagate ouputs to final layer
        final_errors = []
        for j,Oj in enumerate(final_outputs):
            err = Oj*(1.0 - Oj)*(x[j] - Oj)
            final[j][0] += l*err*Oj
            final[j][1] += l*err
            final_errors.append(err)
        final_errors = np.array(final_errors)
        # Backpropagate final error to hidden layer
        for j,Oj in enumerate(hidden_outputs):
            err = Oj*(1.0 - Oj)*np.dot(final_errors, hidden[j][0])
            hidden[j][0] += l*err*Oj
            hidden[j][1] += l*err
        # Print results
        if i%1000 == 0:
            print final_outputs, x
            #print hidden_outputs
    if i%1000 == 0:
        print
