X = importdata('../data/mog/3gaussian.txt');
init = 3
[label, model, llh] = emgm(X', init);
