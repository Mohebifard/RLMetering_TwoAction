This project includes a code for deep RL-based traffic metering for a simple network of four intersections with one-way streets.
To train the controller run Training.py, to test the trained model and see the results run RunFinalModel.py

The project includes the following files:
- CTM: Transportation simulation using Cell Transmission Mdoel
- Environment: Definiton of the environment of RL 
- LPOptimization: A Linear progrmaming model as an upper bound for the problem (see: https://www.sciencedirect.com/science/article/pii/S0968090X18305795) 
- OneWay: The information of the CTM represented netwotk of the case study
